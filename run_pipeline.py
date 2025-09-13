import os
import sys
import json
import time
import datetime as dt
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

from src import data_collection as dc
from src import validation as val
from src import preprocessing as prep
from src import evaluation as evals


FIG_DIR = os.path.join('results', 'figures')
TAB_DIR = os.path.join('results', 'tables')
PROC_DIR = os.path.join('data', 'processed')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)


def summarize_validation(panel: pd.DataFrame) -> pd.DataFrame:
    checked = panel.apply(val.validate_row, axis=1)
    issues = checked['qa_notes'].fillna('')
    flags = issues != ''
    frac_flagged = float(flags.mean()) if len(flags) else 0.0
    issue_counts = Counter([i for s in issues[issues != ''] for i in s.split(';') if i])
    out = pd.DataFrame({
        'metric': list(issue_counts.keys()) + ['_any_flagged_fraction'],
        'value': list(issue_counts.values()) + [frac_flagged]
    })
    return out


def make_baseline_predictions(y_log: pd.Series, dates: pd.Series, outer_splits=5, test_size=14):
    tscv = TimeSeriesSplit(n_splits=outer_splits, test_size=test_size)
    preds = {
        'persistence': [],
        'ma7': [],
    }
    idx_all = []
    for train_idx, test_idx in tscv.split(y_log):
        train_y = y_log.iloc[train_idx]
        test_y = y_log.iloc[test_idx]
        # persistence: y_t = y_{t-1}
        pers = pd.Series(index=test_y.index, dtype=float)
        for i in test_y.index:
            if i - 1 in y_log.index:
                pers[i] = y_log.loc[i - 1]
            else:
                pers[i] = train_y.iloc[-1]
        # 7-day MA on log-scale over available history up to t-1
        ma7 = pd.Series(index=test_y.index, dtype=float)
        full_hist = y_log.copy()
        for i in test_y.index:
            window = full_hist.loc[:i-1].tail(7)
            if len(window) > 0:
                ma7[i] = window.mean()
            else:
                ma7[i] = train_y.mean()
        preds['persistence'].append(pers)
        preds['ma7'].append(ma7)
        idx_all.extend(list(test_idx))
    # Concatenate preserves original index order
    preds = {k: pd.concat(v).sort_index() for k, v in preds.items()}
    return preds


def nested_cv_predictions(X: pd.DataFrame, y: pd.Series, model_name: str, outer_splits=5, test_size=14):
    """Run nested CV producing out-of-fold predictions and fold best params.

    Returns: preds (Series aligned to y index), best_params_per_fold (list), models_for_inspection (list of fitted models)
    """
    if model_name == 'ElasticNet':
        base = Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('est', ElasticNet(max_iter=10000, random_state=42))
        ])
        param_grid = {
            'est__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
            'est__l1_ratio': [0.1, 0.5, 0.9],
        }
    elif model_name == 'RandomForest':
        base = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [200, 500],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
        }
    else:
        raise ValueError('Unsupported model_name')

    outer = TimeSeriesSplit(n_splits=outer_splits, test_size=test_size)
    oof_pred = pd.Series(index=y.index, dtype=float)
    best_params = []
    fitted_models = []
    for tr_idx, te_idx in outer.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        inner = TimeSeriesSplit(n_splits=3, test_size=7)
        gcv = GridSearchCV(base, param_grid, cv=inner, scoring='neg_mean_squared_error', n_jobs=-1)
        gcv.fit(X_tr, y_tr)
        best = gcv.best_estimator_
        best_params.append(gcv.best_params_ if hasattr(gcv, 'best_params_') else {})
        yhat = best.predict(X_te)
        oof_pred.iloc[te_idx] = yhat
        try:
            fitted_models.append(best)
        except Exception:
            pass
    return oof_pred.dropna(), best_params, fitted_models


def metrics_from_oof(y_log: pd.Series, yhat_log: pd.Series) -> dict:
    mask = yhat_log.index.intersection(y_log.index)
    y_true_log = y_log.loc[mask].values
    y_pred_log = yhat_log.loc[mask].values
    return evals.compute_metrics(y_true_log, y_pred_log)


def main():
    load_dotenv(override=False)
    # Date range: use last 150 days to ensure >=90-day coverage
    end = dt.date.today()
    start = end - dt.timedelta(days=150)

    print('Building panel from APIs...')
    panel = dc.build_panel(start, end)

    # Save raw-shaped panel before validation/preprocessing
    csv_path = os.path.join(PROC_DIR, 'panel_raw.csv')
    panel.to_csv(csv_path, index=False)

    print('Validating panel...')
    qa = summarize_validation(panel)
    qa.to_csv(os.path.join(TAB_DIR, 'validation_summary.csv'), index=False)

    # Preprocess and feature engineering
    print('Engineering features...')
    panel_feat = prep.compute_features(panel)
    # Save cleaned panel
    parquet_path = os.path.join(PROC_DIR, 'panel.parquet')
    panel_feat.to_parquet(parquet_path, index=False)

    # Modeling per protocol
    protocols = sorted(panel_feat['protocol'].dropna().unique())
    all_records = []
    decision_records = []
    top_features_records = []

    for proto in protocols:
        pdf = panel_feat[panel_feat['protocol'] == proto].sort_values('date').reset_index(drop=True)
        # Require at least 90 days with target
        if pdf['market_cap_circulating'].notna().sum() < 90:
            continue
        X_full, y_full = prep.build_feature_matrix(pdf, target_col='market_cap_circulating')
        # Drop protocol/date columns from X for modeling; keep for plotting
        meta = X_full[['protocol', 'date']]
        X = X_full.drop(columns=['protocol', 'date'])
        y = y_full
        # Align indices 0..n-1 for CV
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        meta = meta.reset_index(drop=True)

        # Baselines
        base_preds = make_baseline_predictions(y, meta['date'])
        base_metrics = {k: metrics_from_oof(y, v) for k, v in base_preds.items()}
        for bname, mets in base_metrics.items():
            rec = {'protocol': proto, 'model': bname, **mets}
            all_records.append(rec)

        # Learned models: ElasticNet and RandomForest
        for model_name in ['ElasticNet', 'RandomForest']:
            yhat, best_params_list, fitted = nested_cv_predictions(X, y, model_name=model_name)
            mets = metrics_from_oof(y, yhat)
            rec = {'protocol': proto, 'model': model_name, **mets}
            all_records.append(rec)
            # Capture decision info
            decision_records.append({
                'protocol': proto,
                'model': model_name,
                'best_params_last_fold': json.dumps(best_params_list[-1] if len(best_params_list) else {}),
            })

            # Train on full series for feature importances/coefs
            if model_name == 'ElasticNet':
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('est', ElasticNet(max_iter=10000, random_state=42))
                ])
                # Use the most common params across folds if available
                if best_params_list:
                    # Flatten keys like 'est__alpha'
                    agg = defaultdict(list)
                    for d in best_params_list:
                        for k, v in d.items():
                            agg[k].append(v)
                    best = {k: Counter(vs).most_common(1)[0][0] for k, vs in agg.items()}
                    pipe.set_params(**best)
                pipe.fit(X, y)
                coefs = pipe.named_steps['est'].coef_
                feats = X.columns
                abs_top = np.argsort(np.abs(coefs))[::-1][:10]
                for idx in abs_top:
                    top_features_records.append({'protocol': proto, 'model': 'ElasticNet', 'feature': feats[idx], 'importance': float(coefs[idx])})
            elif model_name == 'RandomForest':
                rf = RandomForestRegressor(random_state=42, n_jobs=-1)
                if best_params_list:
                    # Choose most common params
                    agg = defaultdict(list)
                    for d in best_params_list:
                        for k, v in d.items():
                            agg[k].append(v)
                    best = {k: Counter(vs).most_common(1)[0][0] for k, vs in agg.items()}
                    rf.set_params(**best)
                rf.fit(X, y)
                importances = rf.feature_importances_
                feats = X.columns
                top_idx = np.argsort(importances)[::-1][:10]
                for idx in top_idx:
                    top_features_records.append({'protocol': proto, 'model': 'RandomForest', 'feature': feats[idx], 'importance': float(importances[idx])})

            # Figures: time-series and residuals for this protocol using ElasticNet (if available) else RF; here use last yhat
            # Recompute with ElasticNet preference for plotting
        # Choose plotting model predictions: prioritize ElasticNet
        # For simplicity, plot persistence baseline vs. truth
        # Build time series plot
        y_usd = np.exp(y.values)
        # Use persistence baseline predictions
        yhat_log = base_preds['ma7'].sort_index().values
        yhat_usd = np.exp(yhat_log)
        # Align arrays
        mask = ~np.isnan(yhat_usd)
        dates = meta.loc[mask, 'date']
        plt.figure(figsize=(10, 4))
        plt.plot(meta['date'], y_usd, label='True (USD)')
        plt.plot(dates, yhat_usd[mask], label='Baseline MA7 (USD)')
        plt.title(f"{proto}: Market Cap vs Baseline")
        plt.xlabel('Date'); plt.ylabel('Market Cap (USD)'); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{proto}_timeseries.png"), dpi=150)
        plt.close()

        # Residuals plot for baseline MA7
        res = y_usd[mask] - yhat_usd[mask]
        plt.figure(figsize=(6, 4))
        sns.histplot(res, bins=30, kde=False)
        plt.title(f"{proto}: Residuals (MA7 baseline)")
        plt.xlabel('Residual (USD)')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{proto}_residuals.png"), dpi=150)
        plt.close()

    # Save metrics
    metrics_df = pd.DataFrame(all_records)
    metrics_df.to_csv(os.path.join(TAB_DIR, 'metrics_per_protocol.csv'), index=False)
    decision_df = pd.DataFrame(decision_records)
    decision_df.to_csv(os.path.join(TAB_DIR, 'decision_matrix.csv'), index=False)
    topf_df = pd.DataFrame(top_features_records)
    topf_df.to_csv(os.path.join(TAB_DIR, 'top_features.csv'), index=False)

    # Overall metrics: average across protocols by model
    if not metrics_df.empty:
        overall = (metrics_df.groupby('model')[['rmse_usd', 'mae_usd', 'mape', 'r2', 'directional_accuracy']]
                   .mean().reset_index())
        overall.to_csv(os.path.join(TAB_DIR, 'metrics_overall.csv'), index=False)

    # Print summary tree
    print('\nArtifacts written:')
    for root in ['data/processed', 'results/tables', 'results/figures']:
        print(f"- {root}")
        for p in sorted(os.listdir(root)):
            print(f"  - {root}/{p}")


if __name__ == '__main__':
    main()

