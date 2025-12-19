#!/usr/bin/env python3
"""
Demo pipeline for DEX Valuation Study.
Generates synthetic data to demonstrate the analysis when APIs are unavailable.
"""
import os
import json
import datetime as dt
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

from src import validation as val
from src import preprocessing as prep
from src import evaluation as evals


FIG_DIR = os.path.join('results', 'figures')
TAB_DIR = os.path.join('results', 'tables')
PROC_DIR = os.path.join('data', 'processed')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)


def generate_synthetic_data(n_days=150):
    """Generate synthetic DEX protocol data for demonstration."""
    np.random.seed(42)

    protocols = [
        {'name': 'Uniswap', 'base_mc': 5e9, 'base_tvl': 5e9, 'base_volume': 1e9, 'volatility': 0.03},
        {'name': 'SushiSwap', 'base_mc': 500e6, 'base_tvl': 500e6, 'base_volume': 100e6, 'volatility': 0.05},
        {'name': 'Curve', 'base_mc': 800e6, 'base_tvl': 2e9, 'base_volume': 200e6, 'volatility': 0.04},
        {'name': 'Balancer', 'base_mc': 300e6, 'base_tvl': 1e9, 'base_volume': 50e6, 'volatility': 0.04},
        {'name': 'PancakeSwap', 'base_mc': 1.5e9, 'base_tvl': 2e9, 'base_volume': 300e6, 'volatility': 0.05},
    ]

    end_date = dt.date(2025, 9, 10)
    start_date = end_date - dt.timedelta(days=n_days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    rows = []
    for p in protocols:
        # Generate random walk for market cap
        returns = np.random.normal(0.001, p['volatility'], len(dates))
        mc_path = p['base_mc'] * np.cumprod(1 + returns)

        # TVL correlated with market cap but with different dynamics
        tvl_returns = 0.7 * returns + 0.3 * np.random.normal(0, p['volatility'] * 0.5, len(dates))
        tvl_path = p['base_tvl'] * np.cumprod(1 + tvl_returns)

        # Volume with weekly seasonality
        base_volume = p['base_volume']
        for i, date in enumerate(dates):
            dow = date.dayofweek
            weekend_factor = 0.6 if dow >= 5 else 1.0
            volume = base_volume * weekend_factor * np.exp(np.random.normal(0, 0.3))

            # Fees and revenue
            fee_rate = np.random.uniform(0.002, 0.005)
            fees = volume * fee_rate
            revenue = fees * np.random.uniform(0.1, 0.3)  # Protocol take rate

            rows.append({
                'protocol': p['name'],
                'date': date,
                'market_cap_circulating': mc_path[i],
                'volume_24h': volume,
                'fees_24h': fees,
                'revenue_24h': revenue,
                'tvl': tvl_path[i],
                'active_users_24h': np.random.randint(1000, 50000),
                'transactions_24h': np.random.randint(10000, 500000),
                'token_holders': None,
                'governance_proposals_30d': None,
                'token_distribution': None,
                'chain_deployment': len(protocols),
                'token_age_days': None,
            })

    panel = pd.DataFrame(rows)
    return panel


def summarize_validation(panel):
    if panel.empty:
        return pd.DataFrame({'metric': ['_any_flagged_fraction'], 'value': [0.0]})
    checked = panel.apply(val.validate_row, axis=1)
    issues = checked.get('qa_notes', pd.Series(index=checked.index, dtype=object)).fillna('')
    flags = issues != ''
    frac_flagged = float(flags.mean()) if len(flags) else 0.0
    issue_counts = Counter([i for s in issues[issues != ''] for i in s.split(';') if i])
    out = pd.DataFrame({
        'metric': list(issue_counts.keys()) + ['_any_flagged_fraction'],
        'value': list(issue_counts.values()) + [frac_flagged]
    })
    return out


def make_baseline_predictions(y_log, dates, outer_splits=5, test_size=14):
    tscv = TimeSeriesSplit(n_splits=outer_splits, test_size=test_size)
    preds = {'persistence': [], 'ma7': []}

    for train_idx, test_idx in tscv.split(y_log):
        train_y = y_log.iloc[train_idx]
        test_y = y_log.iloc[test_idx]

        pers = pd.Series(index=test_y.index, dtype=float)
        for i in test_y.index:
            if i - 1 in y_log.index:
                pers[i] = y_log.loc[i - 1]
            else:
                pers[i] = train_y.iloc[-1]

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

    preds = {k: pd.concat(v).sort_index() for k, v in preds.items()}
    return preds


def nested_cv_predictions(X, y, model_name, outer_splits=5, test_size=14):
    if model_name == 'ElasticNet':
        base = Pipeline([
            ('scaler', StandardScaler()),
            ('est', ElasticNet(max_iter=10000, random_state=42))
        ])
        param_grid = {
            'est__alpha': [1e-3, 1e-2, 1e-1, 1],
            'est__l1_ratio': [0.1, 0.5, 0.9],
        }
    elif model_name == 'RandomForest':
        base = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 20],
            'min_samples_split': [2, 5],
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
        fitted_models.append(best)

    return oof_pred.dropna(), best_params, fitted_models


def metrics_from_oof(y_log, yhat_log):
    mask = yhat_log.index.intersection(y_log.index)
    y_true_log = y_log.loc[mask].values
    y_pred_log = yhat_log.loc[mask].values
    return evals.compute_metrics(y_true_log, y_pred_log)


def main():
    print('=' * 60)
    print('DEX VALUATION STUDY - DEMO PIPELINE')
    print('=' * 60)

    # Generate synthetic data
    print('\n1. GENERATING SYNTHETIC DATA...')
    panel = generate_synthetic_data(n_days=150)

    # Save raw panel
    csv_path = os.path.join(PROC_DIR, 'panel_raw.csv')
    panel.to_csv(csv_path, index=False)
    print(f'   Raw panel saved: {panel.shape[0]} rows, {panel["protocol"].nunique()} protocols')

    # Validate
    print('\n2. VALIDATING DATA...')
    qa = summarize_validation(panel)
    qa.to_csv(os.path.join(TAB_DIR, 'validation_summary.csv'), index=False)
    print(f'   Flagged fraction: {qa[qa["metric"] == "_any_flagged_fraction"]["value"].values[0]:.2%}')

    # Feature engineering
    print('\n3. ENGINEERING FEATURES...')
    panel_feat = prep.compute_features(panel)
    parquet_path = os.path.join(PROC_DIR, 'panel.parquet')
    panel_feat.to_parquet(parquet_path, index=False)
    print(f'   Features created: {len(panel_feat.columns)} columns')

    # Modeling
    print('\n4. TRAINING MODELS...')
    protocols = sorted(panel_feat['protocol'].dropna().unique())
    all_records = []
    decision_records = []
    top_features_records = []

    for proto in protocols:
        pdf = panel_feat[panel_feat['protocol'] == proto].sort_values('date').reset_index(drop=True)

        if pdf['market_cap_circulating'].notna().sum() < 90:
            print(f'   Skipping {proto}: insufficient data')
            continue

        print(f'   Processing {proto}...')

        X_full, y_full = prep.build_feature_matrix(pdf, target_col='market_cap_circulating')
        meta = X_full[['protocol', 'date']]
        X = X_full.drop(columns=['protocol', 'date'])
        y = y_full

        X = X.ffill().fillna(0.0).reset_index(drop=True)
        y = y.reset_index(drop=True)
        meta = meta.reset_index(drop=True)

        # Baselines
        base_preds = make_baseline_predictions(y, meta['date'])
        base_metrics = {k: metrics_from_oof(y, v) for k, v in base_preds.items()}
        for bname, mets in base_metrics.items():
            rec = {'protocol': proto, 'model': bname, **mets}
            all_records.append(rec)

        # Learned models
        for model_name in ['ElasticNet', 'RandomForest']:
            yhat, best_params_list, fitted = nested_cv_predictions(X, y, model_name=model_name)
            mets = metrics_from_oof(y, yhat)
            rec = {'protocol': proto, 'model': model_name, **mets}
            all_records.append(rec)
            decision_records.append({
                'protocol': proto,
                'model': model_name,
                'best_params_last_fold': json.dumps(best_params_list[-1] if len(best_params_list) else {}),
            })

            # Extract feature importance
            if model_name == 'RandomForest' and fitted:
                rf = fitted[-1]
                importances = rf.feature_importances_
                feats = X.columns
                top_idx = np.argsort(importances)[::-1][:10]
                for idx in top_idx:
                    top_features_records.append({
                        'protocol': proto,
                        'model': 'RandomForest',
                        'feature': feats[idx],
                        'importance': float(importances[idx])
                    })

        # Plot time series
        y_usd = np.exp(y.values)
        yhat_series = base_preds['ma7'].reindex(y.index)
        yhat_log = yhat_series.values
        yhat_usd = np.exp(yhat_log)
        mask = ~np.isnan(yhat_usd)
        dates = meta.loc[mask, 'date']

        plt.figure(figsize=(10, 4))
        plt.plot(meta['date'], y_usd / 1e6, label='True (USD M)')
        plt.plot(dates, yhat_usd[mask] / 1e6, label='Baseline MA7 (USD M)', linestyle='--')
        plt.title(f"{proto}: Market Cap vs Baseline")
        plt.xlabel('Date')
        plt.ylabel('Market Cap (USD Millions)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{proto}_timeseries.png"), dpi=150)
        plt.close()

    # Save metrics
    metrics_df = pd.DataFrame(all_records)
    metrics_df.to_csv(os.path.join(TAB_DIR, 'metrics_per_protocol.csv'), index=False)
    decision_df = pd.DataFrame(decision_records)
    decision_df.to_csv(os.path.join(TAB_DIR, 'decision_matrix.csv'), index=False)
    topf_df = pd.DataFrame(top_features_records)
    topf_df.to_csv(os.path.join(TAB_DIR, 'top_features.csv'), index=False)

    # Overall metrics
    if not metrics_df.empty:
        overall = (metrics_df.groupby('model')[['rmse_usd', 'mae_usd', 'mape', 'r2', 'directional_accuracy']]
                   .mean().reset_index())
        overall.to_csv(os.path.join(TAB_DIR, 'metrics_overall.csv'), index=False)

    # Print results
    print('\n' + '=' * 60)
    print('STUDY RESULTS')
    print('=' * 60)

    print('\n=== OVERALL MODEL COMPARISON ===')
    print(overall.to_string(index=False))

    print('\n=== BEST MODEL BY METRIC ===')
    best_r2 = overall.loc[overall['r2'].idxmax()]
    best_mape = overall.loc[overall['mape'].idxmin()]
    print(f"Best R2: {best_r2['model']} (R2 = {best_r2['r2']:.4f})")
    print(f"Best MAPE: {best_mape['model']} (MAPE = {best_mape['mape']:.4f})")
    print(f"Best Dir. Accuracy: {overall.loc[overall['directional_accuracy'].idxmax()]['model']} ({overall['directional_accuracy'].max():.2%})")

    print('\n=== METRICS BY PROTOCOL ===')
    pivot = metrics_df.pivot(index='protocol', columns='model', values='r2').round(4)
    print(pivot.to_string())

    print('\n=== TOP 10 FEATURES (Random Forest) ===')
    if not topf_df.empty:
        top10 = topf_df.groupby('feature')['importance'].mean().sort_values(ascending=False).head(10)
        for feat, imp in top10.items():
            print(f"  {feat}: {imp:.4f}")

    print('\n=== ARTIFACTS WRITTEN ===')
    for root in ['data/processed', 'results/tables', 'results/figures']:
        print(f"- {root}")
        for p in sorted(os.listdir(root)):
            print(f"    {p}")

    print('\n' + '=' * 60)
    print('DEMO COMPLETE')
    print('=' * 60)


if __name__ == '__main__':
    main()
