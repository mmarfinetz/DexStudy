"""
Modeling functions for the DEX valuation study.
Includes baseline linear models, tree-based models, and support vector regressors with nested time series cross-validation.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def nested_cv_train(X: pd.DataFrame, y: pd.Series, model, param_grid: Dict, outer_splits: int = 5, inner_splits: int = 3, test_size: int = 7) -> Tuple[float, Dict]:
    """Perform nested cross-validation for time series.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        model: Estimator class (not yet instantiated).
        param_grid: Hyperparameter grid for inner CV.
        outer_splits: Number of outer CV splits.
        inner_splits: Number of inner CV splits.
        test_size: Number of observations in each outer test set.

    Returns:
        Mean test score across folds and best parameters.
    """
    outer_tscv = TimeSeriesSplit(n_splits=outer_splits, test_size=test_size)
    test_scores = []
    best_params_list = []
    for train_index, test_index in outer_tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Inner CV
        inner_tscv = TimeSeriesSplit(n_splits=inner_splits, test_size=3)
        gsearch = GridSearchCV(model(), param_grid, cv=inner_tscv, scoring='neg_mean_squared_error')
        gsearch.fit(X_train, y_train)
        best_model = gsearch.best_estimator_
        y_pred = best_model.predict(X_test)
        test_scores.append(np.sqrt(np.mean((y_test - y_pred) ** 2)))
        best_params_list.append(gsearch.best_params_)
    return float(np.mean(test_scores)), best_params_list[-1]  # return last best params
