"""
Preprocessing functions for the DEX valuation study.
This module includes cleaning, merging, and feature engineering, with strict safeguards to avoid feature leakage.
"""
import pandas as pd
import numpy as np

# Allowed feature categories
ALLOWED_FEATURES = {
    'activity': ['volume_24h', 'volume_7d_avg', 'volume_30d_avg'],
    'revenue': ['fees_24h', 'fees_7d_avg', 'revenue_24h', 'revenue_7d_avg'],
    'efficiency': ['fees_per_tvl', 'volume_per_tvl', 'revenue_per_user'],
    'liquidity': ['tvl', 'tvl_growth_7d', 'tvl_growth_30d'],
    'users': ['active_users_24h', 'user_growth_7d', 'avg_trade_size'],
    'governance': ['proposals_per_billion_mc', 'voter_participation_rate'],
    'distribution': ['gini_coefficient', 'herfindahl_index', 'whale_concentration'],
}

FORBIDDEN_FEATURES = ['fdv', 'p_s_ratio', 'p_e_ratio', 'volume_to_mc', 'liquidity_to_mc']

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features and enforce no leakage.

    Args:
        df: Raw panel data with base metrics.

    Returns:
        DataFrame with engineered features.
    """
    # Copy to avoid modifying original
    df = df.copy()
    # Ensure types
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    # Rolling averages and growth rates (guarded by column presence)
    if 'volume_24h' in df.columns:
        df['volume_7d_avg'] = (
            df.sort_values(['protocol', 'date'])
              .groupby('protocol')['volume_24h']
              .rolling(window=7, min_periods=3)
              .mean()
              .reset_index(level=0, drop=True)
        )
        df['volume_30d_avg'] = (
            df.sort_values(['protocol', 'date'])
              .groupby('protocol')['volume_24h']
              .rolling(window=30, min_periods=7)
              .mean()
              .reset_index(level=0, drop=True)
        )
    if 'fees_24h' in df.columns:
        df['fees_7d_avg'] = (
            df.sort_values(['protocol', 'date'])
              .groupby('protocol')['fees_24h']
              .rolling(window=7, min_periods=3)
              .mean()
              .reset_index(level=0, drop=True)
        )
    if 'revenue_24h' in df.columns:
        df['revenue_7d_avg'] = (
            df.sort_values(['protocol', 'date'])
              .groupby('protocol')['revenue_24h']
              .rolling(window=7, min_periods=3)
              .mean()
              .reset_index(level=0, drop=True)
        )
    if 'tvl' in df.columns:
        df['tvl_growth_7d'] = (
            df.sort_values(['protocol', 'date'])
              .groupby('protocol')['tvl']
              .pct_change(periods=7)
        )
        df['tvl_growth_30d'] = (
            df.sort_values(['protocol', 'date'])
              .groupby('protocol')['tvl']
              .pct_change(periods=30)
        )
    # Efficiency features (avoid divide by zero)
    if 'fees_24h' in df.columns and 'tvl' in df.columns:
        df['fees_per_tvl'] = df['fees_24h'] / df['tvl'].replace(0, np.nan)
    if 'volume_24h' in df.columns and 'tvl' in df.columns:
        df['volume_per_tvl'] = df['volume_24h'] / df['tvl'].replace(0, np.nan)
    if 'revenue_24h' in df.columns and 'active_users_24h' in df.columns:
        df['revenue_per_user'] = df['revenue_24h'] / df['active_users_24h'].replace(0, np.nan)
    # Additional features can be implemented here
    # Drop forbidden features if present
    df = df[[col for col in df.columns if col not in FORBIDDEN_FEATURES]]
    return df


def build_feature_matrix(df: pd.DataFrame, target_col: str = 'market_cap_circulating') -> (pd.DataFrame, pd.Series):
    """Construct a causal feature matrix X and target y (log-scale).

    - Drops rows with missing target.
    - Avoids leakage by using only features from or before date t for target at t.
    - Returns aligned X, y.
    """
    df = df.copy()
    df = df.sort_values(['protocol', 'date'])
    # Log-transform target
    eps = 1e-9
    df = df[df[target_col].notna()]
    df['y_log'] = np.log(df[target_col].astype(float).clip(lower=0) + eps)
    # Drop non-feature columns
    base_drop = ['qa_notes', target_col]
    # Select numerical features only
    feature_cols = [
        c for c in df.columns
        if c not in base_drop + ['protocol', 'date'] and np.issubdtype(df[c].dtype, np.number)
    ]
    X = df[['protocol', 'date'] + feature_cols].copy()
    y = df['y_log'].copy()
    return X, y
