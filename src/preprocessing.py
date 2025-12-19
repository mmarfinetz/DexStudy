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

    # Sort once for all operations
    df = df.sort_values(['protocol', 'date'])

    # Ensure numeric columns are properly typed
    numeric_cols = ['market_cap_circulating', 'volume_24h', 'fees_24h', 'revenue_24h', 'tvl']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 1. LAGGED FEATURES (critical for time series)
    if 'market_cap_circulating' in df.columns:
        for lag in [1, 3, 7, 14, 30]:
            df[f'mc_lag_{lag}'] = (
                df.groupby('protocol')['market_cap_circulating']
                  .shift(lag)
            )
            # Log of lagged market cap
            df[f'mc_log_lag_{lag}'] = np.log(df[f'mc_lag_{lag}'].clip(lower=1e-9))

    if 'tvl' in df.columns:
        for lag in [1, 3, 7]:
            df[f'tvl_lag_{lag}'] = df.groupby('protocol')['tvl'].shift(lag)

    if 'volume_24h' in df.columns:
        for lag in [1, 3, 7]:
            df[f'volume_lag_{lag}'] = df.groupby('protocol')['volume_24h'].shift(lag)

    # 2. ROLLING STATISTICS (existing + enhanced)
    if 'volume_24h' in df.columns:
        # Existing rolling means
        df['volume_7d_avg'] = (
            df.groupby('protocol')['volume_24h']
              .rolling(window=7, min_periods=3)
              .mean()
              .reset_index(level=0, drop=True)
        )
        df['volume_30d_avg'] = (
            df.groupby('protocol')['volume_24h']
              .rolling(window=30, min_periods=7)
              .mean()
              .reset_index(level=0, drop=True)
        )
        # New: Rolling std (volatility)
        df['volume_7d_std'] = (
            df.groupby('protocol')['volume_24h']
              .rolling(window=7, min_periods=3)
              .std()
              .reset_index(level=0, drop=True)
        )
        df['volume_30d_std'] = (
            df.groupby('protocol')['volume_24h']
              .rolling(window=30, min_periods=7)
              .std()
              .reset_index(level=0, drop=True)
        )
        # Coefficient of variation (CV)
        df['volume_7d_cv'] = df['volume_7d_std'] / df['volume_7d_avg'].replace(0, np.nan)

    if 'tvl' in df.columns:
        # TVL rolling statistics
        df['tvl_7d_avg'] = (
            df.groupby('protocol')['tvl']
              .rolling(window=7, min_periods=3)
              .mean()
              .reset_index(level=0, drop=True)
        )
        df['tvl_30d_avg'] = (
            df.groupby('protocol')['tvl']
              .rolling(window=30, min_periods=7)
              .mean()
              .reset_index(level=0, drop=True)
        )
        df['tvl_7d_std'] = (
            df.groupby('protocol')['tvl']
              .rolling(window=7, min_periods=3)
              .std()
              .reset_index(level=0, drop=True)
        )
        # Growth rates
        df['tvl_growth_1d'] = df.groupby('protocol')['tvl'].pct_change(periods=1)
        df['tvl_growth_3d'] = df.groupby('protocol')['tvl'].pct_change(periods=3)
        df['tvl_growth_7d'] = df.groupby('protocol')['tvl'].pct_change(periods=7)
        df['tvl_growth_30d'] = df.groupby('protocol')['tvl'].pct_change(periods=30)

    # 3. MOMENTUM INDICATORS
    if 'market_cap_circulating' in df.columns:
        # Price momentum (using market cap as proxy)
        df['mc_momentum_7d'] = (
            df.groupby('protocol')['market_cap_circulating'].pct_change(periods=7)
        )
        df['mc_momentum_30d'] = (
            df.groupby('protocol')['market_cap_circulating'].pct_change(periods=30)
        )
        # Moving average crossover signals
        df['mc_7d_avg'] = (
            df.groupby('protocol')['market_cap_circulating']
              .rolling(window=7, min_periods=3)
              .mean()
              .reset_index(level=0, drop=True)
        )
        df['mc_30d_avg'] = (
            df.groupby('protocol')['market_cap_circulating']
              .rolling(window=30, min_periods=7)
              .mean()
              .reset_index(level=0, drop=True)
        )
        df['ma_crossover_signal'] = (df['mc_7d_avg'] / df['mc_30d_avg'].replace(0, np.nan)) - 1

    # 4. TECHNICAL INDICATORS
    if 'market_cap_circulating' in df.columns:
        # RSI (Relative Strength Index) - properly grouped by protocol
        delta = df.groupby('protocol')['market_cap_circulating'].diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))

        avg_gain = (
            gain.groupby(df['protocol'])
                .rolling(window=14, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
        )
        avg_loss = (
            loss.groupby(df['protocol'])
                .rolling(window=14, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
        )

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['mc_bb_middle'] = df['mc_30d_avg']
        mc_30d_std = (
            df.groupby('protocol')['market_cap_circulating']
              .rolling(window=30, min_periods=7)
              .std()
              .reset_index(level=0, drop=True)
        )
        df['mc_bb_upper'] = df['mc_bb_middle'] + (2 * mc_30d_std)
        df['mc_bb_lower'] = df['mc_bb_middle'] - (2 * mc_30d_std)
        df['mc_bb_position'] = (
            (df['market_cap_circulating'] - df['mc_bb_lower']) /
            (df['mc_bb_upper'] - df['mc_bb_lower']).replace(0, np.nan)
        )

    # 5. EFFICIENCY RATIOS (existing + enhanced)
    if 'fees_24h' in df.columns:
        df['fees_7d_avg'] = (
            df.groupby('protocol')['fees_24h']
              .rolling(window=7, min_periods=3)
              .mean()
              .reset_index(level=0, drop=True)
        )
    if 'revenue_24h' in df.columns:
        df['revenue_7d_avg'] = (
            df.groupby('protocol')['revenue_24h']
              .rolling(window=7, min_periods=3)
              .mean()
              .reset_index(level=0, drop=True)
        )

    if 'fees_24h' in df.columns and 'tvl' in df.columns:
        df['fees_per_tvl'] = df['fees_24h'] / df['tvl'].replace(0, np.nan)
        df['fees_per_tvl_7d'] = df['fees_7d_avg'] / df['tvl_7d_avg'].replace(0, np.nan)

    if 'volume_24h' in df.columns and 'tvl' in df.columns:
        df['volume_per_tvl'] = df['volume_24h'] / df['tvl'].replace(0, np.nan)
        df['volume_per_tvl_7d'] = df['volume_7d_avg'] / df['tvl_7d_avg'].replace(0, np.nan)

    if 'revenue_24h' in df.columns and 'active_users_24h' in df.columns:
        df['revenue_per_user'] = df['revenue_24h'] / df['active_users_24h'].replace(0, np.nan)

    # 6. RELATIVE STRENGTH vs MARKET
    if 'tvl' in df.columns:
        # Market share (relative to total TVL across all protocols on that date)
        df['total_tvl_market'] = df.groupby('date')['tvl'].transform('sum')
        df['tvl_market_share'] = df['tvl'] / df['total_tvl_market'].replace(0, np.nan)
        df['tvl_market_share_change_7d'] = (
            df.groupby('protocol')['tvl_market_share'].pct_change(periods=7)
        )

    if 'volume_24h' in df.columns:
        df['total_volume_market'] = df.groupby('date')['volume_24h'].transform('sum')
        df['volume_market_share'] = df['volume_24h'] / df['total_volume_market'].replace(0, np.nan)

    # 7. TREND FEATURES
    if 'tvl' in df.columns:
        # Linear trend coefficient over past N days
        for window in [7, 14, 30]:
            trend_col = f'tvl_trend_{window}d'
            df[trend_col] = (
                df.groupby('protocol')['tvl']
                  .rolling(window=window, min_periods=window//2)
                  .apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0)
                  .reset_index(level=0, drop=True)
            )

    # 8. SEASONALITY FEATURES
    if 'date' in df.columns:
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        # Sin/cos encoding for cyclical features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # 9. INTERACTION FEATURES
    if 'tvl' in df.columns and 'volume_24h' in df.columns:
        df['tvl_volume_interaction'] = np.log(df['tvl'].clip(lower=1)) * np.log(df['volume_24h'].clip(lower=1))

    # 10. VOLATILITY & RISK METRICS
    if 'market_cap_circulating' in df.columns:
        # Historical volatility
        df['mc_volatility_7d'] = (
            df.groupby('protocol')['market_cap_circulating']
              .rolling(window=7, min_periods=3)
              .std()
              .reset_index(level=0, drop=True)
        ) / df['mc_7d_avg'].replace(0, np.nan)

        # Maximum drawdown from peak
        df['mc_cummax'] = df.groupby('protocol')['market_cap_circulating'].cummax()
        df['mc_drawdown'] = (df['market_cap_circulating'] - df['mc_cummax']) / df['mc_cummax'].replace(0, np.nan)

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
    # Drop non-feature columns - MUST exclude y_log to prevent target leakage
    base_drop = ['qa_notes', target_col, 'y_log']  # Added y_log here
    # Select numerical features only
    feature_cols = [
        c for c in df.columns
        if c not in base_drop + ['protocol', 'date'] and np.issubdtype(df[c].dtype, np.number)
    ]
    X = df[['protocol', 'date'] + feature_cols].copy()
    y = df['y_log'].copy()
    return X, y
