"""
Validation utilities for the DEX valuation study.
This module includes checks for data quality, cross-source consistency, and logical constraints.
"""
import pandas as pd


def cross_source_diff(a: float, b: float, threshold: float = 0.05) -> bool:
    """Return True if the relative difference between two values exceeds threshold."""
    if a is None or b is None:
        return False
    return abs(a - b) / max(abs(a), abs(b)) > threshold


def validate_row(row: pd.Series) -> pd.Series:
    """Apply logical sanity checks to a single row and flag issues."""
    issues = []
    rev = row.get('revenue_24h')
    fees = row.get('fees_24h')
    tvl = row.get('tvl')
    vol = row.get('volume_24h')
    users = row.get('active_users_24h')
    try:
        if pd.notnull(rev) and pd.notnull(fees) and float(rev) > float(fees):
            issues.append('revenue_gt_fees')
    except Exception:
        pass
    try:
        if pd.notnull(tvl) and float(tvl) < 0:
            issues.append('tvl_negative')
    except Exception:
        pass
    try:
        if pd.notnull(vol) and float(vol) < 0:
            issues.append('volume_negative')
    except Exception:
        pass
    try:
        if pd.notnull(users) and float(users) < 0:
            issues.append('users_negative')
    except Exception:
        pass
    row['qa_notes'] = ';'.join(issues)
    return row
