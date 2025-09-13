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
    if row['revenue_24h'] > row['fees_24h']:
        issues.append('revenue_gt_fees')
    if row['tvl'] < 0:
        issues.append('tvl_negative')
    if row['volume_24h'] < 0:
        issues.append('volume_negative')
    if row['active_users_24h'] < 0:
        issues.append('users_negative')
    row['qa_notes'] = ';'.join(issues)
    return row

