
"""
Shares Outstanding Outlier Adjustment
=====================================

Custom calculation functions for KaxaNuk Data Curator that detect and correct
transient spikes in shares outstanding time series.

Output Columns
--------------
c_weighted_average_basic_shares_outstanding_outlier_adjusted
    Outlier-corrected basic shares outstanding.

c_weighted_average_diluted_shares_outstanding_outlier_adjusted
    Outlier-corrected diluted shares outstanding.

Algorithm Overview
------------------
The algorithm identifies **transient spikes**: values that deviate significantly
from the series trend and subsequently revert. These anomalies are replaced with
forward-filled values from before the spike began.

Detection Pipeline
------------------
Outlier detection is performed in four sequential passes, each operating on the
corrected output of the previous pass:

    **Pass 1 - Extreme spikes (>100% change)**
        Detects dramatic jumps that are almost certainly data errors.

    **Pass 2 - Medium spikes (>50% change)**
        Detects significant deviations after extreme spikes are removed.

    **Pass 3 - Small spikes (>35% change)**
        Detects moderate anomalies in the cleaned series.

    **Pass 4 - Rolling median outliers (>40% deviation)**
        Catches isolated outliers that don't fit the reversion pattern.

Reversion Requirement
---------------------
For passes 1-3, a spike is only corrected if the series **reverts** to near
its pre-spike level within the lookahead window. This distinguishes between:

    - **Transient spikes** (corrected): Data errors that revert
    - **Permanent shifts** (preserved): Legitimate events like stock issuances

Configuration
-------------
MAX_LOOKAHEAD_DAYS : int
    Maximum trading days to search for reversion (~2.5 years).

ROLLING_WINDOW_DAYS : int
    Window size for rolling median calculation (~3 months).

ROLLING_DEVIATION_THRESHOLD : float
    Maximum allowed deviation from rolling median (40%).
"""

import numpy as np
import pandas as pd
from kaxanuk.data_curator.modules.data_column import DataColumn


# =============================================================================
# Configuration
# =============================================================================

MAX_LOOKAHEAD_DAYS = 600
ROLLING_WINDOW_DAYS = 63
ROLLING_DEVIATION_THRESHOLD = 0.40


# =============================================================================
# Percentage Change Calculation
# =============================================================================

def calculate_pct_change(series: pd.Series) -> pd.Series:
    """
    Calculate absolute period-to-period percentage change.

    Parameters
    ----------
    series : pd.Series
        Time series of values.

    Returns
    -------
    pd.Series
        Absolute percentage change from previous value.
        First value is NaN (no previous value exists).
    """
    previous = series.shift(1)
    return (series - previous).abs() / previous.abs()


# =============================================================================
# Spike Detection
# =============================================================================

def find_spike_start_indices(pct_change: pd.Series, threshold: float) -> pd.Index:
    """
    Identify indices where percentage change exceeds threshold.

    Parameters
    ----------
    pct_change : pd.Series
        Period-to-period percentage changes.
    threshold : float
        Minimum change to flag as potential spike (e.g., 0.50 for 50%).

    Returns
    -------
    pd.Index
        Indices where spikes potentially begin.
    """
    exceeds_threshold = pct_change > threshold
    return pct_change[exceeds_threshold].index


def find_reversion_index(
    series: pd.Series,
    spike_start_position: int,
    pre_spike_value: float,
    threshold: float,
    max_lookahead: int,
) -> int | None:
    """
    Search for the point where series reverts to pre-spike level.

    Parameters
    ----------
    series : pd.Series
        Time series of values.
    spike_start_position : int
        Positional index where spike begins.
    pre_spike_value : float
        Value immediately before the spike.
    threshold : float
        Maximum deviation from pre-spike value to consider as reverted.
    max_lookahead : int
        Maximum positions to search ahead.

    Returns
    -------
    int or None
        Index label of reversion point, or None if no reversion found.
    """
    if pd.isna(pre_spike_value) or pre_spike_value == 0:
        return None

    search_end = min(spike_start_position + max_lookahead, len(series))
    future_values = series.iloc[spike_start_position:search_end]

    deviation = (future_values - pre_spike_value).abs() / abs(pre_spike_value)
    reverted = deviation <= threshold

    if reverted.any():
        return reverted.idxmax()

    return None


def mark_spike_region(
    outlier_mask: pd.Series,
    spike_start_label: int,
    reversion_label: int,
) -> pd.Series:
    """
    Mark all points in spike region as outliers.

    Parameters
    ----------
    outlier_mask : pd.Series
        Boolean mask to update (modified in place).
    spike_start_label : int
        Index label where spike begins.
    reversion_label : int
        Index label where spike ends (exclusive).

    Returns
    -------
    pd.Series
        Updated outlier mask with spike region marked True.
    """
    start_pos = outlier_mask.index.get_loc(spike_start_label)
    end_pos = outlier_mask.index.get_loc(reversion_label)
    outlier_mask.iloc[start_pos:end_pos] = True
    return outlier_mask


def detect_reverting_spikes(
    series: pd.Series,
    threshold: float,
    max_lookahead: int = MAX_LOOKAHEAD_DAYS,
) -> pd.Series:
    """
    Detect spike regions that revert to pre-spike levels.

    A spike is confirmed when both conditions are met:
        1. Value changes by more than `threshold` from previous period
        2. Within `max_lookahead` periods, value returns to within
           `threshold` of the pre-spike level

    Parameters
    ----------
    series : pd.Series
        Time series of values.
    threshold : float
        Change threshold as decimal (e.g., 0.50 for 50%).
    max_lookahead : int
        Maximum periods to search for reversion.

    Returns
    -------
    pd.Series
        Boolean mask where True indicates an outlier position.
    """
    outlier_mask = pd.Series(False, index=series.index)
    pct_change = calculate_pct_change(series)
    spike_starts = find_spike_start_indices(pct_change, threshold)

    processed_until = -1

    for spike_label in spike_starts:
        spike_pos = series.index.get_loc(spike_label)

        if spike_pos <= processed_until:
            continue

        pre_spike_value = series.iloc[spike_pos - 1] if spike_pos > 0 else np.nan

        reversion_label = find_reversion_index(
            series=series,
            spike_start_position=spike_pos,
            pre_spike_value=pre_spike_value,
            threshold=threshold,
            max_lookahead=max_lookahead,
        )

        if reversion_label is not None:
            outlier_mask = mark_spike_region(outlier_mask, spike_label, reversion_label)
            processed_until = series.index.get_loc(reversion_label)

    return outlier_mask


# =============================================================================
# Rolling Median Outlier Detection
# =============================================================================

def detect_rolling_median_outliers(series: pd.Series) -> pd.Series:
    """
    Detect outliers by comparison to centered rolling median.

    This method catches isolated spikes that don't fit the reversion
    pattern used in other detection passes.

    Parameters
    ----------
    series : pd.Series
        Time series of values.

    Returns
    -------
    pd.Series
        Boolean mask where True indicates an outlier position.
    """
    rolling_median = series.rolling(
        window=ROLLING_WINDOW_DAYS,
        center=True,
        min_periods=5,
    ).median()

    deviation = (series - rolling_median).abs() / rolling_median.abs()
    outlier_mask = deviation > ROLLING_DEVIATION_THRESHOLD

    return outlier_mask.fillna(False)


# =============================================================================
# Correction Application
# =============================================================================

def apply_forward_fill_correction(
    series: pd.Series,
    outlier_mask: pd.Series,
) -> pd.Series:
    """
    Replace outlier values with forward-filled valid values.

    Parameters
    ----------
    series : pd.Series
        Original time series.
    outlier_mask : pd.Series
        Boolean mask indicating outlier positions.

    Returns
    -------
    pd.Series
        Corrected series with outliers replaced by last valid value.
    """
    corrected = series.copy()
    corrected[outlier_mask] = np.nan
    return corrected.ffill()


# =============================================================================
# Main Correction Pipeline
# =============================================================================

def correct_shares_outstanding_outliers(series: pd.Series) -> pd.Series:
    """
    Apply multi-pass outlier correction pipeline.

    Each pass operates on the corrected output of the previous pass,
    allowing progressive cleanup from extreme to subtle anomalies.

    Parameters
    ----------
    series : pd.Series
        Raw shares outstanding values.

    Returns
    -------
    pd.Series
        Outlier-corrected values.
    """
    corrected = series.copy()

    # Pass 1: Extreme spikes (>100%)
    outliers = detect_reverting_spikes(corrected, threshold=1.00)
    corrected = apply_forward_fill_correction(corrected, outliers)

    # Pass 2: Medium spikes (>50%)
    outliers = detect_reverting_spikes(corrected, threshold=0.50)
    corrected = apply_forward_fill_correction(corrected, outliers)

    # Pass 3: Smaller spikes (>35%)
    outliers = detect_reverting_spikes(corrected, threshold=0.35)
    corrected = apply_forward_fill_correction(corrected, outliers)

    # Pass 4: Rolling median outliers
    outliers = detect_rolling_median_outliers(corrected)
    corrected = apply_forward_fill_correction(corrected, outliers)

    return corrected


# =============================================================================
# Output Column Functions
# =============================================================================

def c_weighted_average_basic_shares_outstanding_outlier_adjusted(
    fis_weighted_average_basic_shares_outstanding,
):
    """
    Outlier-adjusted weighted average basic shares outstanding.

    Parameters
    ----------
    fis_weighted_average_basic_shares_outstanding : DataColumn
        Raw basic shares outstanding from data provider.

    Returns
    -------
    DataColumn
        Corrected basic shares outstanding with transient spikes removed.

    See Also
    --------
    c_weighted_average_diluted_shares_outstanding_outlier_adjusted
    """
    series = fis_weighted_average_basic_shares_outstanding.to_pandas()
    corrected = correct_shares_outstanding_outliers(series)
    return DataColumn.load(corrected)


def c_weighted_average_diluted_shares_outstanding_outlier_adjusted(
    fis_weighted_average_diluted_shares_outstanding,
):
    """
    Outlier-adjusted weighted average diluted shares outstanding.

    Parameters
    ----------
    fis_weighted_average_diluted_shares_outstanding : DataColumn
        Raw diluted shares outstanding from data provider.

    Returns
    -------
    DataColumn
        Corrected diluted shares outstanding with transient spikes removed.

    See Also
    --------
    c_weighted_average_basic_shares_outstanding_outlier_adjusted
    """
    series = fis_weighted_average_diluted_shares_outstanding.to_pandas()
    corrected = correct_shares_outstanding_outliers(series)
    return DataColumn.load(corrected)