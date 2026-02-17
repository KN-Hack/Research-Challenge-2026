"""
Simple Moving Average Alpha Signal
==================================

Custom calculation functions for KaxaNuk Data Curator that generate
trading signals based on Simple Moving Average (SMA) crossovers.

Output Columns
--------------
c_50_sma
    50-day Simple Moving Average of the adjusted close price.

c_200_sma
    200-day Simple Moving Average of the adjusted close price.

c_sma_signal
    Binary signal: 1 when short-term trend is above long-term trend, 0 otherwise.

Signal Logic
------------
The SMA crossover is a classic trend-following indicator:

    - **Signal = 1** : SMA(50) > SMA(200) → Bullish trend (upward momentum)
    - **Signal = 0** : SMA(50) ≤ SMA(200) → Bearish trend (downward momentum)
    - **Signal = NaN**: Insufficient data to compute one or both SMAs

This approach avoids false signals during the warmup period by returning
NaN until both moving averages have enough data points.
"""

import numpy as np
import pandas as pd
from kaxanuk.data_curator.modules.data_column import DataColumn
from kaxanuk.data_curator.features import helpers


# =============================================================================
# Moving Average Calculations
# =============================================================================

def c_50_sma(m_close_dividend_and_split_adjusted):
    """
    50-day Simple Moving Average of adjusted close price.

    Parameters
    ----------
    m_close_dividend_and_split_adjusted : DataColumn
        Daily adjusted close prices (adjusted for dividends and splits).

    Returns
    -------
    DataColumn
        50-day SMA values. First 49 values will be NaN (warmup period).
    """
    return helpers.simple_moving_average(
        column=m_close_dividend_and_split_adjusted,
        days=50,
    )


def c_200_sma(m_close_dividend_and_split_adjusted):
    """
    200-day Simple Moving Average of adjusted close price.

    Parameters
    ----------
    m_close_dividend_and_split_adjusted : DataColumn
        Daily adjusted close prices (adjusted for dividends and splits).

    Returns
    -------
    DataColumn
        200-day SMA values. First 199 values will be NaN (warmup period).
    """
    return helpers.simple_moving_average(
        column=m_close_dividend_and_split_adjusted,
        days=200,
    )


# =============================================================================
# Signal Generation
# =============================================================================

def c_sma_signal(c_50_sma, c_200_sma):
    """
    Generate trend signal based on SMA crossover.

    Compares the 50-day and 200-day Simple Moving Averages to determine
    the prevailing trend direction.

    Parameters
    ----------
    c_50_sma : DataColumn
        50-day Simple Moving Average values.
    c_200_sma : DataColumn
        200-day Simple Moving Average values.

    Returns
    -------
    DataColumn
        Signal values:
            - 1.0 : Bullish (SMA 50 > SMA 200)
            - 0.0 : Bearish (SMA 50 ≤ SMA 200)
            - NaN : Insufficient data for either SMA

    Notes
    -----
    The signal is only generated when both SMAs have valid values.
    This ensures no false signals during the 200-day warmup period.
    """
    sma_50 = c_50_sma.to_pandas()
    sma_200 = c_200_sma.to_pandas()

    # Initialize signal as NaN (no signal during warmup)
    signal = pd.Series(np.nan, index=sma_50.index)

    # Only generate signal where both SMAs are valid
    valid_mask = sma_50.notna() & sma_200.notna()

    # Bullish when short-term > long-term
    signal[valid_mask] = (sma_50[valid_mask] > sma_200[valid_mask]).astype(float)

    return DataColumn.load(signal)