"""
Data transformation and preprocessing utilities.

This module provides helper functions for converting between data formats,
particularly for time series data and trajectory representations.
"""

import pandas as pd


def build_timeseries(data, cols):
    """
    Convert numpy array to pandas DataFrame with datetime index.

    Utility function to create time-indexed DataFrames from MRCLAM data arrays.
    Converts Unix timestamps to pandas datetime objects for time series analysis.

    Parameters
    ----------
    data : ndarray
        Input data array where first column contains Unix timestamps.
    cols : list of str
        Column names for the DataFrame. First column should be 'stamp'.

    Returns
    -------
    pandas.DataFrame
        Time-indexed DataFrame with datetime index and remaining columns.

    Examples
    --------
    Basic usage with trajectory data:

    >>> import numpy as np
    >>> from musi_labs.utils.data_utils import build_timeseries
    >>>
    >>> # Create sample trajectory data [timestamp, x, y, theta]
    >>> data = np.array([
    ...     [1234567890.0, 0.0, 0.0, 0.0],
    ...     [1234567891.0, 0.1, 0.0, 0.1],
    ...     [1234567892.0, 0.2, 0.1, 0.2]
    ... ])
    >>> df = build_timeseries(data, cols=['stamp', 'x', 'y', 'theta'])
    >>> print(df.head())
                                x    y  theta
    stamp
    2009-02-13 23:31:30.0     0.0  0.0    0.0
    2009-02-13 23:31:31.0     0.1  0.0    0.1
    2009-02-13 23:31:32.0     0.2  0.1    0.2

    With measurement data:

    >>> measurement_data = np.array([
    ...     [1234567890.0, 6, 2.5, 0.3],  # [time, landmark_id, range, bearing]
    ...     [1234567891.0, 7, 3.2, 0.5]
    ... ])
    >>> df = build_timeseries(measurement_data, cols=['stamp', 'landmark_id', 'range', 'bearing'])

    Notes
    -----
    - Assumes timestamps are in Unix epoch format (seconds since 1970)
    - Sets timestamp column as DataFrame index for temporal operations
    - Enables pandas time series operations and plotting
    - Used by localization algorithms after calling build_dataframes()
    """
    timeseries = pd.DataFrame(data, columns=cols)
    timeseries["stamp"] = pd.to_datetime(timeseries["stamp"], unit="s")
    timeseries = timeseries.set_index("stamp")
    return timeseries
