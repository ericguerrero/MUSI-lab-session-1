"""
Trajectory evaluation metrics for robot localization and SLAM.

This module provides standardized metrics for evaluating localization algorithm
performance, including Absolute Trajectory Error (ATE) and related statistics.
All metrics operate on pandas DataFrames with timestamp indices for robust
temporal alignment.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)


def compute_ate(
    estimated_states: pd.DataFrame,
    groundtruth_data: pd.DataFrame,
    verbose: bool = True
) -> float:
    """
    Compute Absolute Trajectory Error (ATE) using RMSE with timestamp matching.

    ATE measures the root mean squared error between estimated and ground truth
    positions after temporal alignment. This metric is robust to different sampling
    rates and missing data through timestamp-based join operations.

    Parameters
    ----------
    estimated_states : pd.DataFrame
        Estimated trajectory with datetime index (from build_dataframes()).
        Must contain columns: ['x', 'y'] at minimum.
        Typically created by: algorithm.build_dataframes() -> algorithm.states_df
    groundtruth_data : pd.DataFrame
        Ground truth trajectory with datetime index (from build_dataframes()).
        Must contain columns: ['x', 'y'] at minimum.
        Typically created by: algorithm.build_dataframes() -> algorithm.gt
    verbose : bool, optional
        If True, print alignment statistics and diagnostic information.
        Default: True.

    Returns
    -------
    float
        Root Mean Squared Error (RMSE) of position errors in meters.

    Raises
    ------
    ValueError
        If inputs are not DataFrames or missing required columns.
    RuntimeError
        If timestamp alignment produces no matching frames.

    Notes
    -----
    Algorithm:
    1. Validate inputs (type, columns, index)
    2. Perform inner join on timestamp index
    3. Compute per-frame Euclidean errors
    4. Return RMSE across all aligned frames

    The inner join ensures only temporally aligned frames are compared,
    avoiding biased error estimates from misaligned data.

    Examples
    --------
    Basic usage with Dead Reckoning:

    >>> from musi_labs.localization.dead_reckoning import DeadReckoning
    >>> from musi_labs.utils.metrics import compute_ate
    >>>
    >>> dr = DeadReckoning("data/MRCLAM_Dataset1", "Robot1", 3200, plot=False)
    >>> dr.run()
    >>> dr.build_dataframes()  # Creates dr.states_df and dr.gt
    >>>
    >>> ate = compute_ate(dr.states_df, dr.gt)
    >>> print(f"Dead Reckoning ATE: {ate:.3f} m")

    With EKF localization:

    >>> from musi_labs.localization.EKF import ExtendedKalmanFilter
    >>>
    >>> ekf = ExtendedKalmanFilter(dataset, robot, 3200, R, Q, plot=False)
    >>> ekf.build_dataframes()
    >>> ate = compute_ate(ekf.states_df, ekf.gt, verbose=True)
    ✓ ATE computation successful
    ✓ Estimated states: 1234 frames
    ✓ Ground truth: 2456 frames
    ✓ Aligned frames: 1234 (100.0% of estimates)
    ✓ Mean error: 0.123 m
    ✓ ATE (RMSE): 0.156 m

    Comparing algorithms:

    >>> dr.build_dataframes()
    >>> ekf.build_dataframes()
    >>> ate_dr = compute_ate(dr.states_df, dr.gt, verbose=False)
    >>> ate_ekf = compute_ate(ekf.states_df, ekf.gt, verbose=False)
    >>> improvement = (ate_dr - ate_ekf) / ate_dr * 100
    >>> print(f"EKF improvement over DR: {improvement:.1f}%")

    See Also
    --------
    compute_rpe : Relative Pose Error for local consistency
    compute_trajectory_stats : Detailed error statistics
    """
    # Validate input types
    if not isinstance(estimated_states, pd.DataFrame):
        raise ValueError(
            f"estimated_states must be a DataFrame, got {type(estimated_states).__name__}. "
            f"Did you call build_dataframes() and use states_df instead of states?"
        )

    if not isinstance(groundtruth_data, pd.DataFrame):
        raise ValueError(
            f"groundtruth_data must be a DataFrame, got {type(groundtruth_data).__name__}. "
            f"Did you call build_dataframes() and use gt instead of groundtruth_data?"
        )

    # Validate required columns
    required_cols = ['x', 'y']
    for col in required_cols:
        if col not in estimated_states.columns:
            raise ValueError(
                f"estimated_states missing required column '{col}'. "
                f"Available columns: {list(estimated_states.columns)}"
            )
        if col not in groundtruth_data.columns:
            raise ValueError(
                f"groundtruth_data missing required column '{col}'. "
                f"Available columns: {list(groundtruth_data.columns)}"
            )

    if verbose:
        logger.info("=" * 60)
        logger.info("ATE Computation: Input Validation")
        logger.info(f"✓ Estimated states: {len(estimated_states)} frames")
        logger.info(f"✓ Ground truth: {len(groundtruth_data)} frames")
        logger.info(f"✓ Estimated columns: {list(estimated_states.columns)}")
        logger.info(f"✓ Ground truth columns: {list(groundtruth_data.columns)}")
        logger.info(f"✓ Estimated time range: {estimated_states.index.min()} to {estimated_states.index.max()}")
        logger.info(f"✓ Ground truth time range: {groundtruth_data.index.min()} to {groundtruth_data.index.max()}")

    # Join on timestamp index (inner join to get only matching timestamps)
    aligned = estimated_states[['x', 'y']].join(
        groundtruth_data[['x', 'y']],
        how='inner',
        rsuffix='_gt'
    )

    # Check if alignment was successful
    if len(aligned) == 0:
        raise RuntimeError(
            "Timestamp alignment produced 0 matching frames! "
            f"Estimated time range: [{estimated_states.index.min()}, {estimated_states.index.max()}], "
            f"Ground truth time range: [{groundtruth_data.index.min()}, {groundtruth_data.index.max()}]"
        )

    if verbose:
        alignment_pct = len(aligned) / len(estimated_states) * 100
        logger.info("=" * 60)
        logger.info("ATE Computation: Timestamp Alignment")
        logger.info(f"✓ Aligned frames: {len(aligned)} ({alignment_pct:.1f}% of estimates)")

        if alignment_pct < 90:
            logger.warning(
                f"⚠ Only {alignment_pct:.1f}% of frames aligned! "
                "Check timestamp synchronization."
            )

    # Compute per-frame Euclidean errors
    errors = np.sqrt(
        (aligned['x'] - aligned['x_gt']) ** 2 +
        (aligned['y'] - aligned['y_gt']) ** 2
    )

    # Compute ATE (RMSE)
    ate = np.sqrt(np.mean(errors ** 2))

    if verbose:
        logger.info("=" * 60)
        logger.info("ATE Computation: Error Statistics")
        logger.info(f"✓ Mean error: {np.mean(errors):.4f} m")
        logger.info(f"✓ Std dev: {np.std(errors):.4f} m")
        logger.info(f"✓ Median error: {np.median(errors):.4f} m")
        logger.info(f"✓ Min error: {np.min(errors):.4f} m")
        logger.info(f"✓ Max error: {np.max(errors):.4f} m")
        logger.info(f"✓ ATE (RMSE): {ate:.4f} m")
        logger.info("=" * 60)

    return ate


def compute_trajectory_stats(
    estimated_states: pd.DataFrame,
    groundtruth_data: pd.DataFrame
) -> dict:
    """
    Compute detailed trajectory error statistics.

    Parameters
    ----------
    estimated_states : pd.DataFrame
        Estimated trajectory with datetime index and columns ['x', 'y', 'theta'].
    groundtruth_data : pd.DataFrame
        Ground truth trajectory with datetime index and columns ['x', 'y', 'theta'].

    Returns
    -------
    dict
        Dictionary containing:
        - 'ate': Absolute Trajectory Error (RMSE)
        - 'mean_error': Mean position error
        - 'std_error': Standard deviation of errors
        - 'median_error': Median position error
        - 'max_error': Maximum position error
        - 'min_error': Minimum position error
        - 'aligned_frames': Number of temporally aligned frames
        - 'alignment_ratio': Fraction of frames successfully aligned
    """
    # Align trajectories
    aligned = estimated_states[['x', 'y']].join(
        groundtruth_data[['x', 'y']],
        how='inner',
        rsuffix='_gt'
    )

    if len(aligned) == 0:
        raise RuntimeError("No matching timestamps between trajectories")

    # Compute errors
    errors = np.sqrt(
        (aligned['x'] - aligned['x_gt']) ** 2 +
        (aligned['y'] - aligned['y_gt']) ** 2
    )

    return {
        'ate': np.sqrt(np.mean(errors ** 2)),
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'median_error': np.median(errors),
        'max_error': np.max(errors),
        'min_error': np.min(errors),
        'aligned_frames': len(aligned),
        'alignment_ratio': len(aligned) / len(estimated_states)
    }


def compare_algorithms(
    algorithms: dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
) -> pd.DataFrame:
    """
    Compare multiple localization algorithms using ATE and other metrics.

    Parameters
    ----------
    algorithms : dict
        Dictionary mapping algorithm names to (states_df, gt_df) tuples.
        Example: {'DR': (dr.states_df, dr.gt), 'EKF': (ekf.states_df, ekf.gt)}

    Returns
    -------
    pd.DataFrame
        Comparison table with columns:
        ['Algorithm', 'ATE', 'Mean Error', 'Std Error', 'Max Error', 'Aligned Frames']
        Sorted by ATE (best first).

    Examples
    --------
    >>> dr.build_dataframes()
    >>> ekf.build_dataframes()
    >>> results = compare_algorithms({
    ...     'Dead Reckoning': (dr.states_df, dr.gt),
    ...     'EKF': (ekf.states_df, ekf.gt)
    ... })
    >>> print(results)
    """
    results = []

    for name, (states_df, gt_df) in algorithms.items():
        stats = compute_trajectory_stats(states_df, gt_df)
        results.append({
            'Algorithm': name,
            'ATE': stats['ate'],
            'Mean Error': stats['mean_error'],
            'Std Error': stats['std_error'],
            'Max Error': stats['max_error'],
            'Aligned Frames': stats['aligned_frames']
        })

    df = pd.DataFrame(results)
    return df.sort_values('ATE')


def compute_dataset_metrics(reader) -> dict:
    """
    Extract key metrics from a MRCLAM dataset.

    Analyzes the dataset trajectory and measurements to compute characteristics
    useful for dataset selection, experimental design, and result interpretation.

    Parameters
    ----------
    reader : Reader
        Reader object with loaded MRCLAM dataset. Must have:
        - groundtruth_data: numpy array [timestamp, x, y, theta]
        - landmark_locations: dict of landmark positions
        - measurement_data: list/array of measurements

    Returns
    -------
    dict
        Dictionary containing:
        - 'path_length': Total distance traveled (m)
        - 'duration': Total time of trajectory (s)
        - 'n_landmarks': Number of static landmarks
        - 'distance': Direct start-to-end distance (m)
        - 'm_density': Measurement density (observations per meter)

    Examples
    --------
    Basic usage:

    >>> from musi_labs.data.reader import Reader
    >>> from musi_labs.utils.metrics import compute_dataset_metrics
    >>>
    >>> reader = Reader("data/MRCLAM_Dataset1", "Robot1", 5000, plot=False)
    >>> metrics = compute_dataset_metrics(reader)
    >>> print(f"Path length: {metrics['path_length']:.2f} m")
    >>> print(f"Measurement density: {metrics['m_density']:.2f} obs/m")

    Comparing multiple datasets:

    >>> datasets = ["MRCLAM_Dataset1", "MRCLAM_Dataset2", "MRCLAM_Dataset3"]
    >>> for ds in datasets:
    ...     reader = Reader(f"data/{ds}", "Robot1", 5000, plot=False)
    ...     metrics = compute_dataset_metrics(reader)
    ...     print(f"{ds}: {metrics['path_length']:.1f}m, {metrics['m_density']:.1f} obs/m")

    Notes
    -----
    - Path length is computed as cumulative Euclidean distance
    - Measurement density excludes invalid measurements (landmark_id = -1)
    - Duration is in seconds from first to last ground truth timestamp
    - Distance is straight-line distance ignoring trajectory shape
    """
    gt = reader.groundtruth_data

    # Path length (cumulative distance traveled)
    dx = np.diff(gt[:, 1])
    dy = np.diff(gt[:, 2])
    path_length = np.sum(np.sqrt(dx**2 + dy**2))

    # Duration (total time)
    duration = gt[-1, 0] - gt[0, 0]

    # Number of landmarks
    n_landmarks = len(reader.landmark_locations)

    # Direct distance (start to end)
    distance = np.linalg.norm(gt[-1, 1:3] - gt[0, 1:3])

    # Measurement density (measurements per meter)
    n_measurements = len([m for m in reader.measurement_data if m[1] != -1])
    m_density = n_measurements / path_length if path_length > 0 else 0

    return {
        "path_length": path_length,
        "duration": duration,
        "n_landmarks": n_landmarks,
        "distance": distance,
        "m_density": m_density,
    }
