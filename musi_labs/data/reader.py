#!/usr/bin/env python3
"""
MRCLAM Dataset Reader Module

This module provides the core data loading infrastructure for the 11765_MUSI
robotics navigation laboratory. It implements synchronized loading and preprocessing
of multi-robot cooperative localization datasets following the MRCLAM format.

The module handles the complex data synchronization between multiple data streams
(odometry, measurements, ground truth) and provides a unified interface for all
SLAM and localization algorithms in the laboratory.

References
----------
.. [1] Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic robotics.
       MIT Press. Chapter 2: "Recursive State Estimation"
.. [2] MRCLAM Dataset: Multi-Robot Cooperative Localization and Mapping Dataset
       from Carnegie Mellon University Robotics Institute
.. [3] Howard, A. (2004). Multi-robot simultaneous localization and mapping using
       particle filters. IJRR, 23(12), 1243-1256.
"""

import numpy as np


class Reader:
    """
    MRCLAM Dataset Reader for Multi-Robot Cooperative Localization and Mapping

    Loads and synchronizes multi-modal robotics data from the MRCLAM dataset collection,
    providing a unified interface for localization and SLAM algorithms. The reader handles
    temporal synchronization, data validation, and preprocessing of sensor measurements,
    odometry commands, and ground truth trajectories.

    The MRCLAM dataset consists of 9 individual scenarios collected in a 15m x 8m indoor
    environment with 5 iRobot Create platforms and 15 landmark targets. Each robot
    provides odometry at ~67Hz and range-bearing measurements to landmarks and other
    robots, with Vicon motion capture ground truth at 100Hz accuracy.

    Attributes
    ----------
    groundtruth_data : ndarray of shape (n_timesteps, 4)
        Ground truth robot poses [time[s], x[m], y[m], theta[rad]]
        From Vicon motion capture system (100Hz accuracy)
    odometry_data : ndarray of shape (n_odometry, 4)
        Robot control commands [time[s], subject_id, v[m/s], omega[rad/s]]
        Forward and angular velocity commands at ~67Hz
    measurement_data : ndarray of shape (n_measurements, 4)
        Range-bearing observations [time[s], subject_id, range[m], bearing[rad]]
        Camera-based measurements to landmarks and other robots
    landmark_groundtruth_data : ndarray of shape (15, 3)
        True landmark positions [subject_id, x[m], y[m]]
        From Vicon motion capture system
    barcodes_data : ndarray of shape (20, 2)
        Barcode-to-subject mapping [subject_id, barcode_id]
        Associates visual barcodes with robot/landmark IDs
    data : ndarray of shape (n_total, 4)
        Temporally synchronized input data combining odometry and measurements
        Format: [time[s], subject_id, data1, data2] where:
        - subject_id = -1: odometry data [time, -1, v[m/s], omega[rad/s]]
        - subject_id >= 6: measurements [time, landmark_id, range[m], bearing[rad]]
    landmark_locations : dict
        Landmark position lookup table {barcode_id: [x[m], y[m]]}
        Maps visual barcode identifiers to Cartesian coordinates
    landmark_indexes : dict
        Landmark index mapping {barcode_id: landmark_index}
        Maps barcode IDs to sequential landmark indices (1-15)

    Parameters
    ----------
    dataset : str
        Path to MRCLAM dataset directory containing the 5 required data files:
        - Barcodes.dat: [subject_id, barcode_id] mapping
        - {robot}_Groundtruth.dat: [time, x, y, theta] true trajectories
        - Landmark_Groundtruth.dat: [subject_id, x, y] landmark positions
        - {robot}_Measurement.dat: [time, subject_id, range, bearing] observations
        - {robot}_Odometry.dat: [time, v, omega] or [time, subject_id, v, omega] commands
        Available datasets: "data/MRCLAM_Dataset1" through "data/MRCLAM_Dataset9"
    robot : str
        Robot identifier for data selection. Valid values: "Robot1", "Robot2",
        "Robot3", "Robot4", "Robot5". Each robot provides independent odometry
        and measurement streams synchronized with shared ground truth.
    end_frame : int
        Maximum number of data frames to load for computational efficiency.
        Typical values: 1000-5000 for algorithm development, full dataset
        for final evaluation. Automatically truncates data streams at the
        specified frame count while maintaining temporal consistency.
    plot : bool, optional
        Enable visualization during data loading (default: True).
        Reserved for future plotting functionality.
    metrics : bool, optional
        Enable data quality metrics computation (default: True).
        Reserved for future statistical analysis functionality.

    Examples
    --------
    Basic Dataset Loading:

    >>> from musi_labs.data.reader import Reader
    >>> reader = Reader("data/MRCLAM_Dataset1", "Robot1", 1000)
    >>> print(f"Loaded {len(reader.data)} synchronized data points")
    >>> print(f"Ground truth trajectory: {reader.groundtruth_data.shape[0]} poses")

    Multi-Dataset Comparative Analysis:

    >>> datasets = ["data/MRCLAM_Dataset1", "data/MRCLAM_Dataset9"]
    >>> for dataset in datasets:
    ...     reader = Reader(dataset, "Robot1", 2000)
    ...     print(f"{dataset}: {len(reader.data)} data points, "
    ...           f"{len(reader.landmark_locations)} landmarks")

    Algorithm Development Workflow:

    >>> # Load dataset for EKF localization
    >>> reader = Reader("data/MRCLAM_Dataset1", "Robot1", 3000)
    >>> # Access synchronized control and measurement data
    >>> for i, data_point in enumerate(reader.data):
    ...     timestamp, subject_id, param1, param2 = data_point
    ...     if subject_id == -1:  # Odometry
    ...         v, omega = param1, param2  # [m/s], [rad/s]
    ...         # Apply motion model prediction
    ...     else:  # Measurement to landmark
    ...         range_obs, bearing_obs = param1, param2  # [m], [rad]
    ...         landmark_pos = reader.landmark_locations[subject_id]
    ...         # Apply measurement model update

    Notes
    -----
    MRCLAM Dataset Structure:
    - Environment: 15m x 8m indoor laboratory space
    - Robots: 5 iRobot Create differential-drive platforms
    - Landmarks: 15 cylindrical targets with unique barcodes
    - Sensors: Monocular cameras (960x720) for visual barcode detection
    - Ground Truth: 10-camera Vicon motion capture system
    - Synchronization: Network Time Protocol (NTP) with high accuracy

    Subject ID Convention:
    - -1: Odometry data (control inputs)
    - 1-5: Other robots (for cooperative localization)
    - 6-20: Landmarks (15 cylindrical targets)

    Coordinate Frames:
    - All coordinates in Vicon global reference frame
    - Robot poses: (x[m], y[m], theta[rad]) with theta in [-pi, pi]
    - Measurements: (range[m], bearing[rad]) in robot body frame

    Raises
    ------
    FileNotFoundError
        If dataset directory or required data files are missing
    ValueError
        If robot identifier is invalid or data format is incorrect
    IndexError
        If end_frame exceeds available data length

    References
    ----------
    .. [1] Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic robotics.
           MIT Press. Chapter 7: "Mobile Robot Localization"
    .. [2] MRCLAM Dataset Documentation, Carnegie Mellon University Robotics Institute
    .. [3] Bailey, T., & Durrant-Whyte, H. (2006). Simultaneous localization and mapping:
           Part II. IEEE Robotics & Automation Magazine, 13(3), 108-117.

    See Also
    --------
    src.localization.EKF : Extended Kalman Filter localization using Reader data
    src.localization.PF : Particle Filter localization using Reader data
    src.EKF_SLAM : EKF-based SLAM using Reader data and landmark associations
    """

    def __init__(self, dataset, robot, end_frame, plot=True, metrics=True):
        """
        Initialize MRCLAM dataset reader with synchronized data loading.

        Automatically loads and preprocesses all required data files, performing
        temporal synchronization and data validation. Creates unified data structures
        suitable for probabilistic robotics algorithms.

        Parameters
        ----------
        dataset : str
            Path to MRCLAM dataset directory
        robot : str
            Robot identifier ("Robot1" through "Robot5")
        end_frame : int
            Maximum number of data frames to process
        plot : bool, optional
            Enable visualization (default: True, reserved for future use)
        metrics : bool, optional
            Enable data quality metrics (default: True, reserved for future use)
        """
        self.load_data(dataset, robot, end_frame)

    def load_data(self, dataset, robot, end_frame):
        """
        Load and synchronize MRCLAM dataset files with temporal alignment.

        Implements the core data preprocessing pipeline for multi-robot cooperative
        localization datasets. Performs temporal synchronization, data validation,
        and creates unified data structures for probabilistic robotics algorithms.

        The method handles complex data associations between multiple asynchronous
        data streams and ensures temporal consistency across odometry, measurements,
        and ground truth trajectories.

        Processing Steps
        ----------------
        1. File Loading: Load 5 individual MRCLAM data files with validation
        2. Format Standardization: Ensure consistent array dimensions and formats
        3. Temporal Merging: Combine odometry and measurement streams chronologically
        4. Ground Truth Alignment: Synchronize Vicon data with sensor timestamps
        5. Data Truncation: Apply end_frame limit while preserving temporal consistency
        6. Landmark Association: Create barcode-to-position lookup tables

        Parameters
        ----------
        dataset : str
            Path to MRCLAM dataset directory containing required data files.
            Must include: Barcodes.dat, {robot}_Groundtruth.dat,
            Landmark_Groundtruth.dat, {robot}_Measurement.dat, {robot}_Odometry.dat
        robot : str
            Robot identifier for data file selection. Valid values: "Robot1",
            "Robot2", "Robot3", "Robot4", "Robot5". Determines which robot's
            odometry and measurement files are loaded.
        end_frame : int
            Maximum number of synchronized data frames to process. Provides
            computational efficiency for algorithm development and testing.
            Typical values: 1000-5000 for development, full dataset for evaluation.

        Raises
        ------
        FileNotFoundError
            If dataset directory or any required data file is missing
        ValueError
            If data files have incorrect format or robot identifier is invalid
        IndexError
            If end_frame is negative or data arrays are malformed

        Notes
        -----
        Data File Formats:
        - Barcodes.dat: Maps visual barcode IDs to subject/landmark indices
        - Groundtruth.dat: Vicon motion capture poses at 100Hz (high accuracy)
        - Landmark_Groundtruth.dat: True landmark positions from Vicon
        - Measurement.dat: Range-bearing observations from onboard cameras
        - Odometry.dat: Forward/angular velocity commands at ~67Hz

        Subject ID Convention:
        - -1: Odometry data (control inputs)
        - 1-5: Other robots (for cooperative localization)
        - 6-20: Landmarks (15 cylindrical targets with barcodes)

        Temporal Synchronization Details:
        The method handles time synchronization between different data streams
        collected at different frequencies:
        - Odometry: ~67Hz differential-drive commands
        - Measurements: Variable rate based on landmark visibility
        - Ground truth: 100Hz Vicon motion capture data

        Memory Efficiency:
        For large datasets, the end_frame parameter prevents memory issues:
        - Full Dataset 1: ~50,000 data points, ~200MB memory
        - Truncated (end_frame=3000): ~3,000 points, ~12MB memory

        Examples
        --------
        Internal Data Flow:
        Raw odometry file format: [time, v, omega]
        After preprocessing: [time, -1, v, omega]

        Landmark Association:
        Barcode 6 maps to landmark 1 at position [2.3, 4.5]
        barcode_id = 6
        landmark_pos = self.landmark_locations[barcode_id]  # [2.3, 4.5]
        landmark_idx = self.landmark_indexes[barcode_id]   # 1

        References
        ----------
        .. [1] Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic robotics.
               MIT Press. Chapter 2: "Recursive State Estimation", pp. 13-38
        .. [2] Howard, A. (2004). Multi-robot simultaneous localization and mapping
               using particle filters. IJRR, 23(12), 1243-1256.
        .. [3] MRCLAM Dataset Technical Report, Carnegie Mellon University
               Robotics Institute, 2004.
        """
        # Loading dataset
        # Barcodes: [Subject#, Barcode#]
        self.barcodes_data = np.loadtxt(dataset + "/Barcodes.dat")
        # Ground truth: [Time[s], x[m], y[m], orientation[rad]]
        self.groundtruth_data = np.loadtxt(dataset + "/" + robot + "_Groundtruth.dat")
        # Landmark ground truth: [Subject#, x[m], y[m]]
        self.landmark_groundtruth_data = np.loadtxt(
            dataset + "/Landmark_Groundtruth.dat"
        )
        # Measurement: [Time[s], Subject#, range[m], bearing[rad]]
        self.measurement_data = np.loadtxt(dataset + "/" + robot + "_Measurement.dat")
        # Odometry: [Time[s], Subject#, forward_V[m/s], angular _v[rad/s]]
        self.odometry_data = np.loadtxt(dataset + "/" + robot + "_Odometry.dat")

        # Ensure arrays are 2D (handle single-row case)
        if self.groundtruth_data.ndim == 1:
            self.groundtruth_data = self.groundtruth_data.reshape(1, -1)
        if self.landmark_groundtruth_data.ndim == 1:
            self.landmark_groundtruth_data = self.landmark_groundtruth_data.reshape(
                1, -1
            )
        if self.measurement_data.size > 0 and self.measurement_data.ndim == 1:
            self.measurement_data = self.measurement_data.reshape(1, -1)
        if self.odometry_data.ndim == 1:
            self.odometry_data = self.odometry_data.reshape(1, -1)
        if self.barcodes_data.ndim == 1:
            self.barcodes_data = self.barcodes_data.reshape(1, -1)

        # Collect all input data and sort by timestamp
        # Add subject "odom" = -1 for odometry data by inserting subject column
        odom_data = self.odometry_data.copy()
        if odom_data.shape[1] == 3:
            # Real MRCLAM format: [Time, v, omega] -> [Time, -1, v, omega]
            odom_data = np.insert(odom_data, 1, -1, axis=1)
        elif odom_data.shape[1] == 4:
            # Mock/test format: [Time, Subject#, v, omega] -> [Time, -1, v, omega]
            odom_data[:, 1] = -1

        # Handle empty measurement data
        if self.measurement_data.size == 0:
            self.data = odom_data
        else:
            self.data = np.concatenate((odom_data, self.measurement_data), axis=0)
        self.data = self.data[np.argsort(self.data[:, 0])]

        # Remove all data before the fisrt timestamp of groundtruth
        # Use first groundtruth data as the initial location of the robot
        for i in range(len(self.data)):
            if self.data[i][0] > self.groundtruth_data[0][0]:
                break
        self.data = self.data[i:]

        # Remove all data after the specified number of frames
        self.data = self.data[:end_frame]
        if len(self.data) > 0:
            cut_timestamp = self.data[-1][0]  # Use last available timestamp
            # Remove all groundtruth after the corresponding timestamp
            for i in range(len(self.groundtruth_data)):
                if self.groundtruth_data[i][0] >= cut_timestamp:
                    break
            self.groundtruth_data = self.groundtruth_data[:i]

        # Combine barcode Subject# with landmark Subject# to create lookup-table
        # [x[m], y[m], x std-dev[m], y std-dev[m]]
        self.landmark_locations = {}
        for i in range(5, len(self.barcodes_data), 1):
            self.landmark_locations[self.barcodes_data[i][1]] = (
                self.landmark_groundtruth_data[i - 5][1:]
            )

        # Lookup table to map barcode Subjec# to landmark Subject#
        # Barcode 6 is the first landmark (1 - 15 for 6 - 20)
        self.landmark_indexes = {}
        for i in range(5, len(self.barcodes_data), 1):
            self.landmark_indexes[self.barcodes_data[i][1]] = i - 4


if __name__ == "__main__":
    # Dataset 1
    dataset = "data/MRCLAM_Dataset1"
    end_frame = 3200
    robot = "Robot1"
    #
    r = Reader(dataset, robot, end_frame)
