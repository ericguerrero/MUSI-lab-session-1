#!/usr/bin/env python3
"""
Extended Kalman Filter (EKF) Localization Implementation

This module implements the Extended Kalman Filter for mobile robot localization
with known landmark correspondences, following the algorithm described in
"Probabilistic Robotics" by Thrun, Fox, and Burgard.

References
----------
.. [1] Thrun, S., Fox, D., & Burgard, W. (2005). Probabilistic Robotics.
       MIT Press. Chapter 7, Table 7.2, Page 204.

Notes
-----
The EKF localization algorithm represents beliefs bel(x_t) by their first and
second moments: the mean μ_t and the covariance Σ_t. This implementation is
designed for feature-based maps where the robot observes ranges and bearings
to nearby landmarks with known correspondences.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from musi_labs.utils.data_utils import build_timeseries


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for Mobile Robot Localization

    Implements the EKF localization algorithm with known correspondences
    for estimating robot pose in a feature-based environment.

    This class follows Table 7.2 from "Probabilistic Robotics" and uses
    the velocity motion model with linearization around the current pose
    estimate. The algorithm processes odometry and landmark measurements
    sequentially to maintain a Gaussian belief over the robot's pose.

    Mathematical Foundation
    ----------------------
    The EKF maintains a Gaussian belief bel(x_t) = N(μ_t, Σ_t) where:

    - State: x_t = [x, y, θ]^T (robot pose)
    - Motion model: x_t = g(u_t, x_{t-1}) + ε_t
    - Measurement model: z_t = h(x_t, m) + δ_t

    Motion Model (Velocity Model):
        x' = x - (v_t/ω_t) sin θ + (v_t/ω_t) sin(θ + ω_t Δt)
        y' = y + (v_t/ω_t) cos θ - (v_t/ω_t) cos(θ + ω_t Δt)
        θ' = θ + ω_t Δt

    Simplified Motion Model (for small Δt):
        x' = x + v_t cos(θ) Δt
        y' = y + v_t sin(θ) Δt
        θ' = θ + ω_t Δt

    Measurement Model (Range-Bearing):
        r_t = √((m_x - x)² + (m_y - y)²)
        φ_t = atan2(m_y - y, m_x - x) - θ

    Algorithm Steps
    --------------
    1. Motion Update (Prediction):
       - μ̄_t = g(u_t, μ_{t-1})
       - Σ̄_t = G_t Σ_{t-1} G_t^T + R_t

    2. Measurement Update (Correction):
       - K_t = Σ̄_t H_t^T (H_t Σ̄_t H_t^T + Q_t)^{-1}
       - μ_t = μ̄_t + K_t (z_t - ẑ_t)
       - Σ_t = (I - K_t H_t) Σ̄_t

    Parameters
    ----------
    dataset : str
        Path to MRCLAM dataset directory containing robot data files
    robot : str
        Robot identifier (e.g., 'Robot1', 'Robot2', etc.)
    end_frame : int
        Number of data frames to process from the dataset
    R : numpy.ndarray, shape (3, 3)
        Process noise covariance matrix (motion uncertainty)
        Diagonal elements: [σ_x², σ_y², σ_θ²]
    Q : numpy.ndarray, shape (2, 2) or (3, 3)
        Measurement noise covariance matrix
        For range-bearing: [[σ_r², 0], [0, σ_φ²]]
    plot : bool, optional
        Whether to generate trajectory plot (default: True)

    Attributes
    ----------
    states : numpy.ndarray, shape (n_frames, 4)
        Estimated robot trajectory [timestamp, x, y, θ]
    sigma : numpy.ndarray, shape (3, 3)
        Current pose covariance matrix
    groundtruth_data : numpy.ndarray
        True robot trajectory for comparison
    landmark_locations : dict
        Landmark coordinates {barcode_id: [x, y]}
    states_measurement : list
        Pose estimates after measurement updates

    Examples
    --------
    Basic usage with MRCLAM Dataset 1:

    >>> import numpy as np
    >>> from musi_labs.localization.EKF import ExtendedKalmanFilter
    >>>
    >>> # Define noise matrices
    >>> R = np.diagflat([1.0, 1.0, 10.0]) ** 2  # Process noise
    >>> Q = np.diagflat([30, 30]) ** 2           # Measurement noise
    >>>
    >>> # Run EKF localization
    >>> ekf = ExtendedKalmanFilter(
    ...     dataset="data/MRCLAM_Dataset1",
    ...     robot="Robot1",
    ...     end_frame=3200,
    ...     R=R,
    ...     Q=Q,
    ...     plot=True
    ... )
    >>>
    >>> # Access results
    >>> trajectory = ekf.states[:, 1:4]  # [x, y, θ]
    >>> final_uncertainty = ekf.sigma

    Parameter Tuning Guidelines:

    >>> # Conservative tuning (low process noise)
    >>> R_conservative = np.diagflat([0.1, 0.1, 1.0]) ** 2
    >>>
    >>> # Aggressive tuning (high process noise)
    >>> R_aggressive = np.diagflat([10.0, 10.0, 100.0]) ** 2
    >>>
    >>> # Measurement noise based on sensor characteristics
    >>> Q_laser = np.diagflat([5, 5]) ** 2      # High-quality laser
    >>> Q_camera = np.diagflat([50, 50]) ** 2   # Camera-based ranging

    Notes
    -----
    - The implementation assumes known data association between measurements
      and landmarks (correspondence problem is solved)
    - Uses simplified motion model for computational efficiency
    - Requires proper initialization with ground truth starting pose
    - Motion updates are skipped if time delta is too small (< 0.001s)
    - Measurement updates are skipped for unknown landmarks

    The algorithm performance depends heavily on:
    - Proper tuning of R and Q matrices
    - Quality of initial pose estimate
    - Validity of linearization assumptions
    - Accuracy of landmark positions

    See Also
    --------
    src.localization.PF : Particle Filter localization
    src.localization.dead_reckoning : Dead reckoning baseline
    """

    def __init__(self, dataset, robot, end_frame, R, Q, plot=True):
        """
        Initialize and run Extended Kalman Filter localization.

        Parameters
        ----------
        dataset : str
            Path to MRCLAM dataset directory
        robot : str
            Robot identifier ('Robot1', 'Robot2', etc.)
        end_frame : int
            Number of data frames to process
        R : numpy.ndarray, shape (3, 3)
            Process noise covariance matrix
        Q : numpy.ndarray, shape (2, 2) or (3, 3)
            Measurement noise covariance matrix
        plot : bool, optional
            Generate trajectory visualization (default: True)
        """
        self.load_data(dataset, robot, end_frame)
        self.initialization(R, Q)
        for data in self.data:
            if data[1] == -1:
                self.motion_update(data)
            else:
                self.measurement_update(data)
        if plot:
            self.plot_data()
        # if metrics: self.generate_metrics()

    def load_data(self, dataset, robot, end_frame):
        """
        Load and preprocess MRCLAM dataset files.

        Loads odometry, measurement, ground truth, and landmark data from
        the specified MRCLAM dataset. Synchronizes timestamps and creates
        lookup tables for efficient landmark access.

        Parameters
        ----------
        dataset : str
            Path to dataset directory containing .dat files
        robot : str
            Robot identifier for loading robot-specific files
        end_frame : int
            Maximum number of data frames to load

        Notes
        -----
        Creates the following data structures:
        - self.data: Combined odometry and measurement data, sorted by time
        - self.groundtruth_data: True robot poses [time, x, y, θ]
        - self.landmark_locations: {barcode_id: [x, y]} lookup table
        - self.landmark_indexes: {barcode_id: landmark_id} mapping

        Data synchronization ensures that:
        - All data starts from the first available ground truth timestamp
        - Odometry data is marked with subject ID = -1
        - Only static landmarks (IDs 6-20) are included
        """
        # Loading dataset
        # Barcodes: [Subject#, Barcode#]
        self.barcodes_data = np.loadtxt(dataset + "/Barcodes.dat")
        # Ground truth: [Time[s], x[m], y[m], orientation[rad]]
        self.groundtruth_data = np.loadtxt(dataset + "/" + robot + "_Groundtruth.dat")
        # self.groundtruth_data = self.groundtruth_data[2000:] # Remove initial readings
        # Landmark ground truth: [Subject#, x[m], y[m]]
        self.landmark_groundtruth_data = np.loadtxt(
            dataset + "/Landmark_Groundtruth.dat"
        )
        # Measurement: [Time[s], Subject#, range[m], bearing[rad]]
        self.measurement_data = np.loadtxt(dataset + "/" + robot + "_Measurement.dat")
        # Odometry: [Time[s], Subject#, forward_V[m/s], angular _v[rad/s]]
        self.odometry_data = np.loadtxt(dataset + "/" + robot + "_Odometry.dat")

        # Collect all input data and sort by timestamp
        # Add subject "odom" = -1 for odometry data
        odom_data = np.insert(self.odometry_data, 1, -1, axis=1)
        self.data = np.concatenate((odom_data, self.measurement_data), axis=0)
        self.data = self.data[np.argsort(self.data[:, 0])]

        # Remove all data before the fisrt timestamp of groundtruth
        # Use first groundtruth data as the initial location of the robot
        for i in range(len(self.groundtruth_data)):
            if self.groundtruth_data[i][0] > self.data[0][0]:
                break
        self.groundtruth_data = self.groundtruth_data[i:]
        for i in range(len(self.data)):
            if self.data[i][0] > self.groundtruth_data[0][0]:
                break
        self.data = self.data[i:]

        # Remove all data after the specified number of frames
        self.data = self.data[:end_frame]
        cut_timestamp = self.data[end_frame - 1][0]
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

    def initialization(self, R, Q):
        """
        Initialize EKF state and covariance matrices.

        Sets up the initial robot pose estimate using ground truth data
        and initializes the covariance matrices for the filter.

        Parameters
        ----------
        R : numpy.ndarray, shape (3, 3)
            Process noise covariance matrix
            Represents uncertainty in motion model:
            R = diag([σ_x², σ_y², σ_θ²])
        Q : numpy.ndarray, shape (2, 2) or (3, 3)
            Measurement noise covariance matrix
            For range-bearing measurements:
            Q = diag([σ_r², σ_φ²])

        Notes
        -----
        - Initial pose is set from ground truth for accurate starting point
        - Initial covariance is set very small (1e-10) since starting pose is known
        - Process noise R should increase with velocity and time step
        - Measurement noise Q depends on sensor characteristics
        """
        # Initial state
        self.states = np.array([self.groundtruth_data[0]])
        self.last_timestamp = self.states[-1][0]
        # Choose very small process covariance because we are using the ground truth data for initial location
        self.sigma = np.diagflat([1e-10, 1e-10, 1e-10])
        # States with measurement update
        self.states_measurement = []
        # State covariance matrix
        self.R = R
        # Measurement covariance matrix
        self.Q = Q

    def motion_update(self, control):
        """
        Perform EKF motion update (prediction step).

        Implements the prediction phase of the Extended Kalman Filter using
        the velocity motion model. Updates both the state estimate and
        covariance matrix based on odometry measurements.

        Mathematical Foundation
        ----------------------
        Motion Model (simplified for small Δt):
            x_t = x_{t-1} + v_t * cos(θ_{t-1}) * Δt
            y_t = y_{t-1} + v_t * sin(θ_{t-1}) * Δt
            θ_t = θ_{t-1} + ω_t * Δt

        Jacobian Matrix G_t:
            G_t = ∂g(u_t, x_{t-1}) / ∂x_{t-1}
            G_t = [[1, 0, -v_t * Δt * sin(θ_{t-1})],
                   [0, 1,  v_t * Δt * cos(θ_{t-1})],
                   [0, 0,  1]]

        Covariance Update:
            Σ̄_t = G_t * Σ_{t-1} * G_t^T + R_t

        Parameters
        ----------
        control : numpy.ndarray, shape (4,)
            Control input [timestamp, subject_id, v_t, ω_t]
            Where:
            - v_t: Forward velocity [m/s]
            - ω_t: Angular velocity [rad/s]
            - subject_id = -1 for odometry data

        Notes
        -----
        - Skips update if time delta < 0.001s to avoid numerical issues
        - Normalizes orientation to [-π, π] range
        - Uses simplified motion model assuming small time steps
        - Updates self.states and self.sigma

        References
        ----------
        Table 7.2, Lines 2-4 in "Probabilistic Robotics"
        """
        # ------------------ Step 1: Mean update ---------------------#
        # State: [x, y, θ]
        # Control: [v, w]
        # State-transition function (simplified):
        # [x_t, y_t, θ_t] = g(u_t, x_t-1)
        #   x_t  =  x_t-1 + v * cosθ_t-1 * delta_t
        #   y_t  =  y_t-1 + v * sinθ_t-1 * delta_t
        #   θ_t  =  θ_t-1 + w * delta_t
        # Skip motion update if two odometry data are too close
        delta_t = control[0] - self.last_timestamp
        if delta_t < 0.001:
            return
        # Compute updated [x, y, theta]
        x_t = self.states[-1][1] + control[2] * np.cos(self.states[-1][3]) * delta_t
        y_t = self.states[-1][2] + control[2] * np.sin(self.states[-1][3]) * delta_t
        theta_t = self.states[-1][3] + control[3] * delta_t
        # Limit θ within [-pi, pi]
        if theta_t > np.pi:
            theta_t -= 2 * np.pi
        elif theta_t < -np.pi:
            theta_t += 2 * np.pi
        self.last_timestamp = control[0]
        self.states = np.append(
            self.states, np.array([[control[0], x_t, y_t, theta_t]]), axis=0
        )

        # ------ Step 2: Linearize state-transition by Jacobian ------#
        # Jacobian: G = d g(u_t, x_t-1) / d x_t-1
        #         1  0  -v * delta_t * sinθ_t-1
        #   G  =  0  1   v * delta_t * cosθ_t-1
        #         0  0             1
        G_1 = np.array([1, 0, -control[2] * delta_t * np.sin(self.states[-1][3])])
        G_2 = np.array([0, 1, control[2] * delta_t * np.cos(self.states[-1][3])])
        G_3 = np.array([0, 0, 1])
        self.G = np.array([G_1, G_2, G_3])

        # ---------------- Step 3: Covariance update ------------------#
        self.sigma = self.G.dot(self.sigma).dot(self.G.T) + self.R

    def measurement_update(self, measurement):
        """
        Perform EKF measurement update (correction step).

        Implements the correction phase of the Extended Kalman Filter using
        range-bearing measurements to known landmarks. Updates both state
        estimate and covariance based on sensor observations.

        Mathematical Foundation
        ----------------------
        Measurement Model:
            r_t = √((m_x - x_t)² + (m_y - y_t)²)     [range]
            φ_t = atan2(m_y - y_t, m_x - x_t) - θ_t    [bearing]

        Expected Measurement:
            ẑ_t = h(x̄_t, m_j) = [r̄_t, φ̄_t]^T

        Measurement Jacobian H_t:
            H_t = ∂h(x_t, m_j) / ∂x_t
            H_t = [[∂r/∂x,  ∂r/∂y,  ∂r/∂θ ],
                   [∂φ/∂x,  ∂φ/∂y,  ∂φ/∂θ ],
                   [0,      0,      0    ]]

        Where q = (m_x - x_t)² + (m_y - y_t)²:
            ∂r/∂x = -(m_x - x_t) / √q
            ∂r/∂y = -(m_y - y_t) / √q
            ∂r/∂θ = 0
            ∂φ/∂x = (m_y - y_t) / q
            ∂φ/∂y = -(m_x - x_t) / q
            ∂φ/∂θ = -1

        Kalman Gain:
            K_t = Σ̄_t * H_t^T * (H_t * Σ̄_t * H_t^T + Q_t)^{-1}

        State Update:
            x_t = x̄_t + K_t * (z_t - ẑ_t)

        Covariance Update:
            Σ_t = (I - K_t * H_t) * Σ̄_t

        Parameters
        ----------
        measurement : numpy.ndarray, shape (4,)
            Sensor measurement [timestamp, barcode_id, range, bearing]
            Where:
            - barcode_id: Observed landmark identifier
            - range: Distance to landmark [m]
            - bearing: Angle to landmark [rad]

        Notes
        -----
        - Skips update if landmark is not in known landmark database
        - Uses known correspondence (barcode_id maps to landmark)
        - Innovation = measurement - expected_measurement
        - Updates self.states and self.sigma
        - Stores measurement update locations in self.states_measurement

        References
        ----------
        Table 7.2, Lines 5-15 in "Probabilistic Robotics"
        """
        # Continue if landmark is not found in self.landmark_locations
        if measurement[1] not in self.landmark_locations:
            return

        # ---------------- Step 1: Measurement update -----------------#
        #   range   =  sqrt((x_l - x_t)^2 + (y_l - y_t)^2)
        #  bearing  =  atan2((y_l - y_t) / (x_l - x_t)) - θ_t
        x_l = self.landmark_locations[measurement[1]][0]
        y_l = self.landmark_locations[measurement[1]][1]
        x_t = self.states[-1][1]
        y_t = self.states[-1][2]
        theta_t = self.states[-1][3]
        q = (x_l - x_t) * (x_l - x_t) + (y_l - y_t) * (y_l - y_t)
        range_expected = np.sqrt(q)
        bearing_expected = np.arctan2(y_l - y_t, x_l - x_t) - theta_t

        # -------- Step 2: Linearize Measurement by Jacobian ----------#
        # Jacobian: H = d h(x_t) / d x_t
        #        -(x_l - x_t) / sqrt(q)   -(y_l - y_t) / sqrt(q)   0
        #  H  =      (y_l - y_t) / q         -(x_l - x_t) / q     -1
        #                  0                         0             0
        #  q = (x_l - x_t)^2 + (y_l - y_t)^2
        H_1 = np.array([-(x_l - x_t) / np.sqrt(q), -(y_l - y_t) / np.sqrt(q), 0])
        H_2 = np.array([(y_l - y_t) / q, -(x_l - x_t) / q, -1])
        H_3 = np.array([0, 0, 0])
        self.H = np.array([H_1, H_2, H_3])

        # ---------------- Step 3: Kalman gain update -----------------#
        S_t = self.H.dot(self.sigma).dot(self.H.T) + self.Q
        self.K = self.sigma.dot(self.H.T).dot(np.linalg.inv(S_t))

        # ------------------- Step 4: mean update ---------------------#
        difference = np.array(
            [measurement[2] - range_expected, measurement[3] - bearing_expected, 0]
        )
        innovation = self.K.dot(difference)
        self.last_timestamp = measurement[0]
        self.states = np.append(
            self.states,
            np.array(
                [
                    [
                        self.last_timestamp,
                        x_t + innovation[0],
                        y_t + innovation[1],
                        theta_t + innovation[2],
                    ]
                ]
            ),
            axis=0,
        )
        self.states_measurement.append([x_t + innovation[0], y_t + innovation[1]])

        # ---------------- Step 5: covariance update ------------------#
        self.sigma = (np.identity(3) - self.K.dot(self.H)).dot(self.sigma)

    def plot_data(self):
        """
        Visualize EKF localization results.

        Creates a comprehensive plot showing:
        - Ground truth robot trajectory (blue line)
        - EKF estimated trajectory (red line)
        - Start and end points (green/yellow markers)
        - Measurement update locations (black dots)
        - Landmark positions with ID labels (star markers)

        The visualization helps assess:
        - Algorithm accuracy vs. ground truth
        - Consistency of estimates
        - Impact of measurement updates
        - Landmark distribution in environment
        """
        # Ground truth data
        plt.plot(
            self.groundtruth_data[:, 1],
            self.groundtruth_data[:, 2],
            "b",
            label="Robot State Ground truth",
        )

        # States
        plt.plot(
            self.states[:, 1], self.states[:, 2], "r", label="Robot State Estimate"
        )

        # Start and end points
        plt.plot(
            self.groundtruth_data[0, 1],
            self.groundtruth_data[0, 2],
            "go",
            label="Start point",
        )
        plt.plot(
            self.groundtruth_data[-1, 1],
            self.groundtruth_data[-1, 2],
            "yo",
            label="End point",
        )

        # Measurement update locations
        if len(self.states_measurement) > 0:
            self.states_measurement = np.array(self.states_measurement)
            plt.scatter(
                self.states_measurement[:, 0],
                self.states_measurement[:, 1],
                s=10,
                c="k",
                alpha=0.5,
                label="Measurement updates",
            )

        # Landmark ground truth locations and indexes
        landmark_xs = []
        landmark_ys = []
        for location in self.landmark_locations:
            landmark_xs.append(self.landmark_locations[location][0])
            landmark_ys.append(self.landmark_locations[location][1])
            index = self.landmark_indexes[location] + 5
            plt.text(
                landmark_xs[-1], landmark_ys[-1], str(index), alpha=0.5, fontsize=10
            )
        plt.scatter(
            landmark_xs,
            landmark_ys,
            s=200,
            c="k",
            alpha=0.2,
            marker="*",
            label="Landmark Locations",
        )

        # plt.title("Localization with only odometry data")
        plt.title("EKF Localization with Known Correspondences")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()

    def build_dataframes(self):
        """
        Convert data arrays to pandas DataFrames for analysis.

        Creates structured DataFrames with proper timestamps and column names
        for easier data manipulation and analysis.

        Returns
        -------
        Updates class attributes:
        - self.gt: Ground truth trajectory DataFrame
        - self.states_df: Estimated trajectory DataFrame
        - self.measurements: All measurement data DataFrame
        - self.motion: Odometry data DataFrame
        - self.sensor: Landmark measurements DataFrame

        Notes
        -----
        - self.states remains as numpy array for consistent interface
        """
        self.gt = build_timeseries(
            self.groundtruth_data, cols=["stamp", "x", "y", "theta"]
        )
        self.states_df = build_timeseries(self.states, cols=["stamp", "x", "y", "theta"])
        self.measurements = build_timeseries(
            self.data, cols=["stamp", "type", "range_l", "bearing_l"]
        )
        self.motion = self.measurements[self.measurements.type == -1].rename(
            columns={"range_l": "v", "bearing_l": "omega"}
        )
        landmarks = self.measurements[self.measurements.type != -1]
        self.sensor = filter_static_landmarks(landmarks, self.barcodes_data)


def filter_static_landmarks(lm, barcodes):
    """
    Filter measurement data to include only static landmarks.

    Converts barcode IDs to landmark IDs and filters out dynamic
    objects (other robots) keeping only static landmarks (IDs 6-20).

    Parameters
    ----------
    lm : pandas.DataFrame
        Landmark measurement data
    barcodes : numpy.ndarray
        Barcode to landmark ID mapping

    Returns
    -------
    pandas.DataFrame
        Filtered measurements containing only static landmarks
    """
    for L, l in dict(barcodes).items():  # Translate barcode num to landmark num
        lm[lm == l] = L
    lm = lm[lm.type > 5]  # Keep only static landmarks
    return lm


if __name__ == "__main__":
    """
    Example usage of ExtendedKalmanFilter with MRCLAM Dataset 1.

    This example demonstrates typical parameter values and usage patterns
    for EKF localization with the MRCLAM dataset.
    """
    # MRCLAM Dataset 1 parameters
    dataset = "data/MRCLAM_Dataset1"
    end_frame = 3200
    robot = "Robot1"

    # Process noise covariance matrix R
    # Higher values = more uncertainty in motion model
    # [x_uncertainty², y_uncertainty², θ_uncertainty²]
    R = np.diagflat(np.array([1.0, 1.0, 10.0])) ** 2

    # Measurement noise covariance matrix Q
    # Higher values = less trust in sensor measurements
    # [range_uncertainty², bearing_uncertainty², signature_uncertainty²]
    Q = np.diagflat(np.array([30, 30, 1e16])) ** 2

    # Run EKF localization
    ekf = ExtendedKalmanFilter(dataset, robot, end_frame, R, Q)

    # Example parameter tuning guidelines:
    # - Increase R for more aggressive tracking of odometry
    # - Decrease R for smoother, more conservative estimates
    # - Increase Q if sensors are noisy or unreliable
    # - Decrease Q for high-quality sensors
    # - Signature uncertainty (Q[2,2]) set very high to ignore landmark IDs
