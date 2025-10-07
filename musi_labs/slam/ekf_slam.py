#!/usr/bin/env python3
"""
Extended Kalman Filter SLAM with Known Correspondences.

This module implements the EKF-SLAM algorithm for simultaneous localization and mapping
when landmark correspondences are known a priori. The algorithm maintains a joint
estimate of robot pose and landmark positions using an augmented state vector.

Mathematical Foundation
-----------------------
The EKF-SLAM posterior represents the joint distribution over robot pose x_t and map m:
    p(x_t, m | z_{1:t}, u_{1:t})

Augmented State Vector:
    y_t = [x_t, y_t, θ_t, m_{1,x}, m_{1,y}, ..., m_{N,x}, m_{N,y}]^T

Where:
    - (x_t, y_t, θ_t): Robot pose at time t
    - (m_{i,x}, m_{i,y}): Coordinates of landmark i
    - N: Number of landmarks
    - Dimension: 3 + 2N

Covariance Matrix Structure:
    Σ_t = [ Σ_rr  Σ_rm ]
          [ Σ_mr  Σ_mm ]

Where:
    - Σ_rr (3×3): Robot pose uncertainty
    - Σ_rm (3×2N): Robot-landmark correlations
    - Σ_mr (2N×3): Landmark-robot correlations
    - Σ_mm (2N×2N): Landmark-landmark correlations

Algorithm Steps
---------------
1. Motion Update (Prediction):
   - Update robot pose using motion model
   - Landmark positions remain unchanged
   - Propagate uncertainty through linearized motion model

2. State Augmentation:
   - Initialize new landmarks when first observed
   - Augment state vector and covariance matrix

3. Measurement Update (Correction):
   - For each measurement, compute expected observation
   - Calculate innovation and Kalman gain
   - Update entire state vector (robot + all landmarks)
   - Information propagates through correlations

Key Innovation
--------------
Observing any landmark improves estimates of:
    - Robot pose (through measurement model)
    - Previously observed landmarks (through correlations)
    - Future localization accuracy

This information propagation is the fundamental insight of EKF-SLAM.

References
----------
.. [1] Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic Robotics.
       Chapter 10: SLAM with Extended Kalman Filters.
.. [2] Smith, R., Self, M., & Cheeseman, P. (1990). Estimating uncertain
       spatial relationships in robotics. Autonomous Robot Vehicles.

See Also
--------
src.EKF_SLAM.EKF_SLAM_unknown_correspondences : EKF-SLAM with data association
src.localization.EKF : EKF localization with known map

Examples
--------
>>> from musi_labs.EKF_SLAM.EKF_SLAM_known_correspondences import ExtendedKalmanFilterSLAM
>>> import numpy as np
>>>
>>> # Process noise covariance (robot motion)
>>> R = np.diag([0.1, 0.1, 0.05])**2
>>> # Measurement noise covariance (range, bearing, signature)
>>> Q = np.diag([0.1, 0.05, 1e6])**2
>>>
>>> # Run EKF-SLAM on MRCLAM dataset
>>> slam = ExtendedKalmanFilterSLAM(
...     dataset="data/MRCLAM_Dataset1",
...     robot="Robot1",
...     start_frame=800,
...     end_frame=3200,
...     R=R,
...     Q=Q,
...     plot=True,
...     plot_inter=False
... )
>>>
>>> # Access final estimates
>>> final_robot_pose = slam.states[-1][:3]
>>> final_landmarks = slam.states[-1][3:].reshape(-1, 2)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ExtendedKalmanFilterSLAM:
    """
    Extended Kalman Filter for Simultaneous Localization and Mapping with Known Correspondences.

    This class implements the EKF-SLAM algorithm described in Table 10.1 of
    "Probabilistic Robotics" by Thrun, Burgard, and Fox. The algorithm jointly
    estimates robot pose and landmark positions using an augmented state vector.

    The key insight is that landmark observations improve not only the observed
    landmark's position estimate, but also the robot pose and all previously
    observed landmarks through the covariance correlations maintained by the EKF.

    Mathematical Model
    ------------------
    State Vector (dimension 3 + 2N):
        y_t = [x, y, θ, m₁ₓ, m₁ᵧ, m₂ₓ, m₂ᵧ, ..., mₙₓ, mₙᵧ]ᵀ

    Motion Model (bicycle model):
        x_{t+1} = x_t + v·cos(θ_t)·Δt
        y_{t+1} = y_t + v·sin(θ_t)·Δt
        θ_{t+1} = θ_t + ω·Δt

    Measurement Model (range-bearing):
        r_i = √[(mᵢₓ - x)² + (mᵢᵧ - y)²] + noise
        φ_i = atan2(mᵢᵧ - y, mᵢₓ - x) - θ + noise

    Computational Complexity
    ------------------------
    - Memory: O(N²) for covariance matrix
    - Time per update: O(N²) for covariance propagation
    - Practical limit: ~1000 landmarks

    Attributes
    ----------
    states : ndarray of shape (T, 3 + 2N)
        History of state estimates over time.
    sigma : ndarray of shape (3 + 2N, 3 + 2N)
        Current state covariance matrix.
    landmark_observed : ndarray of shape (N+1,)
        Boolean array tracking which landmarks have been observed.
    groundtruth_data : ndarray
        Ground truth robot trajectory for comparison.
    R : ndarray of shape (3, 3)
        Process noise covariance matrix.
    Q : ndarray of shape (3, 3)
        Measurement noise covariance matrix.

    Notes
    -----
    This implementation assumes:
    - Known data association (correspondences provided)
    - Gaussian motion and measurement noise
    - Static landmark positions
    - Range-bearing-signature measurements

    The algorithm fails gracefully with unknown correspondences but does not
    perform data association. For unknown correspondences, use
    EKF_SLAM_unknown_correspondences instead.

    See Also
    --------
    EKF_SLAM_unknown_correspondences : EKF-SLAM with maximum likelihood data association
    """

    def __init__(self, dataset, robot, start_frame, end_frame, R, Q, plot, plot_inter):
        """
        Initialize and run EKF-SLAM algorithm with known correspondences.

        This constructor loads the dataset, initializes the state vector and
        covariance matrix, then processes all odometry and measurement data
        sequentially to build the map and track the robot.

        Parameters
        ----------
        dataset : str
            Path to MRCLAM dataset directory containing measurement files.
        robot : str
            Robot name (e.g., "Robot1", "Robot2", etc.).
        start_frame : int
            Starting frame index for data processing.
        end_frame : int
            Ending frame index for data processing.
        R : ndarray of shape (3, 3)
            Process noise covariance matrix for robot motion:
            R = diag([σ_x², σ_y², σ_θ²])
            Typical values: diag([0.1, 0.1, 0.05])**2
        Q : ndarray of shape (3, 3)
            Measurement noise covariance matrix:
            Q = diag([σ_r², σ_φ², σ_s²]) for range, bearing, signature
            Typical values: diag([0.1, 0.05, 1e6])**2
        plot : bool
            Whether to display final trajectory and map visualization.
        plot_inter : bool
            Whether to display intermediate plots during processing.
            If True, plots every 200 frames after frame 800.

        Notes
        -----
        The algorithm processes data chronologically, alternating between:
        1. Motion updates (when data[1] == -1, indicating odometry)
        2. Measurement updates (when data[1] > 0, indicating landmark observation)

        State initialization uses ground truth for robot pose and large uncertainty
        for all landmark positions until they are first observed.

        Examples
        --------
        >>> import numpy as np
        >>> R = np.diag([0.1, 0.1, 0.05])**2  # Process noise
        >>> Q = np.diag([0.1, 0.05, 1e6])**2  # Measurement noise
        >>> slam = ExtendedKalmanFilterSLAM(
        ...     "data/MRCLAM_Dataset1", "Robot1", 800, 3200, R, Q, True, False
        ... )
        """
        self.load_data(dataset, robot, start_frame, end_frame)
        self.initialization(R, Q)
        for data in self.data:
            if data[1] == -1:
                self.motion_update(data)
            else:
                self.measurement_update(data)
            # Plot every n frames
            if plot and plot_inter:
                if (
                    len(self.states) > (800 - start_frame)
                    and len(self.states) % 200 == 0
                ):
                    self.plot_data()
        if plot:
            self.plot_data()

    def load_data(self, dataset, robot, start_frame, end_frame):
        """
        Load and preprocess MRCLAM dataset files.

        This method loads all necessary data files from the MRCLAM dataset,
        synchronizes odometry and measurement data by timestamp, and creates
        lookup tables for efficient landmark correspondence.

        Parameters
        ----------
        dataset : str
            Path to dataset directory containing MRCLAM files.
        robot : str
            Robot identifier ("Robot1", "Robot2", etc.).
        start_frame : int
            Starting frame index (adjusted to ensure first frame is odometry).
        end_frame : int
            Ending frame index for data processing.

        Notes
        -----
        The method processes the following MRCLAM files:
        - Barcodes.dat: Maps barcode IDs to landmark IDs
        - {Robot}_Groundtruth.dat: True robot trajectory [Time, x, y, θ]
        - Landmark_Groundtruth.dat: True landmark positions [ID, x, y]
        - {Robot}_Measurement.dat: Range/bearing observations [Time, ID, r, φ]
        - {Robot}_Odometry.dat: Velocity commands [Time, ID, v, ω]

        Data Structure Created:
        ----------------------
        self.data : Combined and sorted measurement/odometry data
            Format: [Time, Subject_ID, data1, data2]
            - Subject_ID = -1: Odometry data [Time, -1, v, ω]
            - Subject_ID > 0: Measurement [Time, ID, range, bearing]

        Lookup Tables:
        -------------
        self.landmark_indexes : dict
            Maps barcode ID to internal landmark index (1-15)
        self.landmark_locations : dict
            Maps barcode ID to ground truth coordinates (for evaluation only)
        self.landmark_observed : ndarray
            Boolean array tracking observation status of each landmark

        The method ensures the first processed frame is always odometry data
        to properly initialize the motion model.
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

        # Select data according to start_frame and end_frame
        # Fisrt frame must be control input
        while self.data[start_frame][1] != -1:
            start_frame += 1
        # Remove all data before start_frame and after the end_timestamp
        self.data = self.data[start_frame:end_frame]
        start_timestamp = self.data[0][0]
        end_timestamp = self.data[-1][0]
        # Remove all groundtruth outside the range
        for i in range(len(self.groundtruth_data)):
            if self.groundtruth_data[i][0] >= end_timestamp:
                break
        self.groundtruth_data = self.groundtruth_data[:i]
        for i in range(len(self.groundtruth_data)):
            if self.groundtruth_data[i][0] >= start_timestamp:
                break
        self.groundtruth_data = self.groundtruth_data[i:]

        # Combine barcode Subject# with landmark Subject#
        # Lookup table to map barcode Subjec# to landmark coordinates
        # [x[m], y[m], x std-dev[m], y std-dev[m]]
        # Ground truth data is not used in EKF SLAM
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

        # Table to record if each landmark has been seen or not
        # Element [0] is not used. [1] - [15] represent for landmark# 6 - 20
        self.landmark_observed = np.full(len(self.landmark_indexes) + 1, False)

    def initialization(self, R, Q):
        """
        Initialize EKF-SLAM state vector and covariance matrix.

        Sets up the augmented state vector containing robot pose and all
        possible landmark positions, along with the corresponding covariance
        matrix that captures uncertainties and correlations.

        Parameters
        ----------
        R : ndarray of shape (3, 3)
            Process noise covariance for robot motion model.
        Q : ndarray of shape (3, 3)
            Measurement noise covariance for observations.

        Notes
        -----
        State Vector Structure (dimension 3 + 2N):
        - Indices 0-2: Robot pose [x, y, θ]
        - Indices 3+2i, 4+2i: Landmark i coordinates [x_i, y_i]

        Covariance Matrix Initialization:
        - Robot pose: Small uncertainty (using ground truth initialization)
        - Landmarks: Large uncertainty (1e6) until first observed
        - Cross-correlations: Small values (1e-6) initially

        The large landmark uncertainties represent complete ignorance about
        landmark positions until they are first observed and initialized.

        State Augmentation:
        When a landmark is first observed, its position is computed as:
            x_landmark = x_robot + range * cos(bearing + θ_robot)
            y_landmark = y_robot + range * sin(bearing + θ_robot)

        This initialization is critical for the EKF linearization to be valid.
        """
        # Initial state: 3 for robot, 2 for each landmark
        self.states = np.zeros((1, 3 + 2 * len(self.landmark_indexes)))
        self.states[0][:3] = self.groundtruth_data[0][1:]
        self.last_timestamp = self.groundtruth_data[0][0]
        self.stamps = []
        self.stamps.append(self.last_timestamp)

        # EKF state covariance: (3 + 2n) x (3 + 2n)
        # For robot states, use first ground truth data as initial value
        #   - small values for top-left 3 x 3 matrix
        # For landmark states, we have no information at the beginning
        #   - large values for rest of variances (diagonal) data
        #   - small values for all covariances (off-diagonal) data
        self.sigma = 1e-6 * np.full(
            (3 + 2 * len(self.landmark_indexes), 3 + 2 * len(self.landmark_indexes)), 1
        )
        for i in range(3, 3 + 2 * len(self.landmark_indexes)):
            self.sigma[i][i] = 1e6

        # State covariance matrix
        self.R = R
        # Measurement covariance matrix
        self.Q = Q

    def motion_update(self, control):
        """
        Perform EKF-SLAM motion update (prediction step).

        Updates the robot pose portion of the state vector using the bicycle
        motion model, while leaving landmark positions unchanged. Propagates
        uncertainty through the linearized motion model.

        Parameters
        ----------
        control : ndarray
            Control input: [timestamp, -1, v, ω]
            - v: Linear velocity (m/s)
            - ω: Angular velocity (rad/s)

        Mathematical Model
        ------------------
        Motion equations (bicycle model):
            x_{t+1} = x_t + v * cos(θ_t) * Δt
            y_{t+1} = y_t + v * sin(θ_t) * Δt
            θ_{t+1} = θ_t + ω * Δt

        Linearization:
        The motion Jacobian G_t has the structure:
            G_t = [ G_robot    0    ]
                  [   0     I_2N  ]

        Where G_robot (3×3) is:
            [1  0  -v*Δt*sin(θ)]
            [0  1   v*Δt*cos(θ)]
            [0  0        1      ]

        And I_2N is the 2N×2N identity for unchanged landmarks.

        Covariance Update:
            Σ̄_t = G_t * Σ_{t-1} * G_t^T + F_x^T * R * F_x

        Where F_x maps 3D robot noise to full (3+2N)D state space.

        Notes
        -----
        - Skips update if Δt < 0.001 to avoid numerical issues
        - Wraps angle θ to [-π, π] to maintain proper representation
        - Only robot pose uncertainty increases; landmark uncertainty unchanged
        - Creates new state vector while preserving landmark estimates

        The motion update maintains all landmark-landmark and robot-landmark
        correlations established during previous measurement updates.
        """
        # ------------------ Step 1: Mean update ---------------------#
        # State: [x, y, θ, x_l1, y_l1, ......, x_ln, y_ln]
        # Control: [v, w]
        # Only robot state is updated during each motion update step!
        # [x_t, y_t, θ_t] = g(u_t, x_t-1)
        #   x_t  =  x_t-1 + v * cosθ_t-1 * delta_t
        #   y_t  =  y_t-1 + v * sinθ_t-1 * delta_t
        #   θ_t  =  θ_t-1 + w * delta_t
        # Skip motion update if two odometry data are too close
        delta_t = control[0] - self.last_timestamp
        if delta_t < 0.001:
            return
        # Compute updated [x, y, theta]
        x_t = self.states[-1][0] + control[2] * np.cos(self.states[-1][2]) * delta_t
        y_t = self.states[-1][1] + control[2] * np.sin(self.states[-1][2]) * delta_t
        theta_t = self.states[-1][2] + control[3] * delta_t
        # Limit θ within [-pi, pi]
        if theta_t > np.pi:
            theta_t -= 2 * np.pi
        elif theta_t < -np.pi:
            theta_t += 2 * np.pi
        self.last_timestamp = control[0]
        # Append new state
        new_state = np.copy(self.states[-1])
        new_state[0] = x_t
        new_state[1] = y_t
        new_state[2] = theta_t
        self.states = np.append(self.states, np.array([new_state]), axis=0)
        self.stamps.append(self.last_timestamp)

        # ------ Step 2: Linearize state-transition by Jacobian ------#
        # Jacobian of motion: G = d g(u_t, x_t-1) / d x_t-1
        #         1  0  -v * delta_t * sinθ_t-1
        #   G  =  0  1   v * delta_t * cosθ_t-1        0
        #         0  0             1
        #
        #                      0                    I(2n x 2n)
        self.G = np.identity(3 + 2 * len(self.landmark_indexes))
        self.G[0][2] = -control[2] * delta_t * np.sin(self.states[-2][2])
        self.G[1][2] = control[2] * delta_t * np.cos(self.states[-2][2])

        # ---------------- Step 3: Covariance update ------------------#
        # sigma = G x sigma x G.T + Fx.T x R x Fx
        self.sigma = self.G.dot(self.sigma).dot(self.G.T)
        self.sigma[0][0] += self.R[0][0]
        self.sigma[1][1] += self.R[1][1]
        self.sigma[2][2] += self.R[2][2]

    def measurement_update(self, measurement):
        """
        Perform EKF-SLAM measurement update (correction step).

        Processes a landmark observation to update estimates of robot pose
        and all landmarks through the correlated covariance structure.
        Initializes new landmarks when first observed.

        Parameters
        ----------
        measurement : ndarray
            Measurement data: [timestamp, landmark_id, range, bearing]
            - landmark_id: Barcode ID of observed landmark
            - range: Distance to landmark (meters)
            - bearing: Angle to landmark relative to robot heading (radians)

        Mathematical Model
        ------------------
        Measurement equations:
            r = √[(m_x - x)² + (m_y - y)²] + v_r
            φ = atan2(m_y - y, m_x - x) - θ + v_φ

        Where (m_x, m_y) are landmark coordinates and v_r, v_φ are noise terms.

        Measurement Jacobian H_t:
        The Jacobian maps from full state space to measurement space:
            H_t = h_low * F_x

        Where h_low (3×5) operates on [x, y, θ, m_x, m_y]:
            h_low = [[-Δx/√q  -Δy/√q   0   Δx/√q   Δy/√q ]
                     [ Δy/q   -Δx/q   -1  -Δy/q    Δx/q ]
                     [  0       0      0    0       0   ]]

        And F_x (5 × (3+2N)) selects relevant state components.

        State Augmentation:
        When landmark first observed (landmark_observed[idx] == False):
            m_x = x + r * cos(φ + θ)
            m_y = y + r * sin(φ + θ)

        Information Propagation:
        The Kalman update affects the entire state vector:
            y_new = y_old + K * innovation
            Σ_new = (I - K*H) * Σ_old

        This propagates information from the landmark observation to:
        1. Robot pose estimate (direct effect)
        2. All previously observed landmarks (through correlations)

        Notes
        -----
        - Returns early if landmark_id not in landmark_indexes
        - Landmark initialization only occurs on first observation
        - Innovation vector includes range and bearing residuals
        - Third measurement component (signature) set to zero
        - Kalman gain size is (3+2N) × 3, enabling full state update

        The measurement update is where the fundamental SLAM insight occurs:
        observing any landmark improves the entire map through correlations.
        """
        # Continue if landmark is not found in self.landmark_indexes
        if measurement[1] not in self.landmark_indexes:
            return

        # Get current robot state, measurement and landmark index
        x_t = self.states[-1][0]
        y_t = self.states[-1][1]
        theta_t = self.states[-1][2]
        range_t = measurement[2]
        bearing_t = measurement[3]
        landmark_idx = self.landmark_indexes[measurement[1]]

        # If this landmark has never been seen before: initialize landmark location in the state vector as the observed one
        #   x_l = x_t + range_t * cos(bearing_t + theta_t)
        #   y_l = y_t + range_t * sin(bearing_t + theta_t)
        if not self.landmark_observed[landmark_idx]:
            x_l = x_t + range_t * np.cos(bearing_t + theta_t)
            y_l = y_t + range_t * np.sin(bearing_t + theta_t)
            self.states[-1][2 * landmark_idx + 1] = x_l
            self.states[-1][2 * landmark_idx + 2] = y_l
            self.landmark_observed[landmark_idx] = True
        # Else use current value in state vector
        else:
            x_l = self.states[-1][2 * landmark_idx + 1]
            y_l = self.states[-1][2 * landmark_idx + 2]

        # ---------------- Step 1: Measurement update -----------------#
        #   range   =  sqrt((x_l - x_t)^2 + (y_l - y_t)^2)
        #  bearing  =  atan2((y_l - y_t) / (x_l - x_t)) - θ_t
        delta_x = x_l - x_t
        delta_y = y_l - y_t
        q = delta_x**2 + delta_y**2
        range_expected = np.sqrt(q)
        bearing_expected = np.arctan2(delta_y, delta_x) - theta_t

        # ------ Step 2: Linearize Measurement Model by Jacobian ------#
        # Landmark state becomes a variable in measurement model
        # Jacobian: H = d h(x_t, x_l) / d (x_t, x_l)
        #        1 0 0  0 ...... 0   0 0   0 ...... 0
        #        0 1 0  0 ...... 0   0 0   0 ...... 0
        # F_x =  0 0 1  0 ...... 0   0 0   0 ...... 0
        #        0 0 0  0 ...... 0   1 0   0 ...... 0
        #        0 0 0  0 ...... 0   0 1   0 ...... 0
        #          (2*landmark_idx - 2)
        #          -delta_x/√q  -delta_y/√q  0  delta_x/√q  delta_y/√q
        # H_low =   delta_y/q   -delta_x/q  -1  -delta_y/q  delta_x/q
        #               0            0       0       0          0
        # H = H_low x F_x
        F_x = np.zeros((5, 3 + 2 * len(self.landmark_indexes)))
        F_x[0][0] = 1.0
        F_x[1][1] = 1.0
        F_x[2][2] = 1.0
        F_x[3][2 * landmark_idx + 1] = 1.0
        F_x[4][2 * landmark_idx + 2] = 1.0
        H_1 = np.array(
            [
                -delta_x / np.sqrt(q),
                -delta_y / np.sqrt(q),
                0,
                delta_x / np.sqrt(q),
                delta_y / np.sqrt(q),
            ]
        )
        H_2 = np.array([delta_y / q, -delta_x / q, -1, -delta_y / q, delta_x / q])
        H_3 = np.array([0, 0, 0, 0, 0])
        self.H = np.array([H_1, H_2, H_3]).dot(F_x)

        # ---------------- Step 3: Kalman gain update -----------------#
        S_t = self.H.dot(self.sigma).dot(self.H.T) + self.Q
        self.K = self.sigma.dot(self.H.T).dot(np.linalg.inv(S_t))

        # ------------------- Step 4: mean update ---------------------#
        difference = np.array(
            [range_t - range_expected, bearing_t - bearing_expected, 0]
        )
        innovation = self.K.dot(difference)
        new_state = self.states[-1] + innovation
        self.states = np.append(self.states, np.array([new_state]), axis=0)
        self.last_timestamp = measurement[0]
        self.stamps.append(self.last_timestamp)

        # ---------------- Step 5: covariance update ------------------#
        self.sigma = (
            np.identity(3 + 2 * len(self.landmark_indexes)) - self.K.dot(self.H)
        ).dot(self.sigma)

    def plot_data(self):
        """
        Visualize current SLAM estimates and ground truth data.

        Creates a comprehensive plot showing:
        - Robot trajectory (estimated vs ground truth)
        - Landmark positions (estimated vs ground truth)
        - Start/end points with distinctive markers
        - Landmark labels with corresponding IDs

        The visualization helps assess algorithm performance by comparing
        estimated trajectories and landmark positions against ground truth.

        Plot Elements
        -------------
        - Blue line: Ground truth robot trajectory
        - Red line: EKF-SLAM robot trajectory estimate
        - Green marker: Trajectory start point
        - Yellow marker: Trajectory end point
        - Black stars: Ground truth landmark positions (transparent)
        - Black dots: Estimated landmark positions
        - Text labels: Landmark IDs (barcode numbers 6-20)

        Notes
        -----
        - Only displays landmarks that have been observed
        - Landmark IDs shown are barcode numbers (6-20), not internal indices
        - Plot limits set to show full MRCLAM environment (-2 to 5.5m x, -7 to 7m y)
        - Uses matplotlib.pyplot.pause() for real-time display during processing

        The plot provides immediate visual feedback on:
        1. Trajectory tracking accuracy
        2. Landmark localization precision
        3. Map completeness (which landmarks observed)
        4. Overall SLAM performance
        """
        # Clear all
        plt.cla()

        # Ground truth data
        plt.plot(
            self.groundtruth_data[:, 1],
            self.groundtruth_data[:, 2],
            "b",
            label="Robot State Ground truth",
        )

        # States
        plt.plot(
            self.states[:, 0], self.states[:, 1], "r", label="Robot State Estimate"
        )

        # Start and end points
        plt.plot(
            self.groundtruth_data[0, 1],
            self.groundtruth_data[0, 2],
            "g8",
            markersize=12,
            label="Start point",
        )
        plt.plot(
            self.groundtruth_data[-1, 1],
            self.groundtruth_data[-1, 2],
            "y8",
            markersize=12,
            label="End point",
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
            label="Landmark Ground Truth",
        )

        # Landmark estimated locations
        estimate_xs = []
        estimate_ys = []
        for i in range(1, len(self.landmark_indexes) + 1):
            if self.landmark_observed[i]:
                estimate_xs.append(self.states[-1][2 * i + 1])
                estimate_ys.append(self.states[-1][2 * i + 2])
                plt.text(estimate_xs[-1], estimate_ys[-1], str(i + 5), fontsize=10)
        plt.scatter(
            estimate_xs, estimate_ys, s=50, c="k", marker=".", label="Landmark Estimate"
        )

        plt.title("EKF SLAM with known correspondences")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.xlim((-2.0, 5.5))
        plt.ylim((-7.0, 7.0))
        plt.pause(1e-16)

    def build_dataframes(self):
        """
        Convert SLAM results to pandas DataFrames for analysis.

        Creates time-indexed DataFrames from the SLAM state history and
        ground truth data to enable statistical analysis, error computation,
        and trajectory comparison.

        Creates
        -------
        self.gt : pandas.DataFrame
            Ground truth trajectory with columns ['x', 'y', 'theta']
            indexed by timestamp.
        self.robot_states : pandas.DataFrame
            Estimated robot trajectory with columns ['x', 'y', 'theta']
            indexed by timestamp.

        Notes
        -----
        The DataFrames enable efficient operations like:
        - Temporal alignment of ground truth and estimates
        - Error computation: error = estimates - ground_truth
        - Statistical analysis: mean, std, percentiles of errors
        - Trajectory plotting and visualization

        Time synchronization is handled by converting timestamps to
        pandas datetime objects and setting as index.

        See Also
        --------
        build_timeseries : Helper function for DataFrame construction
        """
        self.gt = build_timeseries(
            self.groundtruth_data, cols=["stamp", "x", "y", "theta"]
        )
        data = np.array(self.states[:, :3])
        stamp = np.expand_dims(self.stamps, axis=1)
        data_s = np.hstack([stamp, data])
        self.robot_states = build_timeseries(data_s, cols=["stamp", "x", "y", "theta"])


def build_timeseries(data, cols):
    """
    Convert numerical data array to time-indexed pandas DataFrame.

    Parameters
    ----------
    data : ndarray
        Data array where first column is timestamp in seconds.
    cols : list of str
        Column names for the DataFrame.

    Returns
    -------
    pandas.DataFrame
        Time-indexed DataFrame with timestamp converted to datetime.
    """
    timeseries = pd.DataFrame(data, columns=cols)
    timeseries["stamp"] = pd.to_datetime(timeseries["stamp"], unit="s")
    timeseries = timeseries.set_index("stamp")
    return timeseries


def build_state_timeseries(stamp, data, cols):
    """
    Build time-indexed DataFrame from separate timestamp and data arrays.

    Parameters
    ----------
    stamp : ndarray
        Timestamp array in seconds.
    data : ndarray
        Data array with state information.
    cols : list of str
        Column names for the DataFrame.

    Returns
    -------
    pandas.DataFrame
        Time-indexed DataFrame with timestamp converted to datetime.
    """
    timeseries = pd.DataFrame(data, columns=cols)
    timeseries["stamp"] = pd.to_datetime(stamp, unit="s")
    timeseries = timeseries.set_index("stamp")
    return timeseries


def filter_static_landmarks(lm, barcodes):
    """
    Filter measurement data to retain only static landmark observations.

    Converts barcode numbers to landmark numbers and filters out dynamic
    objects (robots) to keep only static landmarks for SLAM processing.

    Parameters
    ----------
    lm : pandas.DataFrame
        Measurement data with 'type' column indicating object type.
    barcodes : dict
        Mapping from barcode numbers to landmark numbers.

    Returns
    -------
    pandas.DataFrame
        Filtered measurement data containing only static landmarks.

    Notes
    -----
    Static landmarks in MRCLAM dataset have IDs > 5.
    Dynamic objects (robots) have IDs ≤ 5.
    """
    for L, l in dict(barcodes).items():  # Translate barcode num to landmark num
        lm[lm == l] = L
    lm = lm[lm.type > 5]  # Keep only static landmarks
    return lm


if __name__ == "__main__":
    """
    Example usage of EKF-SLAM with known correspondences.

    This example demonstrates running the algorithm on MRCLAM Dataset 1
    with typical parameter settings for indoor robotics environments.
    """
    # Dataset configuration
    dataset = "data/MRCLAM_Dataset1"
    robot = "Robot1"
    start_frame = 800
    end_frame = 3200

    # Process noise covariance matrix (robot motion uncertainty)
    # [x_noise², y_noise², θ_noise²]
    R = np.diagflat(np.array([5.0, 5.0, 100.0])) ** 2

    # Measurement noise covariance matrix (sensor uncertainty)
    # [range_noise², bearing_noise², signature_noise²]
    Q = np.diagflat(np.array([110.0, 110.0, 1e16])) ** 2

    # Run EKF-SLAM algorithm
    ekf_slam = ExtendedKalmanFilterSLAM(
        dataset, robot, start_frame, end_frame, R, Q, plot=True, plot_inter=False
    )
    plt.show()
