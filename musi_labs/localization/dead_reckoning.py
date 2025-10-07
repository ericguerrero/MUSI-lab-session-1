#!/usr/bin/env python3
"""
Dead Reckoning Localization Implementation.

This module implements the simplest form of robot localization using only
odometry data. Dead reckoning serves as a baseline algorithm for comparison
with more sophisticated localization methods like EKF and Particle Filter.

Dead reckoning accumulates motion estimates from wheel encoders or IMU data
without any external reference corrections, leading to unbounded error growth.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DeadReckoning:
    """
    Dead reckoning localization using only odometry measurements.

    Dead reckoning is the fundamental navigation method where robot pose is
    estimated by integrating motion commands over time. Starting from a known
    initial position, the robot's pose is updated based solely on odometry
    measurements without any external sensor corrections.

    Mathematical Foundation
    ----------------------
    The dead reckoning process follows the motion model:

        x_{t+1} = x_t + v * cos(θ_t) * Δt
        y_{t+1} = y_t + v * sin(θ_t) * Δt
        θ_{t+1} = θ_t + ω * Δt

    Where:
        - (x_t, y_t, θ_t): robot pose at time t (position and orientation)
        - (v, ω): control inputs (linear and angular velocity)
        - Δt: time step between measurements

    Error Characteristics
    --------------------
    Dead reckoning suffers from systematic error accumulation:

    1. **Systematic Errors**: Constant biases in wheel calibration, gear
       backlash, or wheel slippage cause errors that grow linearly with distance.

    2. **Random Errors**: Sensor noise, quantization errors, and measurement
       uncertainty cause errors that grow with the square root of distance.

    3. **Orientation Drift**: Small errors in angular velocity integration
       cause orientation drift, which significantly affects position accuracy
       over long trajectories.

    Error Growth Model:
        σ_position(t) ≈ σ_systematic * d(t) + σ_random * √d(t)

    Where d(t) is the total distance traveled.

    Comparison with Advanced Methods
    -------------------------------
    **Dead Reckoning vs. Extended Kalman Filter (EKF)**:
    - Dead reckoning: No sensor fusion, unbounded error growth
    - EKF: Fuses odometry with landmark observations, bounded error

    **Dead Reckoning vs. Particle Filter**:
    - Dead reckoning: Single trajectory hypothesis
    - Particle Filter: Multiple trajectory hypotheses, robust to ambiguity

    **When to Use Dead Reckoning**:
    - Short-term navigation between sensor updates
    - Sensor fusion prediction step
    - Baseline for algorithm comparison
    - Emergency navigation when sensors fail

    Parameters
    ----------
    dataset : str
        Path to MRCLAM dataset directory containing odometry and ground truth data.
    robot : str
        Robot identifier (e.g., 'Robot1', 'Robot2', etc.).
    end_frame : int
        Maximum number of odometry frames to process.
    plot : bool, optional
        Whether to generate visualization plots. Default: True.

    Attributes
    ----------
    states : ndarray, shape (T, 4)
        Estimated robot trajectory [timestamp, x, y, θ] over time.
    groundtruth_data : ndarray
        True robot poses from motion capture system for comparison.
    data : ndarray
        Combined and sorted odometry and measurement data.
    landmark_locations : dict
        Mapping from landmark IDs to positions {id: [x, y]}.

    Examples
    --------
    Basic dead reckoning on MRCLAM Dataset1:

    >>> dr = DeadReckoning(
    ...     dataset="data/MRCLAM_Dataset1",
    ...     robot="Robot1",
    ...     end_frame=1000
    ... )
    >>> dr.run()
    >>> print(f"Final position error: {np.linalg.norm(dr.states[-1, 1:3] - dr.groundtruth_data[-1, 1:3]):.2f} m")

    Trajectory analysis:

    >>> # Calculate cumulative position error
    >>> errors = np.linalg.norm(
    ...     dr.states[:, 1:3] - dr.groundtruth_data[:len(dr.states), 1:3],
    ...     axis=1
    ... )
    >>> plt.plot(errors)
    >>> plt.xlabel('Time Step')
    >>> plt.ylabel('Position Error (m)')
    >>> plt.title('Dead Reckoning Error Growth')

    Notes
    -----
    This implementation:
    - Uses the bicycle motion model (no slip assumption)
    - Integrates odometry measurements without noise modeling
    - Serves as ground truth for motion model validation
    - Provides baseline for localization algorithm comparison

    Limitations:
    - No error correction or sensor fusion
    - Unbounded error accumulation over time
    - Sensitive to systematic odometry biases
    - Cannot handle wheel slippage or skidding

    For practical robot navigation, dead reckoning should be combined with:
    - Landmark-based localization (EKF, Particle Filter)
    - GPS or other absolute positioning systems
    - Inertial navigation systems (INS)
    - Visual odometry or SLAM

    References
    ----------
    .. [1] Borenstein, J., & Feng, L. (1996). Measurement and correction of
           systematic odometry errors in mobile robots. IEEE Transactions on
           Robotics and Automation, 12(6), 869-880.
    .. [2] Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic Robotics.
           MIT Press. Chapter 5: Robot Motion.
    .. [3] Siegwart, R., Nourbakhsh, I. R., & Scaramuzza, D. (2011).
           Introduction to Autonomous Mobile Robots. MIT Press.
    """

    def __init__(self, dataset, robot, end_frame, plot=True):
        self.load_data(dataset, robot, end_frame)
        self.plot = plot

    def load_data(self, dataset, robot, end_frame):
        """
        Load and preprocess MRCLAM dataset for dead reckoning navigation.

        Loads robot odometry data, ground truth poses, and landmark positions
        from the MRCLAM dataset. The data is synchronized by timestamp to ensure
        proper temporal alignment between odometry commands and ground truth.

        Data Processing Pipeline
        -----------------------
        1. Load all MRCLAM data files (odometry, measurements, ground truth)
        2. Merge odometry and measurement streams (measurement data not used)
        3. Sort combined data by timestamp for chronological processing
        4. Synchronize with ground truth (align start times)
        5. Truncate to specified end_frame for computational efficiency
        6. Create landmark lookup tables for visualization

        Parameters
        ----------
        dataset : str
            Path to MRCLAM dataset directory.
        robot : str
            Robot identifier (e.g., 'Robot1').
        end_frame : int
            Maximum number of data frames to process.

        Notes
        -----
        The MRCLAM dataset contains:
        - Odometry: [timestamp, subject_id, v, ω] at ~67 Hz
        - Ground truth: [timestamp, x, y, θ] at 100 Hz from Vicon
        - Measurements: [timestamp, landmark_id, range, bearing] (ignored)
        - Landmarks: Static cylindrical beacons with barcode IDs

        Data synchronization ensures that:
        - First odometry command aligns with first ground truth pose
        - Timestamps are monotonically increasing
        - Dead reckoning starts from known initial position
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

        # Handle 1D arrays from single-row files (reshape to 2D)
        if self.landmark_groundtruth_data.ndim == 1:
            self.landmark_groundtruth_data = self.landmark_groundtruth_data.reshape(1, -1)
        if self.measurement_data.ndim == 1 and len(self.measurement_data) > 0:
            self.measurement_data = self.measurement_data.reshape(1, -1)
        elif self.measurement_data.ndim == 1:  # Empty file
            self.measurement_data = self.measurement_data.reshape(0, 4)

        # Collect all input data and sort by timestamp
        # Add subject "odom" = -1 for odometry data
        odom_data = np.insert(self.odometry_data, 1, -1, axis=1)

        self.data = np.concatenate((odom_data, self.measurement_data), axis=0)
        self.data = self.data[np.argsort(self.data[:, 0])]

        # Remove all data before the fisrt timestamp of groundtruth
        # Use first groundtruth data as the initial location of the robot
        for i in range(len(self.data)):
            if self.data[i][1] == -1:
                if self.data[i][0] > self.groundtruth_data[0][0]:
                    break
        self.data = self.data[i:]

        for i in range(len(self.groundtruth_data)):
            if self.groundtruth_data[i][0] > self.data[0][0]:
                break
        self.groundtruth_data = self.groundtruth_data[i:]
        for i in range(len(self.data)):
            if self.data[i][1] == -1:
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

    def run(self):
        """
        Execute complete dead reckoning navigation algorithm.

        Processes all odometry data sequentially to estimate robot trajectory
        through integration of motion commands. This is the main entry point
        for dead reckoning localization.

        Algorithm Steps
        --------------
        1. Initialize robot pose with first ground truth position
        2. For each odometry measurement:
           - Extract control inputs (v, ω)
           - Apply motion model to update pose estimate
           - Record updated pose in trajectory history
        3. Generate visualization if requested

        Processing Flow
        --------------
        The algorithm processes data chronologically:
        - Odometry data (subject_id = -1): Update pose estimate
        - Measurement data (subject_id > 0): Ignored for dead reckoning

        Notes
        -----
        - Only processes odometry commands, ignores sensor measurements
        - Maintains complete trajectory history for analysis
        - Provides baseline performance for comparison with other methods
        - Demonstrates pure integration without sensor corrections
        """
        self.initialization()
        for data in self.data:
            if data[1] == -1:
                self.motion_update(data)
        if self.plot:
            self.plot_data()

    def initialization(self):
        """
        Initialize dead reckoning with known starting pose.

        Sets the initial robot pose using the first ground truth measurement
        from the motion capture system. This provides a fair comparison with
        other localization algorithms that also start with known initial pose.

        Initialization Process
        ---------------------
        - Extract first ground truth pose: [timestamp, x_0, y_0, θ_0]
        - Initialize trajectory with this pose
        - Set timestamp for synchronization with odometry data

        Notes
        -----
        - Assumes accurate initial pose (common in indoor robotics)
        - Alternative: Initialize with approximate pose and study convergence
        - Initial pose quality significantly affects final trajectory accuracy
        - Ground truth initialization provides best-case starting condition
        """
        # Initial state
        self.states = np.array([self.groundtruth_data[0]])
        self.last_timestamp = self.states[-1][0]

    def motion_update(self, control):
        """
        Update robot pose using bicycle motion model integration.

        Integrates a single odometry measurement to update the robot's pose
        estimate. This implements the core dead reckoning computation using
        the bicycle motion model without any noise or error correction.

        Motion Model
        -----------
        State update equations:

            x_{t+1} = x_t + v * cos(θ_t) * Δt
            y_{t+1} = y_t + v * sin(θ_t) * Δt
            θ_{t+1} = θ_t + ω * Δt

        Where:
            - (x_t, y_t): robot position at time t
            - θ_t: robot orientation at time t
            - v: linear velocity (m/s)
            - ω: angular velocity (rad/s)
            - Δt: time step duration

        The model assumes:
        - Constant velocity between measurements
        - No wheel slippage or skidding
        - Perfect odometry measurements
        - Bicycle kinematics (point robot)

        Parameters
        ----------
        control : array_like, shape (4,)
            Control data [timestamp, subject_id, v, ω] where:
            - timestamp: time of odometry measurement
            - subject_id: -1 for odometry data
            - v: measured linear velocity
            - ω: measured angular velocity

        Notes
        -----
        - Skips update if Δt < 0.001s to avoid numerical issues
        - Normalizes orientation to [-π, π] range
        - Updates trajectory history with new pose estimate
        - No uncertainty propagation (deterministic integration)

        Error Sources
        ------------
        In practice, dead reckoning errors arise from:
        - Wheel encoder quantization and calibration errors
        - Wheel slippage and terrain irregularities
        - Gear backlash and mechanical compliance
        - Systematic biases in wheel diameter or wheelbase measurements
        """
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

    def plot_data(self):
        """
        Visualize dead reckoning results compared to ground truth.

        Creates a comparison plot showing the estimated trajectory from dead
        reckoning alongside the true robot path from motion capture. This
        visualization clearly demonstrates error accumulation in pure odometry.

        Plot Elements
        ------------
        - **Ground Truth**: Blue line showing true robot trajectory
        - **Dead Reckoning**: Red line showing estimated trajectory
        - **Start Point**: Green circle marking trajectory beginning
        - **End Point**: Yellow circle marking trajectory conclusion
        - **Landmarks**: Black stars with ID labels for spatial reference

        Error Analysis
        -------------
        The visualization reveals typical dead reckoning characteristics:
        - Initial accuracy (small errors near start)
        - Gradual drift accumulation
        - Orientation errors affecting subsequent position estimates
        - Systematic biases creating consistent error patterns

        Comparison Metrics
        -----------------
        Visual inspection can assess:
        - Final position error magnitude
        - Error growth rate over distance/time
        - Systematic vs. random error components
        - Trajectory shape preservation vs. distortion

        Notes
        -----
        - Useful for algorithm validation and comparison
        - Demonstrates need for sensor fusion in long-term navigation
        - Error magnitude depends on odometry quality and trajectory length
        - Typical errors: 1-10% of distance traveled for good odometry
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
        plt.title("Dead Reckoning")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()

    def represent_dataset(self):
        """
        Visualize the MRCLAM dataset structure and environment layout.

        Creates a clean visualization of the ground truth robot trajectory
        and landmark positions to understand the experimental environment.
        This provides context for localization algorithm evaluation.

        Visualization Elements
        ---------------------
        - **Ground Truth Trajectory**: Complete robot path through environment
        - **Start Position**: Green 'x' marker
        - **End Position**: Red 'x' marker
        - **Static Landmarks**: Black stars with numerical IDs

        Dataset Context
        --------------
        The MRCLAM environment features:
        - 15m × 8m indoor laboratory space
        - 15 cylindrical landmarks with unique barcode IDs
        - Complex trajectory with loops and varied motion patterns
        - High-precision ground truth from Vicon motion capture

        Notes
        -----
        - Helps understand algorithm challenges (loops, ambiguous areas)
        - Shows landmark distribution and potential for localization
        - Useful for experimental design and result interpretation
        - Independent of any localization algorithm results
        """
        # Ground truth data
        plt.plot(
            self.groundtruth_data[:, 1],
            self.groundtruth_data[:, 2],
            "b",
            label="Robot State Ground truth",
        )

        # Start and end points
        plt.plot(
            self.groundtruth_data[0, 1],
            self.groundtruth_data[0, 2],
            "gx",
            label="Start point",
        )
        plt.plot(
            self.groundtruth_data[-1, 1],
            self.groundtruth_data[-1, 2],
            "rx",
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
            label="Landmark Locations",
        )

        plt.title("Robot Groundtruth and Map")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.show()

    def build_dataframes(self):
        """
        Convert raw data arrays to pandas DataFrames with time indexing.

        Creates structured pandas DataFrames from numpy arrays for advanced
        data analysis and visualization. Time series indexing enables easy
        temporal analysis and data alignment between different sensors.

        Generated DataFrames
        -------------------
        - **gt**: Ground truth poses with timestamp index
        - **states**: Dead reckoning estimates with timestamp index
        - **measurements**: All sensor data (odometry + observations)
        - **motion**: Filtered odometry data only
        - **sensor**: Filtered landmark observations only

        Data Processing
        --------------
        - Converts timestamps to datetime objects
        - Sets time-based indexing for efficient temporal operations
        - Separates odometry (type=-1) from landmark measurements
        - Filters static landmarks (ID > 5) from robot observations

        Notes
        -----
        - Enables pandas-based analysis and plotting
        - Facilitates data export and external analysis
        - Supports time-aligned comparison between algorithms
        - Optional feature for advanced users and research
        """
        self.gt = build_timeseries(
            self.groundtruth_data, cols=["stamp", "x", "y", "theta"]
        )
        self.states = build_timeseries(self.states, cols=["stamp", "x", "y", "theta"])
        self.measurements = build_timeseries(
            self.data, cols=["stamp", "type", "range_l", "bearing_l"]
        )
        self.motion = self.measurements[self.measurements.type == -1].rename(
            columns={"range_l": "v", "bearing_l": "omega"}
        )
        landmarks = self.measurements[self.measurements.type != -1]
        self.sensor = filter_static_landmarks(landmarks, self.barcodes_data)

    def transform_landmarks(self):
        """
        Transform landmark observations from robot to global coordinates.

        Converts range-bearing observations from the robot's local coordinate
        frame to global map coordinates using ground truth poses. This enables
        comparison of observed vs. true landmark positions for sensor validation.

        Coordinate Transformation
        ------------------------
        For each landmark observation (r, φ) at robot pose (x_t, y_t, θ_t):

        1. Convert to robot-relative Cartesian:
            x_rel = r * cos(φ)
            y_rel = r * sin(φ)

        2. Transform to global coordinates:
            x_global = x_t + x_rel * cos(θ_t) - y_rel * sin(θ_t)
            y_global = y_t + x_rel * sin(θ_t) + y_rel * cos(θ_t)

        Applications
        -----------
        - Sensor calibration and validation
        - Map building from sensor observations
        - Data association analysis
        - Measurement model verification

        Notes
        -----
        - Uses ground truth poses for transformation (ideal case)
        - In practice, estimated poses would be used (with uncertainty)
        - Results stored in sensor_gt DataFrame with x_l, y_l columns
        - Enables comparison with known landmark positions
        """
        self.sensor_gt = self.sensor.join(self.gt).dropna()
        range_l = self.sensor_gt.range_l
        bearing_l = self.sensor_gt.bearing_l
        x_t = self.sensor_gt.x
        y_t = self.sensor_gt.y
        theta_t = self.sensor_gt.theta

        x = range_l * np.cos(bearing_l)
        y = range_l * np.sin(bearing_l)

        self.sensor_gt["x_l"] = x_t + x * np.cos(theta_t) - y * np.sin(theta_t)
        self.sensor_gt["y_l"] = y_t + x * np.sin(theta_t) + y * np.cos(theta_t)


def build_timeseries(data, cols):
    """
    Convert numpy array to pandas DataFrame with datetime indexing.

    Utility function to create time-indexed DataFrames from MRCLAM data arrays.
    Converts Unix timestamps to pandas datetime objects for time series analysis.

    Parameters
    ----------
    data : ndarray
        Input data array where first column contains timestamps.
    cols : list of str
        Column names for the DataFrame.

    Returns
    -------
    pandas.DataFrame
        Time-indexed DataFrame with datetime index.

    Notes
    -----
    - Assumes timestamps are in Unix epoch format (seconds since 1970)
    - Sets timestamp column as DataFrame index
    - Enables pandas time series operations and plotting
    """
    timeseries = pd.DataFrame(data, columns=cols)
    timeseries["stamp"] = pd.to_datetime(timeseries["stamp"], unit="s")
    timeseries = timeseries.set_index("stamp")
    return timeseries


def filter_static_landmarks(lm, barcodes):
    """
    Filter landmark observations to include only static environmental features.

    Processes measurement data to extract observations of static landmarks
    (cylindrical beacons) while excluding robot-to-robot observations.
    Uses barcode mapping to translate between measurement IDs and landmark IDs.

    Parameters
    ----------
    lm : pandas.DataFrame
        Landmark measurement data with 'type' column containing subject IDs.
    barcodes : ndarray
        Barcode mapping table from MRCLAM dataset.

    Returns
    -------
    pandas.DataFrame
        Filtered measurements containing only static landmark observations.

    Notes
    -----
    - Static landmarks have IDs > 5 in MRCLAM dataset
    - Robot observations (IDs 1-5) are filtered out
    - Barcode translation maps sensor IDs to landmark IDs
    - Essential for localization algorithms using known landmarks
    """
    for L, l in dict(barcodes).items():  # Translate barcode num to landmark num
        lm[lm == l] = L
    lm = lm[lm.type > 5]  # Keep only static landmarks
    return lm


if __name__ == "__main__":
    # Example usage of Dead Reckoning localization
    dataset = "data/MRCLAM_Dataset1"
    end_frame = 3200
    robot = "Robot1"

    # Create and run dead reckoning algorithm
    dr = DeadReckoning(dataset, robot, end_frame)
    dr.run()

    # Calculate final position error
    final_error = np.linalg.norm(dr.states[-1, 1:3] - dr.groundtruth_data[-1, 1:3])
    print(f"Final position error: {final_error:.2f} meters")

    # Show trajectory comparison
    plt.show()
