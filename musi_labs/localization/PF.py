#!/usr/bin/env python3
"""
Implementation of Particle Filter Localization with known correspondences.
See Probabilistic Robotics:
    1. Page 252, Table 8.2 for main algorithm.
    2. Page 124, Table 5.3 for motion model.
    3. Page 179, Table 6.4 for measurement model.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


class ParticleFilter:
    """
    Monte Carlo Localization using Particle Filter for mobile robot pose estimation.

    This implementation follows the Monte Carlo Localization (MCL) algorithm from
    Probabilistic Robotics (Thrun et al.), Chapter 8, Table 8.2. The particle filter
    represents the posterior belief bel(x_t) by a set of weighted particles, where
    each particle represents a hypothesis of the robot's pose.

    Mathematical Foundation
    ----------------------
    The particle filter approximates the posterior distribution:

        p(x_t | z_{1:t}, u_{1:t}) ≈ {x_t^[1], x_t^[2], ..., x_t^[M]}

    Where:
        - x_t^[m] represents the m-th particle (pose hypothesis) at time t
        - M is the total number of particles
        - Each particle has an associated importance weight w_t^[m]

    Algorithm Steps:
        1. **Prediction**: Sample new poses from motion model p(x_t | u_t, x_{t-1}^[m])
        2. **Update**: Compute importance weights w_t^[m] = p(z_t | x_t^[m])
        3. **Resampling**: Draw M particles with replacement proportional to weights

    Motion Model (Bicycle Model)
    ---------------------------
    State transition function g(u_t, x_{t-1}):

        x_t = x_{t-1} + v * cos(θ_{t-1}) * Δt + ε_x
        y_t = y_{t-1} + v * sin(θ_{t-1}) * Δt + ε_y
        θ_t = θ_{t-1} + ω * Δt + ε_θ

    Where:
        - (x, y, θ) represents robot pose (position and orientation)
        - (v, ω) represents control input (linear and angular velocity)
        - (ε_x, ε_y, ε_θ) represents motion noise
        - Δt is the time step

    Measurement Model
    ----------------
    Range-bearing observations to known landmarks:

        r = √[(x_l - x_t)² + (y_l - y_t)²] + ε_r
        φ = atan2(y_l - y_t, x_l - x_t) - θ_t + ε_φ

    Where:
        - (x_l, y_l) is the landmark position
        - (x_t, y_t, θ_t) is the robot pose
        - (ε_r, ε_φ) represents measurement noise

    Importance Sampling
    ------------------
    The algorithm uses importance sampling with:
        - Target distribution f: posterior bel(x_t)
        - Proposal distribution g: motion model prior p(x_t | u_t, x_{t-1})
        - Importance weights: w_t^[m] = p(z_t | x_t^[m])

    Convergence rate is O(1/√M) where M is the number of particles.

    Parameters
    ----------
    dataset : str
        Path to MRCLAM dataset directory containing sensor and ground truth data.
    robot : str
        Robot identifier (e.g., 'Robot1', 'Robot2', etc.).
    end_frame : int
        Maximum number of data frames to process for computational efficiency.
    num_particles : int
        Number of particles M in the filter. Typical values: 50-5000.
        Trade-off between accuracy and computational cost.
    motion_noise : array_like, shape (5,)
        Motion model noise parameters [σ_x, σ_y, σ_θ, σ_v, σ_ω] where:
        - σ_x, σ_y, σ_θ: pose noise standard deviations (m, m, rad)
        - σ_v, σ_ω: control noise standard deviations (m/s, rad/s)
    measurement_noise : array_like, shape (2,)
        Measurement model noise parameters [σ_r, σ_φ] where:
        - σ_r: range measurement noise standard deviation (m)
        - σ_φ: bearing measurement noise standard deviation (rad)
    plot : bool, optional
        Whether to generate visualization plots during execution. Default: True.

    Attributes
    ----------
    particles : ndarray, shape (M, 3)
        Current particle set, each row is [x, y, θ] representing a pose hypothesis.
    weights : ndarray, shape (M,)
        Importance weights for each particle, normalized to sum to 1.
    states : ndarray, shape (T, 4)
        Estimated robot trajectory [timestamp, x, y, θ] over time.
    particles_log : ndarray, shape (T*M, 3)
        Historical record of all particles for visualization.
    groundtruth_data : ndarray
        Ground truth robot poses from motion capture system.
    landmark_locations : dict
        Mapping from landmark IDs to positions {id: [x, y]}.

    Examples
    --------
    Basic usage with MRCLAM Dataset1:

    >>> # Motion noise: [x_noise, y_noise, theta_noise, v_noise, omega_noise]
    >>> motion_noise = np.array([0.1, 0.1, 0.1, 0.2, 0.2])
    >>> # Measurement noise: [range_noise, bearing_noise]
    >>> measurement_noise = np.array([0.1, 0.1])
    >>>
    >>> pf = ParticleFilter(
    ...     dataset="data/MRCLAM_Dataset1",
    ...     robot="Robot1",
    ...     end_frame=1000,
    ...     num_particles=100,
    ...     motion_noise=motion_noise,
    ...     measurement_noise=measurement_noise
    ... )

    Parameter selection guidelines:

    >>> # Conservative (smooth, accurate)
    >>> motion_noise = np.array([0.05, 0.05, 0.05, 0.1, 0.1])
    >>> measurement_noise = np.array([0.05, 0.05])
    >>>
    >>> # Aggressive (handles uncertainty, faster convergence)
    >>> motion_noise = np.array([0.2, 0.2, 0.2, 0.3, 0.3])
    >>> measurement_noise = np.array([0.2, 0.2])

    Notes
    -----
    This implementation assumes:
    - Known landmark positions (stored in dataset)
    - Known data associations (landmark correspondences)
    - Synchronized odometry and measurement data
    - Indoor structured environment (MRCLAM dataset)

    For unknown correspondences or outdoor environments, consider:
    - Augmented MCL with random particle injection
    - Adaptive resampling strategies
    - Different motion models (e.g., differential drive)

    References
    ----------
    .. [1] Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic Robotics.
           MIT Press. Chapter 8: Monte Carlo Localization.
    .. [2] Doucet, A., Godsill, S., & Andrieu, C. (2000). On sequential Monte
           Carlo sampling methods for Bayesian filtering. Statistics and
           Computing, 10(3), 197-208.
    .. [3] Dellaert, F., Fox, D., Burgard, W., & Thrun, S. (1999). Monte Carlo
           localization for mobile robots. IEEE International Conference on
           Robotics and Automation (ICRA).
    """

    def __init__(
        self,
        dataset,
        robot,
        end_frame,
        num_particles,
        motion_noise,
        measurement_noise,
        plot=True,
    ):
        self.load_data(dataset, robot, end_frame)
        self.initialization(num_particles, motion_noise, measurement_noise)
        for data in self.data:
            if data[1] == -1:
                self.motion_update(data)
            else:
                self.measurement_update(data)
                self.importance_sampling()
            self.state_update()
            # Plot every n frames
            if len(self.states) > 800 and len(self.states) % (end_frame / 8) == 0:
                if plot:
                    self.plot_data()

    def load_data(self, dataset, robot, end_frame):
        """
        Load and preprocess MRCLAM dataset for particle filter localization.

        Loads robot odometry, sensor measurements, ground truth poses, and landmark
        positions from the MRCLAM dataset. Synchronizes all data streams by timestamp
        and creates lookup tables for efficient landmark access.

        The MRCLAM dataset structure:
        - Barcodes.dat: Barcode to landmark ID mapping
        - {Robot}_Groundtruth.dat: Vicon motion capture poses [time, x, y, θ]
        - Landmark_Groundtruth.dat: True landmark positions [id, x, y]
        - {Robot}_Measurement.dat: Range-bearing observations [time, id, r, φ]
        - {Robot}_Odometry.dat: Control inputs [time, id, v, ω]

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
        Data preprocessing steps:
        1. Load all data files using numpy.loadtxt
        2. Merge and sort odometry and measurements by timestamp
        3. Synchronize with ground truth data (remove data before first GT frame)
        4. Truncate to specified end_frame for computational efficiency
        5. Create landmark lookup tables for O(1) position access
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

    def initialization(self, num_particles, motion_noise, measurement_noise):
        """
        Initialize particle filter with initial particle distribution and parameters.

        Creates initial particle set by sampling around the first ground truth pose.
        This implements the initialization step for Monte Carlo Localization where
        particles are distributed according to initial belief about robot pose.

        Initial Distribution
        -------------------
        Particles are sampled from Gaussian distributions:

            x_0^[m] ~ N(x_gt(0), σ_x²)
            y_0^[m] ~ N(y_gt(0), σ_y²)
            θ_0^[m] ~ N(θ_gt(0), σ_θ²)

        Where (x_gt(0), y_gt(0), θ_gt(0)) is the first ground truth pose.

        Parameters
        ----------
        num_particles : int
            Number of particles M in the filter.
        motion_noise : array_like, shape (5,)
            Motion noise parameters [σ_x, σ_y, σ_θ, σ_v, σ_ω].
        measurement_noise : array_like, shape (2,)
            Measurement noise parameters [σ_r, σ_φ].

        Notes
        -----
        Initial weights are set uniformly: w_0^[m] = 1/M for all particles.
        This represents maximum entropy (uniform) initial belief.

        For global localization (unknown initial pose), particles should be
        distributed uniformly across the entire state space instead.
        """
        # Initial state: use first ground truth data
        self.init = False
        self.states = np.array([self.groundtruth_data[0]])
        self.last_timestamp = self.states[-1][0]

        # Noise covariance
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise

        # Initial particles: set with initial state mean and normalized noise
        # num_particles of [x, y, theta]
        self.particles = np.zeros((num_particles, 3))
        self.particles[:, 0] = np.random.normal(
            self.states[-1][1], self.motion_noise[0], num_particles
        )
        self.particles[:, 1] = np.random.normal(
            self.states[-1][2], self.motion_noise[1], num_particles
        )
        self.particles[:, 2] = np.random.normal(
            self.states[-1][3], self.motion_noise[2], num_particles
        )

        # Initial weights: set with uniform weights for each particle
        self.weights = np.full(num_particles, 1.0 / num_particles)

        # Initialize a variable to log particles
        self.particles_log = self.particles

    def motion_update(self, control):
        """
        Propagate particles through motion model with noise (prediction step).

        Implements the prediction step of particle filter by sampling from the
        motion model p(x_t | u_t, x_{t-1}^[m]) for each particle. Uses the bicycle
        motion model with additive Gaussian noise on control inputs.

        Motion Model
        -----------
        For each particle m, the state transition is:

            x_t^[m] = x_{t-1}^[m] + (v + ε_v) * cos(θ_{t-1}^[m]) * Δt
            y_t^[m] = y_{t-1}^[m] + (v + ε_v) * sin(θ_{t-1}^[m]) * Δt
            θ_t^[m] = θ_{t-1}^[m] + (ω + ε_ω) * Δt

        Where:
            - (v, ω) are control inputs (linear and angular velocity)
            - ε_v ~ N(0, σ_v²), ε_ω ~ N(0, σ_ω²) are control noise
            - Δt is the time step between measurements
            - θ is normalized to [-π, π]

        Parameters
        ----------
        control : array_like, shape (4,)
            Control data [timestamp, subject_id, v, ω] where:
            - timestamp: time of control command
            - subject_id: always -1 for odometry data
            - v: linear velocity (m/s)
            - ω: angular velocity (rad/s)

        Notes
        -----
        - Skips update if Δt < 0.001s to avoid numerical instability
        - Noise is applied to control inputs rather than final poses
        - Each particle evolves independently with different noise realizations
        - Particle diversity is maintained through stochastic motion sampling

        The motion model assumes:
        - Bicycle kinematics (no slip, point robot)
        - Constant velocity between time steps
        - Independent Gaussian noise on control inputs
        """
        # Motion Model (simplified):
        # State: [x, y, θ]
        # Control: [v, w]
        # [x_t, y_t, θ_t] = g(u_t, x_t-1)
        #   x_t  =  x_t-1 + v * cosθ_t-1 * delta_t
        #   y_t  =  y_t-1 + v * sinθ_t-1 * delta_t
        #   θ_t  =  θ_t-1 + w * delta_t

        # Skip motion update if two odometry data are too close
        delta_t = control[0] - self.last_timestamp
        if delta_t < 0.001:
            return

        for particle in self.particles:
            # Apply noise to control input
            v = np.random.normal(control[2], self.motion_noise[3], 1)
            w = np.random.normal(control[3], self.motion_noise[4], 1)

            # Compute updated [x, y, theta]
            particle[0] += v * np.cos(particle[2]) * delta_t
            particle[1] += v * np.sin(particle[2]) * delta_t
            particle[2] += w * delta_t

            # Limit θ within [-pi, pi]
            if particle[2] > np.pi:
                particle[2] -= 2 * np.pi
            elif particle[2] < -np.pi:
                particle[2] += 2 * np.pi

        # Update timestamp
        self.last_timestamp = control[0]

        # Update particles log
        self.particles_log = np.vstack([self.particles_log, self.particles])

    def measurement_update(self, measurement):
        """
        Update particle weights based on sensor measurement likelihood (update step).

        Computes importance weights for each particle based on how well the
        predicted measurement matches the actual observation. This implements
        the measurement update step: w_t^[m] = p(z_t | x_t^[m]).

        Measurement Model
        ----------------
        For range-bearing sensors observing landmark l:

            r_expected = √[(x_l - x_t^[m])² + (y_l - y_t^[m])²]
            φ_expected = atan2(y_l - y_t^[m], x_l - x_t^[m]) - θ_t^[m]

        Likelihood calculation:

            p(z_t | x_t^[m]) = p(r | x_t^[m]) * p(φ | x_t^[m])

        Where:
            p(r | x_t^[m]) ~ N(r_expected, σ_r²)
            p(φ | x_t^[m]) ~ N(φ_expected, σ_φ²)

        Parameters
        ----------
        measurement : array_like, shape (4,)
            Sensor measurement [timestamp, landmark_id, range, bearing] where:
            - timestamp: time of measurement
            - landmark_id: barcode ID of observed landmark
            - range: measured distance to landmark (m)
            - bearing: measured angle to landmark (rad)

        Notes
        -----
        - Unknown landmarks are ignored (no weight update)
        - Weights are normalized to sum to 1 after all particles updated
        - Zero-weight protection: if all weights are zero, reset to uniform
        - Independent range and bearing likelihood calculation
        - Uses known landmark positions from dataset

        Mathematical Foundation
        ----------------------
        The importance weight represents the measurement likelihood:

            w_t^[m] ∝ exp(-½[(r_obs - r_pred)²/σ_r² + (φ_obs - φ_pred)²/σ_φ²])

        This follows from the assumption of independent Gaussian measurement noise.
        """
        # Measurement Model:
        #   range   =  sqrt((x_l - x_t)^2 + (y_l - y_t)^2)
        #  bearing  =  atan2((y_l - y_t) / (x_l - x_t)) - θ_t

        # Continue if landmark is not found in self.landmark_locations
        if measurement[1] not in self.landmark_locations:
            return

        # Importance factor: update weights for each particle (Table 6.4)
        x_l = self.landmark_locations[measurement[1]][0]
        y_l = self.landmark_locations[measurement[1]][1]
        for i in range(len(self.particles)):
            # Compute expected range and bearing given current pose
            x_t = self.particles[i][0]
            y_t = self.particles[i][1]
            theta_t = self.particles[i][2]
            q = (x_l - x_t) * (x_l - x_t) + (y_l - y_t) * (y_l - y_t)
            range_expected = np.sqrt(q)
            bearing_expected = np.arctan2(y_l - y_t, x_l - x_t) - theta_t

            # Compute the probability of range and bearing differences in normal distribution with mean = 0 and sigma = measurement noise
            range_error = measurement[2] - range_expected
            bearing_error = measurement[3] - bearing_expected
            prob_range = stats.norm(0, self.measurement_noise[0]).pdf(range_error)
            prob_bearing = stats.norm(0, self.measurement_noise[1]).pdf(bearing_error)

            # Update weights
            self.weights[i] = prob_range * prob_bearing

        # Normalization
        # Avoid all-zero weights
        if np.sum(self.weights) == 0:
            self.weights = np.ones_like(self.weights)
        self.weights /= np.sum(self.weights)

        # Update timestamp
        self.last_timestamp = measurement[0]

    def importance_sampling(self):
        """
        Resample particles according to importance weights (resampling step).

        Implements the resampling step of particle filter by drawing M particles
        with replacement from the current particle set, where the probability of
        selecting each particle is proportional to its importance weight.

        This step implements "survival of the fittest" - particles with higher
        weights (better match to observations) are more likely to survive and
        reproduce, while particles with low weights are eliminated.

        Resampling Algorithm
        -------------------
        For m = 1 to M:
            1. Draw index i with probability ∝ w_t^[i]
            2. Set x_t^[m] = x_t^[i] (copy selected particle)
            3. Reset weight w_t^[m] = 1/M

        Statistical Properties
        ---------------------
        - Expected number of copies of particle i: M * w_t^[i]
        - Particles can be duplicated multiple times
        - Particles with w_t^[i] ≈ 0 are likely eliminated
        - Total particle count remains constant at M
        - Effective sample size may decrease due to duplication

        Implementation
        -------------
        Uses numpy.random.choice with replacement and probability weights.
        This implements systematic resampling, which has good statistical
        properties and O(M) computational complexity.

        Notes
        -----
        - Weights are implicitly reset to uniform after resampling
        - Particle diversity may decrease (sample impoverishment)
        - Consider adaptive resampling based on effective sample size
        - Alternative methods: stratified, residual, or systematic resampling

        References
        ----------
        .. [1] Arulampalam, M. S., et al. (2002). A tutorial on particle filters
               for online nonlinear/non-Gaussian Bayesian tracking. IEEE
               Transactions on Signal Processing, 50(2), 174-188.
        """
        # Resample according to importance weights
        new_idexes = np.random.choice(
            len(self.particles), len(self.particles), replace=True, p=self.weights
        )

        # Update new particles
        self.particles = self.particles[new_idexes]

    def state_update(self):
        """
        Estimate robot pose from current particle distribution.

        Computes the mean pose estimate from the particle set and adds it to
        the trajectory history. The pose estimate represents the expected value
        of the posterior distribution approximated by particles.

        Estimation Method
        ----------------
        The robot pose estimate is the sample mean:

            x̂_t = (1/M) * Σ_{m=1}^M x_t^[m]
            ŷ_t = (1/M) * Σ_{m=1}^M y_t^[m]
            θ̂_t = (1/M) * Σ_{m=1}^M θ_t^[m]

        Notes
        -----
        - Mean provides minimum mean squared error estimate for Gaussian posteriors
        - For multimodal distributions, mean may not represent any mode well
        - Alternative estimates: maximum a posteriori (MAP), median, mode
        - Orientation averaging assumes θ ∈ [-π, π] (no wraparound issues)
        - Updates states array with [timestamp, x, y, θ] format

        For multimodal posteriors, consider:
        - Clustering particles and reporting multiple hypotheses
        - Weighted mean using particle weights (if not uniform after resampling)
        - Maximum weight particle as MAP estimate
        """
        # Update robot pos by mean of all particles
        state = np.mean(self.particles, axis=0)
        self.states = np.append(
            self.states,
            np.array([[self.last_timestamp, state[0], state[1], state[2]]]),
            axis=0,
        )

    def plot_data(self):
        """
        Visualize current particle filter state and estimation results.

        Creates a comprehensive plot showing:
        - Ground truth robot trajectory (blue line)
        - Estimated robot trajectory (red line)
        - Current particle distribution (orange dots)
        - Historical particle positions (black dots)
        - Landmark positions with ID labels (black stars)
        - Start and end points (green and yellow circles)

        Visualization Elements
        ---------------------
        - **Ground Truth**: Reference trajectory from motion capture system
        - **Estimate**: Mean pose trajectory from particle filter
        - **Particles**: Current pose hypotheses, density indicates confidence
        - **Particle Log**: Historical particles showing filter evolution
        - **Landmarks**: Static features used for localization

        The particle distribution visualization provides insight into:
        - Localization uncertainty (particle spread)
        - Multimodal beliefs (particle clusters)
        - Filter convergence (particle concentration)
        - Estimation quality (agreement with ground truth)

        Notes
        -----
        - Called every n frames during algorithm execution
        - Useful for debugging and algorithm analysis
        - Particle density indicates posterior probability
        - Large spread suggests high uncertainty or poor sensor data
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

        # Particles
        plt.scatter(
            self.particles[:, 0],
            self.particles[:, 1],
            s=40,
            c="orange",
            alpha=0.8,
            label="Particles",
        )
        plt.scatter(
            self.particles_log[:, 0],
            self.particles_log[:, 1],
            s=0.5,
            c="k",
            alpha=0.5,
            label="Particles log",
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

        plt.title("Particle Filter Localization with Known Correspondences")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    def build_dataframes(self):
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


def build_timeseries(data, cols):
    timeseries = pd.DataFrame(data, columns=cols)
    timeseries["stamp"] = pd.to_datetime(timeseries["stamp"], unit="s")
    timeseries = timeseries.set_index("stamp")
    return timeseries


def filter_static_landmarks(lm, barcodes):
    for L, l in dict(barcodes).items():  # Translate barcode num to landmark num
        lm[lm == l] = L
    lm = lm[lm.type > 5]  # Keep only static landmarks
    return lm


if __name__ == "__main__":
    # # Dataset 0
    # dataset = "../0_Dataset0"
    # end_frame = 10000
    # # Number of particles
    # num_particles = 50
    # # Motion model noise (in meters / rad)
    # # [noise_x, noise_y, noise_theta, noise_v, noise_w]
    # motion_noise = np.array([0.2, 0.2, 0.2, 0.1, 0.1])
    # # Measurement model noise (in meters / rad)
    # # [noise_range, noise_bearing]
    # measurement_noise = np.array([0.1, 0.2])

    # Dataset 1
    dataset = "../0_Dataset1"
    end_frame = 3200
    # Number of particles
    num_particles = 50
    # Motion model noise (in meters / rad)
    # [noise_x, noise_y, noise_theta, noise_v, noise_w]
    motion_noise = np.array([0.1, 0.1, 0.1, 0.2, 0.2])
    # Measurement model noise (in meters / rad)
    # [noise_range, noise_bearing]
    measurement_noise = np.array([0.1, 0.1])

    pf = ParticleFilter(
        dataset, end_frame, num_particles, motion_noise, measurement_noise
    )
    plt.show()
