import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    # Load Roboto fonts and apply styling
    mo.Html(
        """
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Roboto+Mono&display=swap" rel="stylesheet">

        <style>
        body, .markdown, .cm-editor, .cm-content {
          font-family: 'Roboto', sans-serif !important;
        }

        .cm-scroller, code, pre {
          font-family: 'Roboto Mono', monospace !important;
        }
        </style>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Session 1: Introduction to SLAM Datasets and Basic Localization

    **Learning Objectives**:
    - Understand the MRCLAM dataset structure and characteristics
    - Implement dead reckoning as a baseline localization method
    - Apply Extended Kalman Filter (EKF) for improved localization
    - Compare algorithm performance across multiple datasets

    **Session Structure**:
    1. **Dataset Exploration**: Load and visualize robot trajectories and landmarks
    2. **Dead Reckoning**: Estimate position using odometry integration
    3. **EKF Localization**: Fuse odometry with landmark observations
    4. **Benchmarking**: Compare DR vs EKF across datasets

    **Interactive Controls**: Throughout this notebook, you can adjust dataset, robot,
    and trajectory length using the controls below. Visualizations and algorithms will
    update automatically.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Import Dependencies""")
    return


@app.cell(hide_code=True)
def _():
    # Standard library
    import os
    import sys

    # Data manipulation and visualization
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import seaborn as sns
    from plotly.subplots import make_subplots
    return go, make_subplots, np, os, pd, plt, sns, sys


@app.cell
def _(os, pd, sys):
    # TODO: Just using UV could help with this?
    # Setup project environment: navigate to project root and configure pandas
    if os.path.basename(os.getcwd()) == "notebooks":
        os.chdir("..")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    pd.set_option("mode.chained_assignment", None)

    # Import project modules
    from musi_labs.data.reader import Reader
    from musi_labs.localization.dead_reckoning import DeadReckoning
    from musi_labs.localization.EKF import ExtendedKalmanFilter
    from musi_labs.utils.metrics import compute_ate, compute_dataset_metrics
    from musi_labs.visualization import marimo_helpers as mh
    return (
        DeadReckoning,
        ExtendedKalmanFilter,
        Reader,
        compute_ate,
        compute_dataset_metrics,
        mh,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Part 1: Dataset Exploration

    The **MRCLAM dataset** contains multi-robot cooperative localization and mapping data:
    - **Environment**: 15m × 8m indoor space with cylindrical landmark tubes
    - **Robots**: 5 iRobot Create platforms with monocular cameras
    - **Landmarks**: 15 landmarks with unique barcodes
    - **Scenarios**: 9 different datasets with varying complexity

    **Data Files**:
    - `Odometry.dat`: Velocity commands (v, ω) at ~67Hz
    - `Measurement.dat`: Range/bearing observations to landmarks and other robots
    - `Groundtruth.dat`: True positions from Vicon motion capture at 100Hz
    - `Landmark_Groundtruth.dat`: True landmark positions

    **Interactive Controls**: Select dataset, robot, and trajectory length below.
    """
    )
    return


@app.cell
def _(mh, mo):
    # Interactive controls: Select which dataset and robot to analyze
    # These controls affect all visualizations and algorithms in this notebook
    dataset_selector = mh.create_dataset_selector(default="MRCLAM_Dataset1")
    robot_selector = mh.create_robot_selector(default="Robot5")
    end_frame_slider = mh.create_end_frame_slider(default=5000, max_frames=50000)

    mo.hstack([dataset_selector, robot_selector, end_frame_slider], justify="start")
    return dataset_selector, end_frame_slider, robot_selector


@app.cell
def _(Reader, dataset_selector, end_frame_slider, robot_selector):
    # Load selected dataset (updates automatically when controls change)
    dataset_path = f"data/{dataset_selector.value}"
    reader = Reader(
        dataset_path, robot_selector.value, end_frame_slider.value, plot=False
    )
    return dataset_path, reader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Visualize Trajectory and Landmarks""")
    return


@app.cell(hide_code=True)
def _(dataset_selector, go, mo, reader, robot_selector):
    # TODO: Remove black edge from landmark markers
    # TODO: Extract some of this plotting logis out to an utils file, this might not be relevant for the students.
    # Interactive map visualization: Hover over points for details, zoom/pan to explore
    fig_map = go.Figure()

    # Robot trajectory with time-gradient coloring
    n_points = len(reader.groundtruth_data)
    fig_map.add_trace(
        go.Scatter(
            x=reader.groundtruth_data[:, 1],
            y=reader.groundtruth_data[:, 2],
            mode="lines+markers",
            name="Robot Trajectory",
            marker=dict(
                size=3,
                color=list(range(n_points)),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Time Step", x=1.15),
            ),
            line=dict(color="#1f77b4", width=2),
            hovertemplate=(
                "<b>Trajectory</b><br>"
                "Time: %{text:.1f}s<br>"
                "X: %{x:.2f}m<br>"
                "Y: %{y:.2f}m<extra></extra>"
            ),
            text=[f"{t:.1f}" for t in reader.groundtruth_data[:, 0]],
        )
    )

    # Start point
    fig_map.add_trace(
        go.Scatter(
            x=[reader.groundtruth_data[0, 1]],
            y=[reader.groundtruth_data[0, 2]],
            mode="markers",
            name="Start",
            marker=dict(size=15, color="green", symbol="diamond"),
            hovertemplate="<b>Start Point</b><br>X: %{x:.2f}m<br>Y: %{y:.2f}m<extra></extra>",
        )
    )

    # End point
    fig_map.add_trace(
        go.Scatter(
            x=[reader.groundtruth_data[-1, 1]],
            y=[reader.groundtruth_data[-1, 2]],
            mode="markers",
            name="End",
            marker=dict(size=15, color="red", symbol="square"),
            hovertemplate="<b>End Point</b><br>X: %{x:.2f}m<br>Y: %{y:.2f}m<extra></extra>",
        )
    )

    # Landmarks
    landmark_xs = [
        reader.landmark_locations[loc][0] for loc in reader.landmark_locations
    ]
    landmark_ys = [
        reader.landmark_locations[loc][1] for loc in reader.landmark_locations
    ]
    landmark_ids = [
        int(reader.landmark_indexes[loc] + 5) for loc in reader.landmark_locations
    ]

    fig_map.add_trace(
        go.Scatter(
            x=landmark_xs,
            y=landmark_ys,
            mode="markers+text",
            name="Landmarks",
            marker=dict(
                size=12,
                color="gold",
                symbol="x",
                line=dict(color="black", width=2),
            ),
            text=[f"LM{id_}" for id_ in landmark_ids],
            textposition="top center",
            textfont=dict(size=9, color="black", family="monospace"),
            hovertemplate="<b>Landmark %{text}</b><br>X: %{x:.2f}m<br>Y: %{y:.2f}m<extra></extra>",
        )
    )

    # Layout
    fig_map.update_layout(
        title=f"Dataset: {dataset_selector.value}, Robot: {robot_selector.value}",
        xaxis_title="X Position (m)",
        yaxis_title="Y Position (m)",
        hovermode="closest",
        height=600,
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        plot_bgcolor="white",
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="LightGray",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="Gray",
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="LightGray",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="Gray",
            scaleanchor="x",
            scaleratio=1,
        ),
    )

    trajectory_plot = mo.ui.plotly(fig_map)
    trajectory_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Dataset Metrics Comparison

    **Objective**: Compute and compare metrics across multiple datasets to understand
    their complexity and suitability for different experiments.

    **Metrics**:
    - **Path Length**: Total distance traveled (m)
    - **Duration**: Total time of trajectory (s)
    - **Number of Landmarks**: Available for observations
    - **Distance**: Direct start-to-end distance (m)
    - **Measurement Density**: Observations per meter traveled

    Run the cell below to compute metrics for all datasets and robots.
    """
    )
    return


@app.cell
def _(Reader, compute_dataset_metrics, pd):
    # Define datasets and robots to analyze
    datasets_list = [
        "MRCLAM_Dataset1",
        "MRCLAM_Dataset2",
        "MRCLAM_Dataset3",
        "MRCLAM_Dataset4",
    ]
    robots_list = ["Robot1", "Robot2", "Robot3", "Robot4", "Robot5"]
    #TODO: Parameterize the 5000

    # Compute metrics for all combinations
    metrics_list = []
    for _ds in datasets_list:
        for _robot in robots_list:
            try:
                # Load dataset
                _temp_reader = Reader(f"data/{_ds}", _robot, 5000, plot=False)

                # Compute metrics
                _metrics = compute_dataset_metrics(_temp_reader)
                _metrics["dataset"] = _ds
                _metrics["robot"] = _robot
                metrics_list.append(_metrics)
            except Exception as _e:
                # Skip if dataset/robot combination doesn't exist
                print(f"Skipping {_ds}/{_robot}: {_e}")
                continue

    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    # Reorder columns for better readability
    metrics_df = metrics_df[
        [
            "dataset",
            "robot",
            "path_length",
            "duration",
            "n_landmarks",
            "distance",
            "m_density",
        ]
    ]

    metrics_df
    return datasets_list, metrics_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Visualize Dataset Metrics""")
    return


@app.cell
def _(go, make_subplots, metrics_df, mo):
    # Create interactive Plotly subplots for metrics comparison using boxplots
    fig_metrics = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Path Length Distribution by Dataset",
            "Trajectory Duration Distribution by Dataset",
            "Measurement Density Distribution by Dataset",
            "Direct Distance Distribution by Dataset",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    datasets_unique = metrics_df["dataset"].unique()

    # Add Path Length boxplots
    for dataset in datasets_unique:
        dataset_data = metrics_df[metrics_df["dataset"] == dataset]
        fig_metrics.add_trace(
            go.Box(
                y=dataset_data["path_length"],
                name=dataset,
                boxmean='sd',
                marker_color="#1f77b4",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Add Duration boxplots
    for dataset in datasets_unique:
        dataset_data = metrics_df[metrics_df["dataset"] == dataset]
        fig_metrics.add_trace(
            go.Box(
                y=dataset_data["duration"],
                name=dataset,
                boxmean='sd',
                marker_color="#ff7f0e",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Add Measurement Density boxplots
    for dataset in datasets_unique:
        dataset_data = metrics_df[metrics_df["dataset"] == dataset]
        fig_metrics.add_trace(
            go.Box(
                y=dataset_data["m_density"],
                name=dataset,
                boxmean='sd',
                marker_color="#2ca02c",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Add Distance boxplots
    for dataset in datasets_unique:
        dataset_data = metrics_df[metrics_df["dataset"] == dataset]
        fig_metrics.add_trace(
            go.Box(
                y=dataset_data["distance"],
                name=dataset,
                boxmean='sd',
                marker_color="#d62728",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    # Update axes labels
    fig_metrics.update_yaxes(title_text="Path Length (m)", row=1, col=1)
    fig_metrics.update_yaxes(title_text="Duration (s)", row=1, col=2)
    fig_metrics.update_yaxes(title_text="Measurements/m", row=2, col=1)
    fig_metrics.update_yaxes(title_text="Distance (m)", row=2, col=2)

    # Update layout
    fig_metrics.update_layout(
        height=700,
        showlegend=False,
        plot_bgcolor="white",
    )

    mo.ui.plotly(fig_metrics)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Analysis of Dataset Metrics

    **Observations**:
    - Different datasets have varying path lengths and complexities
    - Measurement density affects localization algorithm performance
    - Higher density = more landmark observations = better accuracy potential

    ### TASK 1:  Dataset Selection for Further Experiments:
    Based on these metrics, select datasets with different characteristics:
    - **Simple**: High measurement density, short path
    - **Moderate**: Medium density, moderate path length
    - **Complex**: Low density, long path or loop closures
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Part 2: Dead Reckoning

    **Dead Reckoning** estimates position by integrating velocity commands over time:

    $$
    \begin{align}
    x_{t+1} &= x_t + v \cdot \cos(\theta_t) \cdot \Delta t \\
    y_{t+1} &= y_t + v \cdot \sin(\theta_t) \cdot \Delta t \\
    \theta_{t+1} &= \theta_t + \omega \cdot \Delta t
    \end{align}
    $$

    **Limitations**:
    - Errors accumulate over time (drift)
    - No correction from observations
    - Useful as a baseline for comparison

    **Objective**: Run dead reckoning and analyze error accumulation.
    """
    )
    return


@app.cell
def _(DeadReckoning, dataset_path, end_frame_slider, robot_selector):
    # Reactive Dead Reckoning execution
    dr = DeadReckoning(
        dataset_path, robot_selector.value, end_frame_slider.value, plot=False
    )
    dr.run()
    dr.build_dataframes()
    return (dr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Velocity Commands Analysis""")
    return


@app.cell
def _(dr, mo, np):
    # Add ground truth velocities for comparison
    dr.motion_gt = dr.motion.join(dr.gt).dropna()
    dr.motion_gt["dx"] = dr.motion_gt["x"].diff()
    dr.motion_gt["dy"] = dr.motion_gt["y"].diff()
    dr.motion_gt["dtheta"] = dr.motion_gt["theta"].diff()
    dr.motion_gt["dt"] = dr.motion_gt.index.to_series().diff().dt.total_seconds()
    dr.motion_gt["v_gt"] = (
        np.linalg.norm([dr.motion_gt["dx"], dr.motion_gt["dy"]], axis=0)
        / dr.motion_gt["dt"]
    )
    dr.motion_gt["omega_gt"] = dr.motion_gt["dtheta"] / dr.motion_gt["dt"]

    # Compute velocity errors
    dr.motion_gt["v_error"] = (dr.motion_gt["v_gt"] - dr.motion_gt["v"]) ** 2
    dr.motion_gt["omega_error"] = (
        dr.motion_gt["omega_gt"] - dr.motion_gt["omega"]
    ) ** 2

    dr.motion_gt = dr.motion_gt.dropna()
    mo.md(f"**Velocity data points**: {len(dr.motion_gt)}")
    return


@app.cell
def _(dr, go, make_subplots, mo):
    # Interactive velocity comparison plots
    fig_vel = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Linear Velocity: Commands vs Ground Truth",
            "Angular Velocity: Commands vs Ground Truth",
        ),
        vertical_spacing=0.12,
    )

    # Linear velocity traces
    fig_vel.add_trace(
        go.Scatter(
            x=dr.motion_gt.index,
            y=dr.motion_gt["v"],
            mode="lines",
            name="Command v",
            line=dict(color="#1f77b4", width=1.5),
            hovertemplate="<b>Command v</b><br>Time: %{x}<br>v: %{y:.3f} m/s<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig_vel.add_trace(
        go.Scatter(
            x=dr.motion_gt.index,
            y=dr.motion_gt["v_gt"],
            mode="lines",
            name="Ground Truth v",
            line=dict(color="#ff7f0e", width=1.5, dash="dash"),
            hovertemplate="<b>GT v</b><br>Time: %{x}<br>v: %{y:.3f} m/s<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Angular velocity traces
    fig_vel.add_trace(
        go.Scatter(
            x=dr.motion_gt.index,
            y=dr.motion_gt["omega"],
            mode="lines",
            name="Command ω",
            line=dict(color="#2ca02c", width=1.5),
            showlegend=True,
            hovertemplate="<b>Command ω</b><br>Time: %{x}<br>ω: %{y:.3f} rad/s<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig_vel.add_trace(
        go.Scatter(
            x=dr.motion_gt.index,
            y=dr.motion_gt["omega_gt"],
            mode="lines",
            name="Ground Truth ω",
            line=dict(color="#d62728", width=1.5, dash="dash"),
            showlegend=True,
            hovertemplate="<b>GT ω</b><br>Time: %{x}<br>ω: %{y:.3f} rad/s<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Update axes
    fig_vel.update_yaxes(title_text="v (m/s)", row=1, col=1)
    fig_vel.update_yaxes(title_text="ω (rad/s)", row=2, col=1)
    fig_vel.update_xaxes(title_text="Time", row=2, col=1)

    # Update layout
    fig_vel.update_layout(
        height=600,
        showlegend=True,
        hovermode="x unified",
        plot_bgcolor="white",
        legend=dict(x=1.02, y=1, xanchor="left"),
    )

    mo.ui.plotly(fig_vel)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### TASK 2: Trajectory Comparison and ATE Computation

    **Absolute Trajectory Error (ATE)** measures the Root Mean Square Error between
    estimated and ground truth positions:

    $$
    \text{ATE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \|(x_i, y_i) - (x_i^{gt}, y_i^{gt})\|^2}
    $$

    Lower ATE indicates better localization accuracy.
    """
    )
    return


@app.cell
def _(compute_ate, dr):
    # Compute ATE for Dead Reckoning (using DataFrames from build_dataframes())
    ate_dr = compute_ate(dr.states_df, dr.gt, verbose=True)
    ate_dr
    return (ate_dr,)


@app.cell(hide_code=True)
def _(dr, go, mo):
    # Interactive trajectory comparison
    fig_dr_traj = go.Figure()

    # Dead Reckoning estimate
    fig_dr_traj.add_trace(
        go.Scatter(
            x=dr.states_df["x"],
            y=dr.states_df["y"],
            mode="lines",
            name="Dead Reckoning",
            line=dict(color="#d62728", width=2),
            hovertemplate="<b>Dead Reckoning</b><br>X: %{x:.3f}m<br>Y: %{y:.3f}m<extra></extra>",
        )
    )

    # Ground Truth
    fig_dr_traj.add_trace(
        go.Scatter(
            x=dr.gt["x"],
            y=dr.gt["y"],
            mode="lines",
            name="Ground Truth",
            line=dict(color="#1f77b4", width=2, dash="dot"),
            hovertemplate="<b>Ground Truth</b><br>X: %{x:.3f}m<br>Y: %{y:.3f}m<extra></extra>",
        )
    )

    # Start point
    fig_dr_traj.add_trace(
        go.Scatter(
            x=[dr.gt["x"].iloc[0]],
            y=[dr.gt["y"].iloc[0]],
            mode="markers",
            name="Start",
            marker=dict(size=12, color="green", symbol="diamond"),
            hovertemplate="<b>Start</b><extra></extra>",
        )
    )

    # End points
    fig_dr_traj.add_trace(
        go.Scatter(
            x=[dr.gt["x"].iloc[-1]],
            y=[dr.gt["y"].iloc[-1]],
            mode="markers",
            name="GT End",
            marker=dict(size=12, color="blue", symbol="square"),
            hovertemplate="<b>GT End</b><extra></extra>",
        )
    )

    fig_dr_traj.add_trace(
        go.Scatter(
            x=[dr.states_df["x"].iloc[-1]],
            y=[dr.states_df["y"].iloc[-1]],
            mode="markers",
            name="DR End",
            marker=dict(size=12, color="red", symbol="square"),
            hovertemplate="<b>DR End</b><extra></extra>",
        )
    )

    # Layout
    fig_dr_traj.update_layout(
        title="Dead Reckoning vs Ground Truth",
        xaxis_title="X Position (m)",
        yaxis_title="Y Position (m)",
        hovermode="closest",
        height=600,
        showlegend=True,
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="LightGray"),
        yaxis=dict(showgrid=True, gridcolor="LightGray", scaleanchor="x", scaleratio=1),
    )

    mo.ui.plotly(fig_dr_traj)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Observation**: Dead reckoning drift increases over time due to error accumulation.
    The estimated trajectory diverges from ground truth, especially in long trajectories.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Part 3: Extended Kalman Filter Localization

    **Extended Kalman Filter (EKF)** improves upon dead reckoning by fusing odometry
    with landmark observations using Bayesian filtering.

    **Algorithm Steps**:
    1. **Prediction**: Use motion model to predict next state (like dead reckoning)
    2. **Update**: Correct prediction using landmark measurements

    **Key Parameters**:
    - **R**: Process noise covariance (odometry uncertainty)
    - **Q**: Measurement noise covariance (sensor uncertainty)

    **Motion Model** (same as dead reckoning):

    $$
    \begin{bmatrix}
    x' \\\\
    y' \\\\
    \theta'
    \end{bmatrix} =
    \begin{bmatrix}
    x + v \cos\theta \Delta t \\\\
    y + v \sin\theta \Delta t \\\\
    \theta + \omega \Delta t
    \end{bmatrix}
    $$

    **Measurement Model** (range and bearing to landmarks):

    $$
    \begin{bmatrix}
    r \\\\
    \phi
    \end{bmatrix} =
    \begin{bmatrix}
    \sqrt{(l_x - x)^2 + (l_y - y)^2} \\\\
    \text{atan2}(l_y - y, l_x - x) - \theta
    \end{bmatrix}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### TASK 3: Run EKF and Compare with Dead Reckoning""")
    return


@app.cell
def _(
    ExtendedKalmanFilter,
    dataset_path,
    end_frame_slider,
    np,
    robot_selector,
):
    # EKF noise matrices
    R = np.diagflat(np.array([0.10, 0.10, 1.0])) ** 2  # Process noise
    Q = np.diagflat(np.array([500, 500, 1e16])) ** 2  # Measurement noise

    # Reactive EKF execution
    ekf = ExtendedKalmanFilter(
        dataset_path, robot_selector.value, end_frame_slider.value, R, Q, plot=False
    )
    ekf.build_dataframes()
    return Q, R, ekf


@app.cell(hide_code=True)
def _(compute_ate, ekf):
    # Compute ATE for EKF (using DataFrames from build_dataframes())
    ate_ekf = compute_ate(ekf.states_df, ekf.gt, verbose=True)
    ate_ekf
    return (ate_ekf,)


@app.cell(hide_code=True)
def _(dr, ekf, go, mo):
    # Three-way trajectory comparison
    fig_comparison = go.Figure()

    # Ground Truth
    fig_comparison.add_trace(
        go.Scatter(
            x=ekf.groundtruth_data[:, 1],
            y=ekf.groundtruth_data[:, 2],
            mode="lines",
            name="Ground Truth",
            line=dict(color="#2ca02c", width=3, dash="dot"),
            hovertemplate="<b>Ground Truth</b><br>X: %{x:.3f}m<br>Y: %{y:.3f}m<extra></extra>",
        )
    )

    # Dead Reckoning
    fig_comparison.add_trace(
        go.Scatter(
            x=dr.states_df["x"],
            y=dr.states_df["y"],
            mode="lines",
            name="Dead Reckoning",
            line=dict(color="#d62728", width=2),
            hovertemplate="<b>Dead Reckoning</b><br>X: %{x:.3f}m<br>Y: %{y:.3f}m<extra></extra>",
        )
    )

    # EKF
    fig_comparison.add_trace(
        go.Scatter(
            x=ekf.states_df["x"],
            y=ekf.states_df["y"],
            mode="lines",
            name="EKF",
            line=dict(color="#ff7f0e", width=2),
            hovertemplate="<b>EKF</b><br>X: %{x:.3f}m<br>Y: %{y:.3f}m<extra></extra>",
        )
    )

    # Start and end markers
    fig_comparison.add_trace(
        go.Scatter(
            x=[ekf.groundtruth_data[0, 1]],
            y=[ekf.groundtruth_data[0, 2]],
            mode="markers",
            name="Start",
            marker=dict(size=12, color="green", symbol="diamond"),
            hovertemplate="<b>Start</b><extra></extra>",
        )
    )

    fig_comparison.add_trace(
        go.Scatter(
            x=[ekf.groundtruth_data[-1, 1]],
            y=[ekf.groundtruth_data[-1, 2]],
            mode="markers",
            name="End",
            marker=dict(size=12, color="red", symbol="square"),
            hovertemplate="<b>End</b><extra></extra>",
        )
    )

    # Layout
    fig_comparison.update_layout(
        title="Trajectory Comparison: Ground Truth vs Dead Reckoning vs EKF",
        xaxis_title="X Position (m)",
        yaxis_title="Y Position (m)",
        hovermode="closest",
        height=600,
        showlegend=True,
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="LightGray"),
        yaxis=dict(showgrid=True, gridcolor="LightGray", scaleanchor="x", scaleratio=1),
    )

    mo.ui.plotly(fig_comparison)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Error Comparison: DR vs EKF""")
    return


@app.cell(hide_code=True)
def _(ate_dr, ate_ekf, go, mo):
    # Bar chart comparing errors
    fig_ate = go.Figure()

    fig_ate.add_trace(
        go.Bar(
            x=["Dead Reckoning", "EKF"],
            y=[ate_dr, ate_ekf],
            marker=dict(color=["#d62728", "#ff7f0e"]),
            text=[f"{ate_dr:.3f}m", f"{ate_ekf:.3f}m"],
            textposition="outside",
        )
    )

    improvement = (ate_dr - ate_ekf) / ate_dr * 100

    fig_ate.update_layout(
        title=f"Absolute Trajectory Error Comparison (EKF improves by {improvement:.1f}%)",
        yaxis_title="ATE (m)",
        height=400,
        showlegend=False,
        plot_bgcolor="white",
        yaxis=dict(showgrid=True, gridcolor="LightGray"),
    )

    mo.ui.plotly(fig_ate)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Key Insight**: EKF significantly reduces trajectory error by incorporating
    landmark observations. The Kalman filter balances between trusting the motion
    model (odometry) and sensor measurements based on their noise characteristics.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Part 4: Cross-Dataset Benchmarking

    **Objective**: Compare Dead Reckoning and EKF performance across multiple datasets
    to understand algorithm robustness and generalization.

    **Experimental Design**:
    - Run both algorithms on multiple dataset/robot combinations
    - Compute ATE for each experiment
    - Analyze performance patterns

    **Note**: This may take 1-2 minutes to complete.
    """
    )
    return


@app.cell
def _(
    DeadReckoning,
    ExtendedKalmanFilter,
    Q,
    R,
    compute_ate,
    datasets_list,
    pd,
):
    # Automated experiment runner
    results_list = []

    # Limit to first 3 datasets and 3 robots for demo (adjust as needed)
    benchmark_datasets = datasets_list[:3]
    benchmark_robots = ["Robot2","Robot3", "Robot5"]

    for _ds_bench in benchmark_datasets:
        for _robot_bench in benchmark_robots:
            try:
                _dataset_path_temp = f"data/{_ds_bench}"

                # Run Dead Reckoning
                _dr_temp = DeadReckoning(
                    _dataset_path_temp, _robot_bench, 5000, plot=False
                )
                _dr_temp.run()
                _dr_temp.build_dataframes()
                _ate_dr_temp = compute_ate(
                    _dr_temp.states_df, _dr_temp.gt, verbose=False
                )

                # Run EKF
                _ekf_temp = ExtendedKalmanFilter(
                    _dataset_path_temp, _robot_bench, 5000, R, Q, plot=False
                )
                _ekf_temp.build_dataframes()
                _ate_ekf_temp = compute_ate(
                    _ekf_temp.states_df, _ekf_temp.gt, verbose=False
                )

                # Store results
                results_list.append(
                    {
                        "dataset": _ds_bench,
                        "robot": _robot_bench,
                        "ate_dr": _ate_dr_temp,
                        "ate_ekf": _ate_ekf_temp,
                        "improvement_pct": (_ate_dr_temp - _ate_ekf_temp)
                        / _ate_dr_temp
                        * 100,
                    }
                )
                print(f"✓ Completed {_ds_bench}/{_robot_bench}")
            except Exception as _e_bench:
                print(f"✗ Skipped {_ds_bench}/{_robot_bench}: {_e_bench}")
                continue

    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    results_df
    return (results_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Visualize Benchmarking Results""")
    return


@app.cell
def _(mo, plt, results_df, sns):
    # Create comparison plots
    fig_bench, axes_bench = plt.subplots(1, 2, figsize=(14, 5))

    # Dead Reckoning ATE
    sns.barplot(data=results_df, x="dataset", y="ate_dr", hue="robot", ax=axes_bench[0])
    axes_bench[0].set_title("Dead Reckoning - ATE by Dataset and Robot")
    axes_bench[0].set_ylabel("ATE (m)")
    axes_bench[0].legend(title="Robot", bbox_to_anchor=(1.05, 1), loc="upper left")

    # EKF ATE
    sns.barplot(
        data=results_df, x="dataset", y="ate_ekf", hue="robot", ax=axes_bench[1]
    )
    axes_bench[1].set_title("EKF - ATE by Dataset and Robot")
    axes_bench[1].set_ylabel("ATE (m)")
    axes_bench[1].legend(title="Robot", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    mo.mpl.interactive(fig_bench)
    return


@app.cell
def _(mo, plt, results_df, sns):
    # Improvement visualization
    fig_improvement, ax_improvement = plt.subplots(1, 1, figsize=(10, 5))

    sns.barplot(
        data=results_df,
        x="dataset",
        y="improvement_pct",
        hue="robot",
        ax=ax_improvement,
    )
    ax_improvement.set_title("EKF Improvement over Dead Reckoning")
    ax_improvement.set_ylabel("Improvement (%)")
    ax_improvement.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax_improvement.legend(title="Robot", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    mo.mpl.interactive(fig_improvement)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Summary Statistics""")
    return


@app.cell
def _(results_df):
    # Compute summary statistics
    summary_stats = results_df[["ate_dr", "ate_ekf", "improvement_pct"]].describe()
    summary_stats
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Conclusions and Analysis

    **Key Findings**:
    1. **Consistent Improvement**: EKF consistently outperforms Dead Reckoning across datasets
    2. **Dataset Dependency**: Performance varies by dataset complexity (path length, measurement density)
    3. **Robustness**: EKF is more robust to different scenarios

    **Why does EKF perform better?**
    - **Sensor Fusion**: Combines odometry with landmark observations
    - **Uncertainty Modeling**: Explicitly models and propagates uncertainty
    - **Correction Mechanism**: Landmark updates correct accumulated drift

    **When would Dead Reckoning be preferred?**
    - Very short trajectories where drift is minimal
    - Environments with no observable landmarks
    - Computationally constrained systems (DR is much simpler)

    **Next Steps**:
    - Experiment with different R and Q matrices (noise parameters)
    - Try longer trajectories to see error accumulation
    - Explore other datasets (5-9) with different characteristics
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
