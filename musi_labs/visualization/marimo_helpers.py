"""
Marimo UI widget helpers for robotics parameter controls.

Provides standardized widget creation functions for common MUSI project parameters
like datasets, robots, noise matrices, and algorithm settings. All widgets are
designed to work with Marimo's reactive execution model.

Example:
    import marimo as mo
    from musi_labs.visualization.marimo_helpers import (
        create_dataset_selector,
        create_r_matrix_sliders
    )

    # Create reactive controls
    dataset = create_dataset_selector()
    r_sliders = create_r_matrix_sliders()

    # Use in dependent cell
    R = np.diagflat([s.value for s in r_sliders]) ** 2
"""

import marimo as mo


def create_parameter_slider(
    name: str,
    min_val: float,
    max_val: float,
    default: float,
    step: float | None = None,
    description: str = "",
) -> mo.ui.slider:
    """
    Create a standardized parameter slider with consistent styling.

    Args:
        name: Slider label (e.g., "R_x (m)")
        min_val: Minimum slider value
        max_val: Maximum slider value
        default: Default/initial value
        step: Step size (default: (max-min)/100)
        description: Optional tooltip description

    Returns:
        Marimo slider widget with show_value=True

    Example:
        r_x = create_parameter_slider("R_x (m)", 0.01, 5.0, 1.0)
        # In dependent cell:
        measurement_noise_x = r_x.value
    """
    if step is None:
        step = (max_val - min_val) / 100

    return mo.ui.slider(
        min_val,
        max_val,
        value=default,
        step=step,
        label=name,
        show_value=True,
    )


def create_dataset_selector(
    datasets: list[str] | None = None, default: str | None = None
) -> mo.ui.dropdown:
    """
    Create dropdown for MRCLAM dataset selection.

    Args:
        datasets: List of dataset names (default: MRCLAM_Dataset1-9)
        default: Default selection (default: first dataset)

    Returns:
        Marimo dropdown widget

    Example:
        dataset = create_dataset_selector()
        # In dependent cell:
        path = f"data/{dataset.value}"
    """
    if datasets is None:
        datasets = [f"MRCLAM_Dataset{i}" for i in range(1, 10)]

    if default is None:
        default = datasets[0]

    return mo.ui.dropdown(datasets, label="Dataset", value=default)


def create_robot_selector(default: str = "Robot1") -> mo.ui.dropdown:
    """
    Create dropdown for robot selection (Robot1-Robot5).

    Args:
        default: Default robot selection

    Returns:
        Marimo dropdown widget

    Example:
        robot = create_robot_selector()
        reader = Reader(dataset_path, robot.value, end_frame)
    """
    robots = [f"Robot{i}" for i in range(1, 6)]
    return mo.ui.dropdown(robots, label="Robot", value=default)


def create_end_frame_slider(
    max_frames: int = 50000, default: int = 5000, step: int = 1000
) -> mo.ui.slider:
    """
    Create slider for end frame selection.

    Args:
        max_frames: Maximum number of frames
        default: Default frame count
        step: Step size for slider

    Returns:
        Marimo slider widget

    Example:
        end_frame = create_end_frame_slider()
        reader = Reader(dataset, robot, end_frame.value)
    """
    return mo.ui.slider(
        1000,
        max_frames,
        value=default,
        step=step,
        label="End Frame",
        show_value=True,
    )


def create_r_matrix_sliders(
    r_x_default: float = 1.0,
    r_y_default: float = 1.0,
    r_theta_default: float = 10.0,
    min_val: float = 0.01,
    max_val: float = 20.0,
) -> dict[str, mo.ui.slider]:
    """
    Create sliders for EKF measurement noise matrix (R).

    Args:
        r_x_default: Default x position noise (m)
        r_y_default: Default y position noise (m)
        r_theta_default: Default theta noise (rad)
        min_val: Minimum slider value
        max_val: Maximum slider value

    Returns:
        Dictionary with keys 'r_x', 'r_y', 'r_theta'

    Example:
        r_sliders = create_r_matrix_sliders()
        R = np.diagflat([
            r_sliders['r_x'].value,
            r_sliders['r_y'].value,
            r_sliders['r_theta'].value
        ]) ** 2
    """
    return {
        "r_x": create_parameter_slider("R_x (m)", min_val, max_val, r_x_default, 0.01),
        "r_y": create_parameter_slider("R_y (m)", min_val, max_val, r_y_default, 0.01),
        "r_theta": create_parameter_slider(
            "R_θ (rad)", min_val, max_val, r_theta_default, 0.1
        ),
    }


def create_q_matrix_sliders(
    q_x_default: float = 300.0,
    q_y_default: float = 300.0,
    min_val: float = 0.1,
    max_val: float = 1000.0,
) -> dict[str, mo.ui.slider]:
    """
    Create sliders for EKF process noise matrix (Q).

    Args:
        q_x_default: Default x process noise (m²)
        q_y_default: Default y process noise (m²)
        min_val: Minimum slider value
        max_val: Maximum slider value

    Returns:
        Dictionary with keys 'q_x', 'q_y'

    Example:
        q_sliders = create_q_matrix_sliders()
        Q = np.diagflat([
            q_sliders['q_x'].value,
            q_sliders['q_y'].value,
            1e16  # theta noise fixed
        ]) ** 2
    """
    return {
        "q_x": create_parameter_slider("Q_x (m²)", min_val, max_val, q_x_default, 1.0),
        "q_y": create_parameter_slider("Q_y (m²)", min_val, max_val, q_y_default, 1.0),
    }


def create_particle_filter_controls(
    num_particles_default: int = 20,
    max_particles: int = 500,
) -> dict[str, mo.ui.slider]:
    """
    Create sliders for Particle Filter parameters.

    Args:
        num_particles_default: Default number of particles
        max_particles: Maximum particles allowed

    Returns:
        Dictionary with particle count and noise parameter sliders

    Example:
        pf_controls = create_particle_filter_controls()
        pf = ParticleFilter(
            dataset, robot, end_frame,
            num_particles=pf_controls['num_particles'].value,
            motion_noise=np.array([
                pf_controls['noise_x'].value,
                pf_controls['noise_y'].value,
                # ... etc
            ])
        )
    """
    return {
        "num_particles": mo.ui.slider(
            10,
            max_particles,
            value=num_particles_default,
            step=10,
            label="Number of Particles",
            show_value=True,
        ),
        "noise_x": create_parameter_slider("Motion Noise X", 0.01, 1.0, 0.1, 0.01),
        "noise_y": create_parameter_slider("Motion Noise Y", 0.01, 1.0, 0.1, 0.01),
        "noise_theta": create_parameter_slider("Motion Noise θ", 0.01, 1.0, 0.1, 0.01),
        "noise_v": create_parameter_slider("Motion Noise v", 0.01, 1.0, 0.2, 0.01),
        "noise_omega": create_parameter_slider("Motion Noise ω", 0.01, 1.0, 0.2, 0.01),
        "meas_range": create_parameter_slider(
            "Measurement Range Noise", 0.01, 1.0, 0.1, 0.01
        ),
        "meas_bearing": create_parameter_slider(
            "Measurement Bearing Noise", 0.01, 1.0, 0.1, 0.01
        ),
    }


def create_algorithm_selector(
    algorithms: list[str] | None = None, default: str | None = None
) -> mo.ui.dropdown:
    """
    Create dropdown for algorithm selection.

    Args:
        algorithms: List of algorithm names
        default: Default algorithm

    Returns:
        Marimo dropdown widget

    Example:
        algo = create_algorithm_selector(["EKF", "Particle Filter", "Dead Reckoning"])
        # In dependent cell:
        if algo.value == "EKF":
            run_ekf()
    """
    if algorithms is None:
        algorithms = ["Dead Reckoning", "EKF", "Particle Filter"]

    if default is None:
        default = algorithms[0]

    return mo.ui.dropdown(algorithms, label="Algorithm", value=default)


def create_time_scrubber(max_timesteps: int, default: int = 0) -> mo.ui.slider:
    """
    Create a time scrubber slider for trajectory playback.

    Args:
        max_timesteps: Maximum number of timesteps
        default: Starting timestep (default: 0)

    Returns:
        Marimo slider for time scrubbing

    Example:
        time_slider = create_time_scrubber(len(groundtruth))
        # In dependent cell:
        trajectory_up_to_now = groundtruth[:time_slider.value+1]
        plot_trajectory(trajectory_up_to_now)
    """
    return mo.ui.slider(
        0,
        max_timesteps - 1,
        value=default,
        step=1,
        label="Trajectory Progress",
        show_value=True,
    )


def build_control_panel(widgets: dict):
    """
    Build a standardized vertical control panel from widgets.

    Args:
        widgets: Dictionary of {label: widget}

    Returns:
        Marimo vstack containing labeled widgets

    Example:
        controls = build_control_panel({
            "Dataset": dataset_selector,
            "Robot": robot_selector,
            "End Frame": end_frame_slider
        })
    """
    elements = []
    for label, widget in widgets.items():
        if label.startswith("##"):  # Section header
            elements.append(mo.md(label))
        else:
            elements.append(widget)

    return mo.vstack(elements)
