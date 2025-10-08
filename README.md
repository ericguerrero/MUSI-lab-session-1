# MUSI Lab Session 1: Introduction and Basics

Educational environment for learning mobile robot localization algorithms using the MRCLAM dataset.

## Quick Start

### 1. Install UV (Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone Repository

```bash
git clone https://github.com/ericguerrero/MUSI-lab-session-1.git
cd MUSI-lab-session-1
```

### 3. Install Dependencies

```bash
uv sync
```

This command creates a virtual environment and installs all required packages (7 dependencies).

### 4. Launch Session 1 Notebook

```bash
uv run marimo edit notebooks/session1_intro_and_basics.py
```

Your browser will open with the interactive notebook at `http://localhost:2718`.

## Updating the Repository

If you've already cloned the repository and need to get the latest updates:

```bash
# Navigate to your repository
cd MUSI-lab-session-1

# Pull the latest changes
git pull
```

If you have local changes and encounter conflicts:

```bash
# Stash your local changes
git stash

# Pull the latest updates
git pull

# Reapply your local changes
git stash pop
```

## Session 1 Contents

### Part 1: Dataset Exploration
- Load and visualize MRCLAM datasets (1-9 available)
- Understand robot odometry and measurements
- Explore ground truth data and landmark observations
- Compare dataset characteristics across scenarios

### Part 2: Dead Reckoning Baseline
- Implement basic motion model integration
- Analyze error accumulation over time
- Compute Absolute Trajectory Error (ATE)

### Part 3: Extended Kalman Filter (EKF)
- Understand Bayesian localization framework
- Implement EKF prediction and update steps
- Tune noise parameters (R and Q matrices)
- Compare EKF vs Dead Reckoning performance

### Part 4: Cross-Dataset Benchmarking
- Automated experiments across multiple robots and datasets
- Statistical analysis of algorithm performance
- Interactive visualization with Plotly and Seaborn

## Repository Structure

```
MUSI-lab-session-1/
├── musi_labs/                      # Algorithm implementations
│   ├── data/                       # MRCLAM dataset reader
│   ├── localization/               # Dead Reckoning, EKF, Particle Filter
│   ├── slam/                       # EKF-SLAM (Session 1 introduction)
│   ├── utils/                      # Metrics and data utilities
│   └── visualization/              # Marimo plotting helpers
├── notebooks/                      # Interactive Marimo sessions
│   └── session1_intro_and_basics.py
├── data/                           # MRCLAM Datasets 1-9
│   ├── MRCLAM_Dataset1/
│   ├── MRCLAM_Dataset2/
│   └── ...
├── pyproject.toml                  # Project configuration
└── README.md                       # This file
```

## Learning Path

1. **Start with the notebook**: Open `session1_intro_and_basics.py` in Marimo
2. **Run cells interactively**: Execute cells sequentially, observe outputs
3. **Experiment with parameters**: Use sliders to tune EKF noise matrices (R, Q)
4. **Compare algorithms**: Dead Reckoning to EKF to understand improvements
5. **Visualize results**: Interactive Plotly plots with hover details and zoom
6. **Try different datasets**: Use the dataset selector to explore different scenarios

## Troubleshooting

### UV not found after installation
Restart your terminal. If the issue persists, check the UV installation documentation: https://docs.astral.sh/uv/

### Marimo notebook won't open
```bash
# Check if Marimo is installed
uv run marimo --version

# Try explicit browser opening
uv run marimo edit notebooks/session1_intro_and_basics.py --headless
```

### Import errors
```bash
# Ensure dependencies are synced
cd MUSI-lab-session-1
uv sync

# Verify imports work
uv run python -c "from musi_labs.data.reader import Reader; print('OK')"
```

### Dataset not found
```bash
# Verify dataset exists
ls data/MRCLAM_Dataset1/

# If missing, check git clone completed successfully
git pull
```

## Available Datasets

This repository includes **all 9 MRCLAM datasets** (~500MB total):

- **MRCLAM_Dataset1**: Basic trajectory with good landmark coverage
- **MRCLAM_Dataset2**: Loop closure scenario
- **MRCLAM_Dataset3**: Longer trajectory with varied motion
- **MRCLAM_Dataset4**: Multi-robot coordination scenario
- **MRCLAM_Dataset5-9**: Additional scenarios with varying complexity

Each dataset includes data for 5 robots with:
- Odometry measurements (velocity commands)
- Landmark observations (range and bearing)
- Ground truth poses from motion capture
- Landmark positions

## Technical Details

### Dependencies (7 core packages)
- **NumPy** - Numerical computing
- **SciPy** - Scientific algorithms
- **pandas** - Data manipulation and time series
- **Matplotlib** - Static plotting
- **Plotly** - Interactive visualizations
- **Seaborn** - Statistical plots
- **Marimo** - Reactive notebook environment

### MRCLAM Dataset Details
- **Environment**: 15m x 8m indoor space
- **Robots**: 5 iRobot Create platforms with monocular cameras
- **Landmarks**: 15 cylindrical tubes with unique barcodes
- **Frequency**: Odometry at ~67Hz, Ground truth at 100Hz
- **Accuracy**: Ground truth accurate to 1e-3 meters (Vicon system)

### Algorithms Implemented
- **Dead Reckoning**: Pure odometry integration baseline
- **Extended Kalman Filter**: Gaussian belief localization with landmark fusion
- **Metrics**: Absolute Trajectory Error (ATE) with timestamp alignment

## Next Steps

After completing Session 1, continue to **Session 2: Advanced Localization and SLAM** covering:
- Particle Filter implementation
- Multi-algorithm benchmarking
- EKF-SLAM and FastSLAM introduction
- Loop closure detection

## Support

For questions or issues:
1. Check this README's Troubleshooting section
2. Review Marimo documentation: https://docs.marimo.io
3. Contact course instructor

## License

Educational use only. Dataset credit: University of Toronto ASRL.

## References

- MRCLAM Dataset: http://asrl.utias.utoronto.ca/datasets/mrclam/
- Probabilistic Robotics (Thrun, Burgard, Fox)
- Marimo Documentation: https://docs.marimo.io
