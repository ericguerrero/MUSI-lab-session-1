# MRCLAM Dataset Collection

## Included Dataset

### MRCLAM_Dataset1
- **Status**: Included in repository (~41MB)
- **Environment**: Indoor 15m × 8m space
- **Robots**: 5 iRobot Create platforms (Robot1-5)
- **Landmarks**: 15 cylindrical tubes with unique barcodes
- **Files**:
  - `Odometry.dat` - Velocity and angular velocity commands
  - `Measurement.dat` - Range/bearing to landmarks and other robots
  - `Groundtruth.dat` - True robot positions (Vicon motion capture)
  - `Landmark_Groundtruth.dat` - True landmark positions
  - `Barcodes.dat` - Barcode-to-landmark ID mapping

## Additional Datasets (Optional)

Download from [MRCLAM website](http://asrl.utias.utoronto.ca/datasets/mrclam/):

### Download Instructions

```bash
cd data/

# Dataset 2
wget http://asrl.utias.utoronto.ca/datasets/mrclam/MRCLAM_Dataset2.zip
unzip MRCLAM_Dataset2.zip
rm MRCLAM_Dataset2.zip

# Dataset 3
wget http://asrl.utias.utoronto.ca/datasets/mrclam/MRCLAM_Dataset3.zip
unzip MRCLAM_Dataset3.zip
rm MRCLAM_Dataset3.zip

# Repeat for Datasets 4-9 as needed
```

### Dataset Characteristics

All datasets share the same environment and robot setup but differ in:
- Robot trajectories
- Observation patterns
- Landmark visibility sequences
- Duration and complexity

### Storage Requirements

- **Dataset1**: ~41MB (included)
- **All 9 datasets**: ~400MB total
- Recommended: Download only datasets needed for specific experiments

## Data Format

### Odometry.dat
```
Time(sec)  Subject#  Forward_V  Angular_V
0.000000   1         0.0        0.0
0.015000   1         0.05       0.01
...
```

### Measurement.dat
```
Time(sec)  Subject#  Range(m)  Bearing(rad)  [ID]
1.234000   5         2.45      0.78          [landmark/robot ID]
...
```

### Groundtruth.dat
```
Time(sec)  X(m)  Y(m)  Theta(rad)
0.000000   0.0   0.0   0.0
...
```

## Using Data in Notebooks

```python
from musi_labs.data import Reader

# Load Dataset1 (default)
reader = Reader("data/MRCLAM_Dataset1", "Robot1", 5000)

# Access data
groundtruth = reader.groundtruth_data  # [Time, x, y, theta]
measurements = reader.measurement_data  # [Time, Subject#, range, bearing]
odometry = reader.odometry_data         # [Time, Subject#, v, omega]
landmarks = reader.landmark_locations   # {barcode_id: [x, y]}
```

## Citation

If using this data in research, please cite:

```
K. Y. K. Leung, T. D. Barfoot, H. H. T. Liu,
"The UTIAS Multi-Robot Cooperative Localization and Mapping Dataset,"
International Journal of Robotics Research,
vol. 30, no. 8, pp. 969-974, July 2011.
```

## Dataset Source

University of Toronto - Autonomous Space Robotics Lab (ASRL)
Website: http://asrl.utias.utoronto.ca/datasets/mrclam/
