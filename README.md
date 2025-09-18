# Time Series Plot Animation Tool

A Python tool for creating animated GIF visualizations from time series CSV data. Generate x-y plots that animate over time, showing data point movements and optional trajectories.

## Features

- 📊 **Time Series Animation**: Convert CSV data to animated GIF plots
- ⏱️ **Real Timing**: Animation speed matches actual timestamp intervals (precise 100ms = 10 FPS)
- 🚀 **Performance Optimized**: Multiprocessing support for faster generation
- 🎯 **Trajectory Display**: Optional trajectory trails with configurable length
- 🔗 **Connection Lines**: Draw lines between data points based on configuration
- 🏷️ **Custom Labels**: Use meaningful names for data points
- 📋 **Flexible CSV Format**: Automatic detection of XXX_x, XXX_y coordinate pairs
- 🔧 **Name-Based Configuration**: Use data names instead of indices in config files
- ⚙️ **Comprehensive Options**: Full command-line interface with performance tuning

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yneo918/time_series_plot_from_csv_to_gif.git
cd time_series_plot_from_csv_to_gif
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Generate an animation from sample data:
```bash
python src/plot_animator.py data/sample.csv
```

Run the example script:
```bash
python src/example.py
```

## Data Format

### CSV Data Structure
The CSV file should contain coordinate pairs with flexible column naming:
```csv
timestamp,data0_x,data0_y,data1_x,data1_y,robot_x,robot_y,sensor1_x,sensor1_y,...
1756717200000000000,0.0,0.0,1.0,0.0,2.0,0.0,3.0,1.0
1756717200100000000,-0.1,0.0,1.0,-0.1,1.8,0.0,3.1,1.1
...
```

**Requirements:**
- **First column**: Timestamp in nanoseconds
- **Coordinate pairs**: Any columns ending with `_x` and `_y` with matching base names
- **Mixed data**: Other columns are ignored - only `XXX_x`, `XXX_y` pairs are processed
- **Flexible naming**: Use meaningful names like `robot_x`, `robot_y` or `sensor1_x`, `sensor1_y`

**Examples of valid coordinate pairs:**
- `data0_x`, `data0_y` → Base name: `data0`
- `robot_x`, `robot_y` → Base name: `robot`  
- `sensor1_x`, `sensor1_y` → Base name: `sensor1`

### Configuration Files

#### `data/label.txt` (Optional)
Define custom display labels for data points:
```
data0, Robot
robot, Main Robot
sensor1, Temperature Sensor
```
**Format:** `data_name, display_label`

**Backward compatibility:** Numeric indices still supported:
```
0, Robot
1, Target
```

#### `data/connection.txt` (Optional)
Define lines between data points:
```
data0, data1
robot, sensor1
data0, robot
```
**Format:** `data_name1, data_name2` - draws line between the two data points

**Backward compatibility:** Numeric indices still supported:
```
0, 1
0, 2
```

## Usage

### Basic Usage
```bash
python src/plot_animator.py data/sample.csv
```

### Advanced Options
```bash
# Custom output filename and directory
python src/plot_animator.py data/sample.csv -o my_animation.gif --output-dir results

# Fixed frame duration (overrides real timing)
python src/plot_animator.py data/sample.csv -d 100

# Enable trajectory display
python src/plot_animator.py data/sample.csv --show-trajectory --trajectory-length 30

# Performance options
python src/plot_animator.py data/sample.csv --workers 4
python src/plot_animator.py data/sample.csv --no-multiprocessing
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `csv_file` | Path to input CSV file | Required |
| `-o, --output` | Output GIF filename | `animation.gif` |
| `-d, --duration` | Fixed frame duration in ms | Auto (real timing) |
| `--output-dir` | Output directory | `output` |
| `--show-trajectory` | Show trajectory trails | Off |
| `--trajectory-length` | Number of trail points | 50 |
| `--no-real-timing` | Disable real timestamp timing | Off |
| `--no-multiprocessing` | Disable parallel processing | Off |
| `--workers` | Number of worker processes | Auto |

## Performance

The tool includes several optimizations for faster generation:

- **Multiprocessing**: Parallel frame generation (default for >10 frames)
- **Optimized Rendering**: Reduced resolution and efficient matplotlib backend
- **Trajectory Limiting**: Configurable trail length to reduce computation
- **Smart Timing**: Real timestamp intervals with minimum duration limits

### Performance Comparison
- Small datasets (<10 frames): Sequential processing
- Large datasets (>10 frames): Up to 4x speedup with multiprocessing
- Trajectory disabled: Significantly faster generation

## Output

- **Format**: Animated GIF with optimized compression
- **Timing**: Precise real timestamp intervals (100ms = 10 FPS for sample data)
- **Frame Rate**: Varies with data intervals (typically 10-20 FPS)
- **Features**: 
  - Data points with distinct colors and meaningful labels
  - Optional trajectory trails with configurable length
  - Connection lines between specified data points
  - Relative time display starting from 0.0s (0.1s precision)
  - Automatic legend with custom names from label.txt

## Examples

### Sample Data Visualization
The included sample data demonstrates:
- **5 data points**: `data0` through `data4` with custom labels (`pioneer_1` to `pioneer_5`)
- **Precise timing**: 0.1-second intervals creating smooth 10 FPS animation
- **Network structure**: Connection lines forming relationships between points
- **10+ seconds**: Extended movement data showing complex trajectories

### Real-World Applications
- **Robotics**: Multi-robot trajectory visualization and coordination
- **IoT/Sensors**: Sensor network data animation with spatial relationships  
- **Scientific**: Time-series data presentation with connection patterns
- **Monitoring**: Multi-agent system tracking with custom labeling
- **Research**: Any coordinate-based time series with flexible data formats

## File Structure

```
time_series_plot_from_csv_to_gif/
├── data/
│   ├── sample.csv          # Sample time series data
│   ├── connection.txt      # Connection definitions
│   └── label.txt          # Data point labels
├── src/
│   ├── plot_animator.py   # Main animation tool
│   └── example.py         # Usage examples
├── output/                # Generated animations (gitignored)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Dependencies

- **pandas**: CSV data processing
- **matplotlib**: Plot generation
- **numpy**: Numerical operations
- **Pillow**: Image processing and GIF creation

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Troubleshooting

### Common Issues

**Slow generation**: Enable multiprocessing and reduce trajectory length
```bash
python src/plot_animator.py data/sample.csv --trajectory-length 20
```

**Memory issues**: Disable trajectory or use sequential processing
```bash
python src/plot_animator.py data/sample.csv --no-multiprocessing
```

**Timing issues**: Use fixed duration instead of real timing
```bash
python src/plot_animator.py data/sample.csv -d 100 --no-real-timing
```

**CSV format issues**: Ensure coordinate columns end with `_x` and `_y`
```bash
# Correct: data0_x, data0_y, robot_x, robot_y
# Incorrect: data0_X, data0_Y, robot_pos_x, robot_pos_y_coord
```

**Configuration not working**: Use data names from CSV base names
```bash
# If CSV has robot_x, robot_y columns, use "robot" in config files
# label.txt: robot, My Robot
# connection.txt: robot, data0
```