# Time Series Plot Animation Tool

A Python tool for creating animated GIF visualizations from time series CSV data. Generate x-y plots that animate over time, showing data point movements and optional trajectories with advanced memory optimization and frame saving capabilities.

## Features

- 📊 **Time Series Animation**: Convert CSV data to animated GIF plots
- ⏱️ **Real Timing**: Animation speed matches actual timestamp intervals (precise 100ms = 10 FPS)
- 🚀 **Performance Optimized**: Unlimited parallel processing with memory-efficient frame saving
- 💾 **Frame Saving**: Save individual PNG frames for large datasets and memory efficiency
- 🎯 **Trajectory Display**: Optional trajectory trails with configurable length
- 🔗 **Connection Lines**: Draw lines between data points based on configuration
- 🏷️ **Custom Labels**: Use meaningful names for data points
- 📐 **Axis Units**: Configure axis labels via optional unit definitions (e.g., X (m), Y (m))
- 🔤 **Font Customization**: Adjustable font sizes with `--font-scale` option
- 📋 **Flexible CSV Format**: Automatic detection of XXX_x, XXX_y coordinate pairs
- ⚠️ **Robust Error Handling**: Automatic skipping of invalid data (NaN/Inf values)
- 🔧 **Name-Based Configuration**: Use data names instead of indices in config files
- ⚙️ **Comprehensive Options**: Full command-line interface with performance tuning
- 🧠 **Memory Optimization**: Configurable figure size, DPI, and batch processing for large datasets
- 📈 **Progress Tracking**: Real-time progress bars for both sequential and parallel processing
- 🕒 **Unix Timestamp Support**: Automatic detection of nanoseconds, microseconds, milliseconds, and seconds
- 🎛️ **Data Selection**: Choose specific data series using name.txt configuration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yneo918/time_series_plot_from_csv_to_gif.git
cd time_series_plot_from_csv_to_gif
```

2. Install dependencies:
```bash
pip install pandas matplotlib numpy pillow
```

**Requirements:**
- Python 3.8+
- NumPy 2.x (compatible with latest versions)
- matplotlib 3.x
- pandas 2.x
- Pillow 10.x+

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
- **First column**: Timestamp (auto-detects nanoseconds, microseconds, milliseconds, or seconds)
- **Coordinate pairs**: Any columns ending with `_x` and `_y` with matching base names
- **Mixed data**: Other columns are ignored - only `XXX_x`, `XXX_y` pairs are processed
- **Flexible naming**: Use meaningful names like `robot_x`, `robot_y` or `sensor1_x`, `sensor1_y`

**Timestamp Format Support:**
- **Nanoseconds**: `1756717200000000000` (19 digits)
- **Microseconds**: `1756717200000000` (16 digits)
- **Milliseconds**: `1756717200000` (13 digits)
- **Seconds**: `1756717200.123` (decimal precision supported)

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
**Legend Order:** The order of labels in this file determines the legend display order

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

#### `data/name.txt` (Optional)
Select specific data series to process:
```
data0
robot
sensor1
```
**Format:** One data name per line (base names without `_x`, `_y` suffixes)
**Purpose:** Process only selected data series instead of all available coordinate pairs

#### `data/unit.txt` (Optional)
Define axis unit labels displayed in the plot:
```
x, m
y, m
```
**Format:** `axis, unit` where `axis` is `x` or `y` (case-insensitive)
**Comments:** Lines beginning with `#` or blank lines are ignored
**Effect:** Axis labels appear as `X (m)` or `Y (m)` when units are provided

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

# Customize font size and hide time display
python src/plot_animator.py data/sample.csv --font-scale 1.5 --no-time-display

# Performance and memory options
python src/plot_animator.py data/sample.csv --workers 4
python src/plot_animator.py data/sample.csv --no-multiprocessing

# Memory optimization for large datasets
python src/plot_animator.py data/sample.csv --save-frames --dpi 60 --figure-width 8 --figure-height 6

# Data selection and processing
python src/plot_animator.py data/sample.csv --use-name-file
python src/plot_animator.py data/sample.csv --force-sequential
```

### Frame Saving Feature
For large datasets (1000+ frames), use the `--save-frames` option:

```bash
# Save individual PNG frames and create GIF
python src/plot_animator.py large_data.csv --save-frames

# Optimize for memory usage
python src/plot_animator.py large_data.csv --save-frames --dpi 40 --figure-width 6 --figure-height 4
```

**Benefits:**
- **Unlimited parallel processing**: No memory constraints for large datasets
- **Individual frames saved**: Each frame saved as `output/frames_CSVNAME/frame_XXXXXX.png`
- **Memory efficient**: Frames not kept in memory during processing
- **Progress tracking**: Real-time progress for both frame generation and GIF creation

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `csv_file` | Path to input CSV file | Required |
| `-o, --output` | Output GIF filename | `{CSV_filename}.gif` |
| `-d, --duration` | Fixed frame duration in ms | Auto (real timing) |
| `--output-dir` | Output directory | `output` |
| `--show-trajectory` | Show trajectory trails | Off |
| `--trajectory-length` | Number of trail points | 50 |
| `--no-real-timing` | Disable real timestamp timing | Off |
| `--no-multiprocessing` | Disable parallel processing | Off |
| `--workers` | Number of worker processes | Auto |
| `--use-name-file` | Use data/name.txt for data selection | Off |
| `--force-sequential` | Force sequential processing | Off |
| `--save-frames` | Save individual PNG frames | Off |
| `--figure-width` | Figure width in inches | 10 |
| `--figure-height` | Figure height in inches | 8 |
| `--dpi` | Figure DPI (lower = less memory) | 80 |
| `--font-scale` | Font size scale factor | 1.0 |
| `--no-time-display` | Hide time stamps in frame titles | Off |

## Performance

The tool includes several optimizations for faster generation and memory efficiency:

- **Unlimited Multiprocessing**: Parallel frame generation with `--save-frames` option (no memory limits)
- **Memory-Efficient Processing**: Configurable figure size and DPI for memory optimization
- **Batch Processing**: Intelligent chunking for large datasets to prevent "too many open files" errors
- **Optimized Rendering**: Efficient matplotlib backend with minimal styling
- **Trajectory Limiting**: Configurable trail length to reduce computation
- **Smart Timing**: Real timestamp intervals with high-precision arithmetic
- **Progress Tracking**: Real-time progress bars with ETA calculation

### Performance Comparison
- **Small datasets** (<10 frames): Sequential processing
- **Medium datasets** (10-1000 frames): Up to 4x speedup with standard multiprocessing
- **Large datasets** (1000+ frames): Unlimited parallel processing with `--save-frames`
- **Very large datasets** (5000+ frames): Memory-optimized processing with frame saving
- **Trajectory disabled**: Significantly faster generation

### Memory Usage
- **Default**: ~800MB for 1000 frames (10×8 inches, 80 DPI)
- **Optimized**: ~200MB for 1000 frames (6×4 inches, 40 DPI, `--save-frames`)
- **Large datasets**: Constant memory usage with `--save-frames` option

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
│   ├── label.txt          # Data point labels
│   └── name.txt           # Data selection configuration
├── src/
│   ├── plot_animator.py   # Main animation tool
│   └── example.py         # Usage examples
├── output/                # Generated animations (gitignored)
│   ├── *.gif              # Generated GIF animations
│   └── frames_*/          # Individual frame directories (with --save-frames)
└── README.md             # This file
```

## Dependencies

- **pandas**: CSV data processing and timestamp handling
- **matplotlib**: Plot generation and rendering
- **numpy**: Numerical operations (NumPy 2.x compatible)
- **Pillow**: Image processing and GIF creation
- **multiprocessing**: Parallel frame generation (built-in)
- **pathlib**: File system operations (built-in)

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

**Slow generation**: Enable frame saving for large datasets
```bash
python src/plot_animator.py data/sample.csv --save-frames --trajectory-length 20
```

**Memory issues (large datasets)**: Use frame saving with memory optimization
```bash
python src/plot_animator.py data/sample.csv --save-frames --dpi 40 --figure-width 6 --figure-height 4
```

**System hangs with very large datasets**: Force sequential processing with frame saving
```bash
python src/plot_animator.py data/sample.csv --save-frames --force-sequential
```

**"Too many open files" error**: This is automatically handled with batch processing in newer versions

**BrokenPipeError in multiprocessing**: Use `--save-frames` option to eliminate inter-process communication
```bash
python src/plot_animator.py data/sample.csv --save-frames
```

**Timing precision issues**: Automatic high-precision arithmetic is used for seconds format timestamps

**CSV format issues**: Ensure coordinate columns end with `_x` and `_y`
```bash
# Correct: data0_x, data0_y, robot_x, robot_y
# Incorrect: data0_X, data0_Y, robot_pos_x, robot_pos_y_coord
```

**Invalid data (NaN/Inf values)**: Rows with invalid data are automatically skipped
```bash
# The tool will display: "Warning: Skipped N rows with NaN or Inf values (M valid rows remaining)"
# Processing continues with valid data only
```

**Empty CSV or no valid columns**: Clear error messages guide you to fix the data format
```bash
# "Error: CSV file is empty"
# "Error: No valid x,y coordinate pairs found in CSV"
```

**Configuration not working**: Use data names from CSV base names
```bash
# If CSV has robot_x, robot_y columns, use "robot" in config files
# label.txt: robot, My Robot
# connection.txt: robot, data0
# name.txt: robot
```

**NumPy compatibility**: Tool is compatible with NumPy 2.x (automatic detection)

### Performance Tips

**For datasets with 1000+ frames**:
```bash
python src/plot_animator.py large_data.csv --save-frames --dpi 60
```

**For datasets with 5000+ frames**:
```bash
python src/plot_animator.py huge_data.csv --save-frames --dpi 40 --figure-width 6 --figure-height 4 --force-sequential
```

**To reduce GIF file size**:
```bash
python src/plot_animator.py data.csv --dpi 50 --figure-width 8 --figure-height 6
```
