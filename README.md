# Time Series Plot Animation Tool

A Python tool for creating animated GIF visualizations from time series CSV data. Generate x-y plots that animate over time, showing data point movements and optional trajectories.

## Features

- 📊 **Time Series Animation**: Convert CSV data to animated GIF plots
- ⏱️ **Real Timing**: Animation speed matches actual timestamp intervals
- 🚀 **Performance Optimized**: Multiprocessing support for faster generation
- 🎯 **Trajectory Display**: Optional trajectory trails with configurable length
- 🔗 **Connection Lines**: Draw lines between data points based on configuration
- 🏷️ **Custom Labels**: Use meaningful names for data points
- ⚙️ **Flexible Options**: Comprehensive command-line interface

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
The CSV file should have the following format:
```csv
timestamp,data0_x,data0_y,data1_x,data1_y,data2_x,data2_y,...
1756717200000000000,0.0,0.0,1.0,0.0,2.0,0.0
1756717200100000000,-0.1,0.0,1.0,-0.1,1.8,0.0
...
```

- **First column**: Timestamp in nanoseconds
- **Remaining columns**: x,y coordinate pairs for each data point
- Column naming: `dataN_x`, `dataN_y` where N is the data point index

### Configuration Files

#### `data/label.txt` (Optional)
Define custom labels for data points:
```
0, Robot
1, Target
2, Obstacle
```
Format: `index, label_name`

#### `data/connection.txt` (Optional)
Define lines between data points:
```
0, 1
0, 2
1, 3
```
Format: `index1, index2` - draws line from data point index1 to index2

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
- **Timing**: Matches real timestamp intervals (minimum 50ms per frame)
- **Frame Rate**: Typically 10-20 FPS depending on data intervals
- **Features**: 
  - Data points with distinct colors
  - Optional trajectory trails
  - Connection lines between specified points
  - Time display (relative, starting from 0.0s)
  - Legend with custom labels

## Examples

### Sample Data Visualization
The included sample data shows 5 data points (pioneer_1 through pioneer_5) with:
- 0.1-second intervals (10 FPS animation)
- Connection lines forming a network structure
- 10+ seconds of movement data

### Real-World Applications
- Robot trajectory visualization
- Multi-agent system monitoring
- Sensor network data animation
- Time-series scientific data presentation

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