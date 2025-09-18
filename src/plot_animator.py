import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import io
import multiprocessing as mp
from functools import partial


class TimeSeriesAnimator:
    def __init__(self, csv_path, output_dir="output"):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.data = None
        self.data_columns = []
        
    def load_data(self):
        """Load CSV data and extract data columns"""
        self.data = pd.read_csv(self.csv_path)
        
        # Store the first timestamp for relative time calculation
        timestamp_col = self.data.columns[0]
        self.start_timestamp = self.data[timestamp_col].iloc[0]
        
        # Extract data columns (all columns except timestamp)
        all_cols = self.data.columns[1:].tolist()
        
        # Group x,y pairs
        self.data_columns = []
        for i in range(0, len(all_cols), 2):
            if i + 1 < len(all_cols):
                x_col = all_cols[i]
                y_col = all_cols[i + 1]
                data_name = x_col.replace('_x', '').replace('_y', '')
                self.data_columns.append({
                    'name': data_name,
                    'x_col': x_col,
                    'y_col': y_col,
                    'index': i // 2  # Store the data index for label lookup
                })
        
        # Load connection data
        self.connections = []
        connection_path = self.csv_path.parent / 'connection.txt'
        if connection_path.exists():
            try:
                with open(connection_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):  # Skip empty lines and comments
                            parts = [int(x.strip()) for x in line.split(',')]
                            if len(parts) == 2:
                                self.connections.append(tuple(parts))
                print(f"Loaded {len(self.connections)} connections from connection.txt")
            except Exception as e:
                print(f"Warning: Could not load connections from {connection_path}: {e}")
                self.connections = []
        else:
            print(f"No connection.txt found at {connection_path}")
            self.connections = []
        
        # Load label data
        self.labels = {}
        label_path = self.csv_path.parent / 'label.txt'
        if label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):  # Skip empty lines and comments
                            parts = [x.strip() for x in line.split(',', 1)]  # Split only on first comma
                            if len(parts) == 2:
                                idx = int(parts[0])
                                label = parts[1]
                                self.labels[idx] = label
                print(f"Loaded {len(self.labels)} labels from label.txt")
            except Exception as e:
                print(f"Warning: Could not load labels from {label_path}: {e}")
                self.labels = {}
        else:
            print(f"No label.txt found at {label_path}")
            self.labels = {}
        
        # Update data column names with labels
        for data_info in self.data_columns:
            idx = data_info['index']
            if idx in self.labels:
                data_info['display_name'] = self.labels[idx]
            else:
                data_info['display_name'] = data_info['name']
        
        print(f"Loaded {len(self.data)} timestamps with {len(self.data_columns)} data series")
        
    def format_timestamp(self, timestamp_ns):
        """Convert nanosecond timestamp to relative seconds from start with 0.1s precision (truncated)"""
        # Calculate relative time from the start timestamp
        relative_ns = timestamp_ns - self.start_timestamp
        relative_sec = relative_ns / 1e9
        # Truncate to 0.1s precision
        truncated_sec = int(relative_sec * 10) / 10
        return f"{truncated_sec:.1f}s"
    
    def create_frame(self, timestamp_idx, ax_limits=None):
        """Create a single frame for the given timestamp index"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get current row data
        row = self.data.iloc[timestamp_idx]
        timestamp_ns = row.iloc[0]
        timestamp_display = self.format_timestamp(timestamp_ns)
        
        # Plot each data series
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.data_columns)))
        
        for i, data_info in enumerate(self.data_columns):
            x_val = row[data_info['x_col']]
            y_val = row[data_info['y_col']]
            
            ax.scatter(x_val, y_val, 
                      color=colors[i], 
                      s=100, 
                      label=data_info['display_name'],
                      alpha=0.8)
            
            # Add trajectory (previous points)
            if timestamp_idx > 0:
                prev_data = self.data.iloc[:timestamp_idx + 1]
                prev_x = prev_data[data_info['x_col']]
                prev_y = prev_data[data_info['y_col']]
                ax.plot(prev_x, prev_y, 
                       color=colors[i], 
                       alpha=0.3, 
                       linewidth=1)
        
        # Draw connection lines between data points
        for conn in self.connections:
            idx1, idx2 = conn
            if idx1 < len(self.data_columns) and idx2 < len(self.data_columns):
                data1 = self.data_columns[idx1]
                data2 = self.data_columns[idx2]
                
                x1 = row[data1['x_col']]
                y1 = row[data1['y_col']]
                x2 = row[data2['x_col']]
                y2 = row[data2['y_col']]
                
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.6, linewidth=1.5, zorder=0)
        
        # Set plot properties
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Time Series Plot - Time: {timestamp_display}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set axis limits if provided
        if ax_limits:
            ax.set_xlim(ax_limits['x'])
            ax.set_ylim(ax_limits['y'])
        else:
            # Auto-scale with some padding
            all_x = []
            all_y = []
            for data_info in self.data_columns:
                all_x.extend(self.data[data_info['x_col']])
                all_y.extend(self.data[data_info['y_col']])
            
            x_margin = (max(all_x) - min(all_x)) * 0.1
            y_margin = (max(all_y) - min(all_y)) * 0.1
            ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
            ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
        
        return fig
    
    def calculate_frame_durations(self):
        """Calculate frame durations based on actual timestamp intervals"""
        timestamp_col = self.data.columns[0]
        timestamps = self.data[timestamp_col].values
        
        # Calculate intervals directly in nanoseconds to avoid precision loss
        intervals_ns = np.diff(timestamps)
        
        # Convert to milliseconds with proper rounding
        durations_ms = np.round(intervals_ns / 1e6).astype(int)
        
        # For the last frame, use the last interval or a default duration
        if len(intervals_ns) > 0:
            # Use the last interval for the final frame
            last_duration = int(round(intervals_ns[-1] / 1e6))
        else:
            # Only one frame, use default
            last_duration = 200
        
        # Add the last frame duration
        durations_ms = np.append(durations_ms, last_duration)
        
        # Ensure minimum duration of 50ms for visibility
        durations_ms = np.maximum(durations_ms, 50)
        
        print(f"Frame durations range: {durations_ms.min():.1f}ms - {durations_ms.max():.1f}ms")
        print(f"Total frames: {len(self.data)}, Duration array length: {len(durations_ms)}")
        return durations_ms.tolist()
    
    def create_frame_data(self, timestamp_idx, ax_limits=None, show_trajectory=False, trajectory_length=50):
        """Create frame data without matplotlib display - optimized for speed"""
        # Pre-calculate axis limits once if not provided
        if ax_limits is None:
            all_x = []
            all_y = []
            for data_info in self.data_columns:
                all_x.extend(self.data[data_info['x_col']])
                all_y.extend(self.data[data_info['y_col']])
            
            x_margin = (max(all_x) - min(all_x)) * 0.1
            y_margin = (max(all_y) - min(all_y)) * 0.1
            ax_limits = {
                'x': (min(all_x) - x_margin, max(all_x) + x_margin),
                'y': (min(all_y) - y_margin, max(all_y) + y_margin)
            }
        
        # Use Agg backend for faster rendering without display
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(8, 6), dpi=80)  # Reduced size and DPI
        
        # Get current row data
        row = self.data.iloc[timestamp_idx]
        timestamp_ns = row.iloc[0]
        timestamp_display = self.format_timestamp(timestamp_ns)
        
        # Plot each data series
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.data_columns)))
        
        for i, data_info in enumerate(self.data_columns):
            x_val = row[data_info['x_col']]
            y_val = row[data_info['y_col']]
            
            ax.scatter(x_val, y_val, 
                      color=colors[i], 
                      s=80,  # Reduced marker size
                      label=data_info['display_name'],
                      alpha=0.8)
            
            # Add trajectory (previous points) only if enabled
            if show_trajectory and timestamp_idx > 0:
                # Only show last N points for trajectory to improve performance
                start_idx = max(0, timestamp_idx - trajectory_length)
                prev_data = self.data.iloc[start_idx:timestamp_idx + 1]
                prev_x = prev_data[data_info['x_col']]
                prev_y = prev_data[data_info['y_col']]
                ax.plot(prev_x, prev_y, 
                       color=colors[i], 
                       alpha=0.3, 
                       linewidth=1)
        
        # Draw connection lines between data points
        for conn in self.connections:
            idx1, idx2 = conn
            if idx1 < len(self.data_columns) and idx2 < len(self.data_columns):
                data1 = self.data_columns[idx1]
                data2 = self.data_columns[idx2]
                
                x1 = row[data1['x_col']]
                y1 = row[data1['y_col']]
                x2 = row[data2['x_col']]
                y2 = row[data2['y_col']]
                
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.6, linewidth=1.5, zorder=0)
        
        # Set plot properties
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Time Series Plot - Time: {timestamp_display}')
        ax.legend(fontsize=8)  # Smaller font
        ax.grid(True, alpha=0.3)
        ax.set_xlim(ax_limits['x'])
        ax.set_ylim(ax_limits['y'])
        
        # Convert to PIL Image more efficiently
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img = Image.open(buf)
        img_copy = img.copy()
        
        plt.close(fig)
        buf.close()
        
        return img_copy

    def create_gif(self, output_filename="animation.gif", duration=None, ax_limits=None, 
                   use_real_timing=True, use_multiprocessing=True, max_workers=None,
                   show_trajectory=False, trajectory_length=50):
        """Create animated GIF from all timestamps with performance optimizations"""
        print(f"Creating {len(self.data)} frames...")
        if show_trajectory:
            print(f"Trajectory enabled (showing last {trajectory_length} points)")
        
        # Pre-calculate axis limits once for all frames
        if ax_limits is None:
            all_x = []
            all_y = []
            for data_info in self.data_columns:
                all_x.extend(self.data[data_info['x_col']])
                all_y.extend(self.data[data_info['y_col']])
            
            x_margin = (max(all_x) - min(all_x)) * 0.1
            y_margin = (max(all_y) - min(all_y)) * 0.1
            ax_limits = {
                'x': (min(all_x) - x_margin, max(all_x) + x_margin),
                'y': (min(all_y) - y_margin, max(all_y) + y_margin)
            }
        
        if use_multiprocessing and len(self.data) > 10:
            # Use multiprocessing for large datasets
            if max_workers is None:
                max_workers = min(mp.cpu_count(), 4)  # Limit to 4 cores max
            
            print(f"Using {max_workers} processes for parallel frame generation...")
            
            # Create partial function with fixed parameters
            create_frame_partial = partial(self._create_frame_worker, 
                                         ax_limits=ax_limits, 
                                         show_trajectory=show_trajectory,
                                         trajectory_length=trajectory_length)
            
            with mp.Pool(max_workers) as pool:
                frames = pool.map(create_frame_partial, range(len(self.data)))
        else:
            # Sequential processing for small datasets
            frames = []
            for i in range(len(self.data)):
                frame = self.create_frame_data(i, ax_limits, show_trajectory, trajectory_length)
                frames.append(frame)
                
                if (i + 1) % 10 == 0:
                    print(f"Created frame {i + 1}/{len(self.data)}")
        
        print("Frames created, saving GIF...")
        
        # Calculate frame durations if using real timing
        if use_real_timing:
            durations = self.calculate_frame_durations()
        else:
            durations = duration if duration is not None else 200
        
        # Save as GIF with optimization
        output_path = self.output_dir / output_filename
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            optimize=True  # Enable GIF optimization
        )
        
        print(f"Animation saved to: {output_path}")
        return output_path
    
    def _create_frame_worker(self, timestamp_idx, ax_limits, show_trajectory=False, trajectory_length=50):
        """Worker function for multiprocessing"""
        return self.create_frame_data(timestamp_idx, ax_limits, show_trajectory, trajectory_length)


def main():
    parser = argparse.ArgumentParser(description='Create animated x-y plots from time series CSV data')
    parser.add_argument('csv_file', help='Path to CSV file')
    parser.add_argument('-o', '--output', default='animation.gif', help='Output GIF filename')
    parser.add_argument('-d', '--duration', type=int, help='Fixed frame duration in milliseconds (overrides real timing)')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--no-real-timing', action='store_true', help='Disable real timestamp timing')
    parser.add_argument('--no-multiprocessing', action='store_true', help='Disable multiprocessing for frame generation')
    parser.add_argument('--workers', type=int, help='Number of worker processes (default: auto)')
    parser.add_argument('--show-trajectory', action='store_true', help='Show trajectory of previous points (default: off)')
    parser.add_argument('--trajectory-length', type=int, default=50, help='Number of previous points to show in trajectory (default: 50)')
    
    args = parser.parse_args()
    
    # Create animator
    animator = TimeSeriesAnimator(args.csv_file, args.output_dir)
    
    # Load data and create animation
    animator.load_data()
    
    # Determine timing mode
    use_real_timing = not args.no_real_timing and args.duration is None
    duration = args.duration if args.duration is not None else 200
    use_multiprocessing = not args.no_multiprocessing
    
    animator.create_gif(args.output, duration, use_real_timing=use_real_timing, 
                       use_multiprocessing=use_multiprocessing, max_workers=args.workers,
                       show_trajectory=args.show_trajectory, trajectory_length=args.trajectory_length)


if __name__ == "__main__":
    main()