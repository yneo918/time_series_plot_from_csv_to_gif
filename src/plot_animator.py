import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import io
import multiprocessing as mp
from functools import partial
import time
import sys


class TimeSeriesAnimator:
    def __init__(self, csv_path, output_dir="output", use_name_file=False,
                 figure_size=(10, 8), dpi=80):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_name_file = use_name_file
        self.figure_size = figure_size
        self.dpi = dpi
        self.data = None
        self.data_columns = []
        
    def load_selected_names(self):
        """Load selected data names from data/name.txt if use_name_file is True"""
        if not self.use_name_file:
            return None

        name_path = self.csv_path.parent / 'name.txt'
        if not name_path.exists():
            print(f"Warning: name.txt not found at {name_path}, using all available data")
            return None

        try:
            selected_names = set()
            with open(name_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip empty lines and comments
                        selected_names.add(line)

            if selected_names:
                print(f"Loaded {len(selected_names)} selected data names from name.txt: {sorted(selected_names)}")
                return selected_names
            else:
                print("Warning: name.txt is empty, using all available data")
                return None

        except Exception as e:
            print(f"Warning: Could not load names from {name_path}: {e}, using all available data")
            return None

    def detect_timestamp_format(self, timestamp_col):
        """Detect timestamp format and return conversion factor to nanoseconds"""
        first_timestamp = self.data[timestamp_col].iloc[0]

        # Check if it's a string representation of a number
        if isinstance(first_timestamp, str):
            try:
                first_timestamp = float(first_timestamp)
            except ValueError:
                raise ValueError(f"Cannot parse timestamp: {first_timestamp}")

        # Determine format based on magnitude
        # Typical Unix timestamp ranges:
        # Seconds: ~1.7e9 (10 digits)
        # Milliseconds: ~1.7e12 (13 digits)
        # Microseconds: ~1.7e15 (16 digits)
        # Nanoseconds: ~1.7e18 (19 digits)

        if first_timestamp >= 1e16:
            # Nanoseconds (e.g., 1756717200000000000)
            print("Detected nanosecond timestamps")
            return 1, 'nanoseconds'
        elif first_timestamp >= 1e13:
            # Microseconds (e.g., 1756717200000000)
            print("Detected microsecond timestamps")
            return 1000, 'microseconds'
        elif first_timestamp >= 1e10:
            # Milliseconds (e.g., 1756717200000)
            print("Detected millisecond timestamps")
            return 1000000, 'milliseconds'
        else:
            # Seconds (e.g., 1756717200.0995314)
            print("Detected second timestamps")
            return 1000000000, 'seconds'

    def load_data(self):
        """Load CSV data and extract data columns"""
        self.data = pd.read_csv(self.csv_path)

        # Check if CSV is empty
        if self.data.empty:
            raise ValueError(f"CSV file is empty: {self.csv_path}")

        # Detect timestamp format and convert to nanoseconds
        timestamp_col = self.data.columns[0]
        self.timestamp_factor, self.timestamp_unit = self.detect_timestamp_format(timestamp_col)

        # Convert timestamps to nanoseconds using high precision arithmetic
        if self.timestamp_unit == 'seconds':
            # Use Decimal for high precision floating point operations
            from decimal import Decimal, getcontext
            getcontext().prec = 50  # Set high precision

            # Convert to nanoseconds with high precision
            timestamps_decimal = [Decimal(str(ts)) * Decimal('1000000000') for ts in self.data[timestamp_col]]
            self.data[timestamp_col] = [int(ts) for ts in timestamps_decimal]
        else:
            # Integer arithmetic for other formats
            self.data[timestamp_col] = self.data[timestamp_col] * self.timestamp_factor

        # Store the first timestamp for relative time calculation
        self.start_timestamp = self.data[timestamp_col].iloc[0]
        
        # Load selective data names if specified
        selected_names = self.load_selected_names()

        # Extract x,y coordinate pairs from columns ending with _x and _y
        all_cols = self.data.columns.tolist()

        # Find all columns ending with _x
        x_columns = [col for col in all_cols if col.endswith('_x')]

        # Group x,y pairs by matching base names
        self.data_columns = []
        for x_col in x_columns:
            base_name = x_col[:-2]  # Remove '_x' suffix
            y_col = base_name + '_y'

            if y_col in all_cols:
                # Filter by selected names if specified
                if selected_names is None or base_name in selected_names:
                    self.data_columns.append({
                        'name': base_name,
                        'x_col': x_col,
                        'y_col': y_col
                    })

        # Check if no data columns were found
        if not self.data_columns:
            raise ValueError(f"No valid x,y coordinate pairs found in CSV. "
                           f"Columns must end with '_x' and '_y' (e.g., 'data1_x', 'data1_y')")

        # Validate data: remove rows with NaN and Inf values
        original_row_count = len(self.data)

        # Create a mask for valid rows (no NaN or Inf in any data column)
        valid_mask = pd.Series([True] * len(self.data))

        for data_info in self.data_columns:
            x_col = data_info['x_col']
            y_col = data_info['y_col']

            # Mark rows with NaN or Inf as invalid
            valid_mask &= ~(self.data[x_col].isna() | self.data[y_col].isna() |
                           np.isinf(self.data[x_col]) | np.isinf(self.data[y_col]))

        # Filter out invalid rows
        self.data = self.data[valid_mask].reset_index(drop=True)

        skipped_rows = original_row_count - len(self.data)
        if skipped_rows > 0:
            print(f"Warning: Skipped {skipped_rows} rows with NaN or Inf values ({len(self.data)} valid rows remaining)")

        # Check if all rows were invalid
        if len(self.data) == 0:
            raise ValueError(f"All rows contain NaN or Inf values. No valid data to process.")

        if selected_names:
            print(f"Using selected data: {[col['name'] for col in self.data_columns]} (from {len(selected_names)} specified)")
        else:
            print(f"Found coordinate pairs: {[col['name'] for col in self.data_columns]}")

        # Load connection data (supports both numeric indices and name-based)
        self.connections = []
        connection_path = self.csv_path.parent / 'connection.txt'
        if connection_path.exists():
            try:
                with open(connection_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):  # Skip empty lines and comments
                            parts = [x.strip() for x in line.split(',')]
                            if len(parts) == 2:
                                # Try to parse as integers first (backward compatibility)
                                try:
                                    idx1, idx2 = int(parts[0]), int(parts[1])
                                    self.connections.append((idx1, idx2))
                                except ValueError:
                                    # Parse as data names
                                    name1, name2 = parts[0], parts[1]
                                    self.connections.append((name1, name2))
                print(f"Loaded {len(self.connections)} connections from connection.txt")
            except Exception as e:
                print(f"Warning: Could not load connections from {connection_path}: {e}")
                self.connections = []
        else:
            print(f"No connection.txt found at {connection_path}")
            self.connections = []
        
        # Load label data (supports both numeric indices and name-based)
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
                                # Try to parse as integer first (backward compatibility)
                                try:
                                    idx = int(parts[0])
                                    label = parts[1]
                                    self.labels[idx] = label
                                except ValueError:
                                    # Parse as data name
                                    name = parts[0]
                                    label = parts[1]
                                    self.labels[name] = label
                print(f"Loaded {len(self.labels)} labels from label.txt")
            except Exception as e:
                print(f"Warning: Could not load labels from {label_path}: {e}")
                self.labels = {}
        else:
            print(f"No label.txt found at {label_path}")
            self.labels = {}
        
        # Update data column names with labels
        for i, data_info in enumerate(self.data_columns):
            name = data_info['name']
            # Check for name-based label first, then index-based (backward compatibility)
            if name in self.labels:
                data_info['display_name'] = self.labels[name]
            elif i in self.labels:
                data_info['display_name'] = self.labels[i]
            else:
                data_info['display_name'] = name
        
        print(f"Loaded {len(self.data)} timestamps with {len(self.data_columns)} data series")
        
    def format_timestamp(self, timestamp_ns):
        """Convert nanosecond timestamp to relative seconds from start with 0.1s precision (truncated)"""
        # Calculate relative time from the start timestamp
        relative_ns = timestamp_ns - self.start_timestamp
        relative_sec = relative_ns / 1e9
        # Truncate to 0.1s precision
        truncated_sec = int(relative_sec * 10) / 10
        return f"{truncated_sec:.1f}s"

    def print_progress(self, current, total, start_time=None, prefix="Progress"):
        """Print progress bar with percentage and optional time estimation"""
        percent = (current / total) * 100
        bar_length = 40
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        # Time estimation
        time_info = ""
        if start_time and current > 0:
            elapsed = time.time() - start_time
            if current < total:
                estimated_total = elapsed * total / current
                remaining = estimated_total - elapsed
                time_info = f" | ETA: {remaining:.1f}s"

        # Print with carriage return to overwrite previous line
        print(f"\r{prefix}: |{bar}| {current}/{total} ({percent:.1f}%){time_info}", end="", flush=True)

        # Print newline when complete
        if current == total:
            print()
    
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
            data1, data2 = None, None
            
            # Handle both numeric indices and name-based connections
            if isinstance(conn[0], int) and isinstance(conn[1], int):
                # Numeric indices (backward compatibility)
                idx1, idx2 = conn
                if idx1 < len(self.data_columns) and idx2 < len(self.data_columns):
                    data1 = self.data_columns[idx1]
                    data2 = self.data_columns[idx2]
            else:
                # Name-based connections
                name1, name2 = conn
                # Find data columns by name
                for data_info in self.data_columns:
                    if data_info['name'] == name1:
                        data1 = data_info
                    elif data_info['name'] == name2:
                        data2 = data_info
            
            # Draw line if both data points found
            if data1 and data2:
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
    
    def create_frame_data(self, timestamp_idx, ax_limits=None, show_trajectory=False, trajectory_length=50, save_frames=False, frame_dir=None, show_time=True):
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
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
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
                      s=40,  # Further reduced marker size
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
            data1, data2 = None, None
            
            # Handle both numeric indices and name-based connections
            if isinstance(conn[0], int) and isinstance(conn[1], int):
                # Numeric indices (backward compatibility)
                idx1, idx2 = conn
                if idx1 < len(self.data_columns) and idx2 < len(self.data_columns):
                    data1 = self.data_columns[idx1]
                    data2 = self.data_columns[idx2]
            else:
                # Name-based connections
                name1, name2 = conn
                # Find data columns by name
                for data_info in self.data_columns:
                    if data_info['name'] == name1:
                        data1 = data_info
                    elif data_info['name'] == name2:
                        data2 = data_info
            
            # Draw line if both data points found
            if data1 and data2:
                x1 = row[data1['x_col']]
                y1 = row[data1['y_col']]
                x2 = row[data2['x_col']]
                y2 = row[data2['y_col']]
                
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.6, linewidth=1.5, zorder=0)
        
        # Set plot properties with minimal styling
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        if show_time:
            ax.set_title(f'Time: {timestamp_display}', fontsize=9)  # Show time if enabled
        else:
            ax.set_title('Time Series Plot', fontsize=9)  # Generic title without time
        ax.legend(fontsize=6, loc='upper right')  # Smaller font and fixed location
        ax.grid(True, alpha=0.2, linewidth=0.5)  # Thinner grid
        ax.set_xlim(ax_limits['x'])
        ax.set_ylim(ax_limits['y'])
        ax.tick_params(labelsize=6)  # Smaller tick labels

        # Convert to PIL Image with maximum compression
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img = Image.open(buf)
        img_copy = img.copy()
        
        # Save frame to file if requested
        if save_frames and frame_dir:
            frame_filename = f"frame_{timestamp_idx:06d}.png"
            frame_path = frame_dir / frame_filename
            fig.savefig(frame_path, format='png', dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')

        plt.close(fig)
        buf.close()

        # Return None when saving frames to avoid memory issues in multiprocessing
        if save_frames:
            return None
        return img_copy

    def create_gif(self, output_filename="animation.gif", duration=None, ax_limits=None,
                   use_real_timing=True, use_multiprocessing=True, max_workers=None,
                   show_trajectory=False, trajectory_length=50, save_frames=False, show_time=True):
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
        
        # Create frame directory if saving frames
        frame_dir = None
        if save_frames:
            # Use CSV filename (without extension) for frame directory
            csv_base_name = self.csv_path.stem
            frame_dir = self.output_dir / f"frames_{csv_base_name}"
            frame_dir.mkdir(exist_ok=True)
            print(f"Saving individual frames to: {frame_dir}")

        start_time = time.time()

        # For very large datasets, disable multiprocessing to avoid memory issues
        # However, when saving frames to disk, memory is not an issue so we can use multiprocessing
        large_dataset_threshold = 1000 if not save_frames else float('inf')
        if use_multiprocessing and len(self.data) > 10 and len(self.data) <= large_dataset_threshold:
            # Use multiprocessing for datasets (unlimited size when saving frames to disk)
            if max_workers is None:
                max_workers = min(mp.cpu_count(), 4)  # Limit to 4 cores max

            if save_frames:
                print(f"Using {max_workers} processes for parallel frame generation (no memory limit with --save-frames)...")
            else:
                print(f"Using {max_workers} processes for parallel frame generation...")

            # Create partial function with fixed parameters
            create_frame_partial = partial(self._create_frame_worker,
                                         ax_limits=ax_limits,
                                         show_trajectory=show_trajectory,
                                         trajectory_length=trajectory_length,
                                         save_frames=save_frames,
                                         frame_dir=frame_dir,
                                         show_time=show_time)

            # Use imap with chunking for better memory management
            chunksize = max(1, len(self.data) // (max_workers * 4))
            with mp.Pool(max_workers) as pool:
                print("Generating frames in parallel...")
                frames = []

                # Use imap with chunking to reduce memory pressure
                for i, frame in enumerate(pool.imap(create_frame_partial, range(len(self.data)), chunksize=chunksize)):
                    if not save_frames and frame is not None:  # Only keep in memory if not saving to files
                        frames.append(frame)
                    # Update progress every frame for real-time feedback in parallel processing
                    self.print_progress(i + 1, len(self.data), start_time, "Creating frames (parallel)")
        elif len(self.data) > large_dataset_threshold and not save_frames:
            # For very large datasets without frame saving, force sequential processing with informative message
            print(f"Large dataset detected ({len(self.data)} frames). Using sequential processing to avoid memory issues.")
            frames = []
            for i in range(len(self.data)):
                frame = self.create_frame_data(i, ax_limits, show_trajectory, trajectory_length, save_frames, frame_dir, show_time)
                frames.append(frame)
                # Update progress every frame for large datasets
                self.print_progress(i + 1, len(self.data), start_time, "Creating frames")
        else:
            # Sequential processing with progress bar
            frames = []
            for i in range(len(self.data)):
                frame = self.create_frame_data(i, ax_limits, show_trajectory, trajectory_length, save_frames, frame_dir, show_time)
                if not save_frames:  # Only keep in memory if not saving to files
                    frames.append(frame)

                # Update progress bar every frame
                self.print_progress(i + 1, len(self.data), start_time, "Creating frames")

        frame_time = time.time() - start_time
        print(f"Frame generation completed in {frame_time:.1f}s")

        # GIF saving progress
        print("Saving GIF...")
        save_start = time.time()
        
        # Calculate frame durations if using real timing
        if use_real_timing:
            durations = self.calculate_frame_durations()
        else:
            durations = duration if duration is not None else 200
        
        # Save as GIF with optimization
        output_path = self.output_dir / output_filename
        if save_frames and frame_dir:
            # Create GIF from saved frame files to save memory
            self._create_gif_from_files(frame_dir, output_path, durations)
        else:
            # Create GIF from frames in memory
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=0,
                optimize=True  # Enable GIF optimization
            )

        save_time = time.time() - save_start
        total_time = time.time() - start_time
        print(f"GIF saved in {save_time:.1f}s")
        print(f"Total processing time: {total_time:.1f}s")
        print(f"Animation saved to: {output_path}")
        return output_path
    
    def _create_gif_from_files(self, frame_dir, output_path, durations):
        """Create GIF from saved frame files to save memory"""
        frame_files = sorted(frame_dir.glob("frame_*.png"))
        if not frame_files:
            raise ValueError(f"No frame files found in {frame_dir}")

        print(f"Loading {len(frame_files)} frame files for GIF creation...")

        # For very large datasets, warn user about potential memory usage
        if len(frame_files) > 2000:
            print(f"Warning: Large number of frames ({len(frame_files)}). This may require significant memory.")
            print("Processing all frames for GIF creation...")

        # Load frames in smaller chunks to manage memory
        chunk_size = 100
        frames_to_save = []

        for i in range(0, len(frame_files), chunk_size):
            chunk_end = min(i + chunk_size, len(frame_files))
            print(f"Loading frames {i} to {chunk_end-1} ({chunk_end-i} frames)...")

            chunk_frames = []
            for frame_file in frame_files[i:chunk_end]:
                try:
                    with Image.open(frame_file) as img:
                        chunk_frames.append(img.copy())
                except Exception as e:
                    print(f"Error loading {frame_file}: {e}")
                    continue

            frames_to_save.extend(chunk_frames)

            # Progress update
            if chunk_end % 500 == 0 or chunk_end == len(frame_files):
                print(f"Loaded {chunk_end}/{len(frame_files)} frames...")

        print(f"Creating GIF from {len(frames_to_save)} frames...")

        if frames_to_save:
            # Save GIF
            frames_to_save[0].save(
                output_path,
                save_all=True,
                append_images=frames_to_save[1:],
                duration=durations,
                loop=0,
                optimize=True
            )

            # Close frames to free memory
            for frame in frames_to_save:
                if hasattr(frame, 'close'):
                    frame.close()
        else:
            raise ValueError("No frames could be loaded successfully")

    def _create_frame_worker(self, timestamp_idx, ax_limits, show_trajectory=False, trajectory_length=50, save_frames=False, frame_dir=None, show_time=True):
        """Worker function for multiprocessing"""
        return self.create_frame_data(timestamp_idx, ax_limits, show_trajectory, trajectory_length, save_frames, frame_dir, show_time)


def main():
    parser = argparse.ArgumentParser(description='Create animated x-y plots from time series CSV data')
    parser.add_argument('csv_file', help='Path to CSV file')
    parser.add_argument('-o', '--output', help='Output GIF filename (default: CSV_filename.gif)')
    parser.add_argument('-d', '--duration', type=int, help='Fixed frame duration in milliseconds (overrides real timing)')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--no-real-timing', action='store_true', help='Disable real timestamp timing')
    parser.add_argument('--no-multiprocessing', action='store_true', help='Disable multiprocessing for frame generation')
    parser.add_argument('--workers', type=int, help='Number of worker processes (default: auto)')
    parser.add_argument('--show-trajectory', action='store_true', help='Show trajectory of previous points (default: off)')
    parser.add_argument('--trajectory-length', type=int, default=50, help='Number of previous points to show in trajectory (default: 50)')
    parser.add_argument('--use-name-file', action='store_true', help='Use data/name.txt to select specific data columns')
    parser.add_argument('--force-sequential', action='store_true', help='Force sequential processing (useful for very large datasets)')
    parser.add_argument('--figure-width', type=float, default=10, help='Figure width in inches (default: 10)')
    parser.add_argument('--figure-height', type=float, default=8, help='Figure height in inches (default: 8)')
    parser.add_argument('--dpi', type=int, default=80, help='Figure DPI for output resolution (default: 80, lower = less memory)')
    parser.add_argument('--save-frames', action='store_true', help='Save individual frame images to output/frames/ directory')
    parser.add_argument('--no-time-display', action='store_true', help='Hide time display in frame titles')

    args = parser.parse_args()

    # Set default output filename if not provided
    if args.output is None:
        csv_path = Path(args.csv_file)
        args.output = f"{csv_path.stem}.gif"

    try:
        # Create animator with configurable figure size and DPI
        figure_size = (args.figure_width, args.figure_height)
        animator = TimeSeriesAnimator(args.csv_file, args.output_dir, args.use_name_file,
                                    figure_size=figure_size, dpi=args.dpi)

        # Load data and create animation
        animator.load_data()

        # Determine timing mode
        use_real_timing = not args.no_real_timing and args.duration is None
        duration = args.duration if args.duration is not None else 200
        use_multiprocessing = not args.no_multiprocessing and not args.force_sequential

        animator.create_gif(args.output, duration, use_real_timing=use_real_timing,
                           use_multiprocessing=use_multiprocessing, max_workers=args.workers,
                           show_trajectory=args.show_trajectory, trajectory_length=args.trajectory_length,
                           save_frames=args.save_frames, show_time=not args.no_time_display)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"Skipping file: {args.csv_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()