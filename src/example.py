#!/usr/bin/env python3
"""
Example usage of the TimeSeriesAnimator with trajectory options
"""

from plot_animator import TimeSeriesAnimator
import time

def main():
    # Create animator with sample data
    animator = TimeSeriesAnimator("data/sample.csv", "output")
    
    # Load data
    animator.load_data()
    
    print("Creating animation without trajectory (default, faster)...")
    start_time = time.time()
    
    # Create animation without trajectory (default)
    animator.create_gif("sample_animation_no_trajectory.gif", show_trajectory=False)
    
    elapsed_time = time.time() - start_time
    print(f"Animation without trajectory created in {elapsed_time:.2f} seconds")
    
    print("\nCreating animation with trajectory...")
    start_time = time.time()
    
    # Create animation with trajectory
    animator.create_gif("sample_animation_with_trajectory.gif", 
                       show_trajectory=True, trajectory_length=30)
    
    elapsed_time = time.time() - start_time
    print(f"Animation with trajectory created in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()