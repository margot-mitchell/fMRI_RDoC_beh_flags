"""Calculate behavioral metrics for each task."""

import glob
import os
from typing import Callable, Dict

import polars as pl

from .tasks import (
    ax_cpt_rdoc,
    ax_cpt_rdoc_time_resolved,
    cued_task_switching_rdoc,
    flanker_rdoc,
    go_nogo_rdoc,
    n_back_rdoc,
    operation_span_rdoc,
    op_only_span_rdoc,
    simple_span_rdoc,
    spatial_cueing_rdoc,
    spatial_task_switching_rdoc,
    stop_signal_rdoc,
    stroop_rdoc,
    visual_search_rdoc,
)


def get_function_mapping() -> Dict[str, Callable]:
    """Get mapping of task names to their metric calculation functions.
    
    Returns:
        Dict[str, Callable]: Dictionary mapping task names to functions.
    """
    return {
        'axCPT': ax_cpt_rdoc,
        'axCPT_time_resolved': ax_cpt_rdoc_time_resolved,
        'cuedTS': cued_task_switching_rdoc,
        'flanker': flanker_rdoc,
        'goNogo': go_nogo_rdoc,
        'nBack': n_back_rdoc,
        'OpSpan': operation_span_rdoc,
        'opOnly': op_only_span_rdoc,
        'simpleSpan': simple_span_rdoc,
        'spatialCueing': spatial_cueing_rdoc,
        'spatialTS': spatial_task_switching_rdoc,
        'stopSignal': stop_signal_rdoc,
        'stroop': stroop_rdoc,
        'visualSearch': visual_search_rdoc,
    }


def calculate_subject_metrics(subject_folder: str) -> None:
    """Calculate metrics for all tasks in a subject folder.
    
    Args:
        subject_folder (str): Path to subject folder containing preprocessed data.
    """
    # Get list of task folders that exist
    task_folders = [f for f in os.listdir(subject_folder) 
                   if os.path.isdir(os.path.join(subject_folder, f))]
    
    # Get function mapping for available tasks
    function_mapping = get_function_mapping()
    available_functions = {
        task: func for task, func in function_mapping.items()
        if task in task_folders
    }
    
    if not available_functions:
        print(f"No available tasks found in {subject_folder}")
        return
        
    # Calculate metrics for each available task
    all_metrics = []
    for task, func in available_functions.items():
        task_files = glob.glob(os.path.join(subject_folder, task, "*.parquet"))
        if not task_files:
            print(f"No parquet files found for {task} in {subject_folder}")
            continue
            
        # Process each task file
        for task_file in task_files:
            try:
                df = pl.read_parquet(task_file)
                metrics = func(df)
                
                # Ensure metrics has the correct column names
                if 'metric' not in metrics.columns or 'value' not in metrics.columns:
                    print(f"Warning: {task_file} has incorrect column names: {metrics.columns}")
                    continue
                    
                # Convert value column to float to handle mixed types
                metrics = metrics.with_columns(
                    pl.col('value').cast(pl.Float64),
                    pl.lit(task).alias('task'),
                    pl.lit(os.path.basename(task_file)).alias('file')
                )
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Error processing {task_file}: {e}")
    
    if all_metrics:
        # Combine all metrics
        combined_metrics = pl.concat(all_metrics)
        # Save to CSV
        output_file = os.path.join(subject_folder, 'behavioral_summary.csv')
        combined_metrics.write_csv(output_file)
        print(f"Saved metrics to {output_file}")
    else:
        print(f"No metrics calculated for {subject_folder}")


def main():
    """Main function to calculate metrics for all subjects."""
    # Get list of subject folders
    subject_folders = [f for f in os.listdir('preprocessed_data') 
                      if os.path.isdir(os.path.join('preprocessed_data', f))]
    
    for subject_folder in subject_folders:
        print(f"Calculating metrics for {subject_folder}...")
        calculate_subject_metrics(os.path.join('preprocessed_data', subject_folder))


if __name__ == '__main__':
    main() 