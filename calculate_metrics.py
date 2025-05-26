"""Calculates behavioral metrics from preprocessed parquet files.

This script reads preprocessed parquet files and calculates various metrics
for quality control and analysis.

Usage:
    python calculate_metrics.py <subject_folder>
    Example: python calculate_metrics.py sub-SK
"""

import os
import sys
import logging
from typing import List

import polars as pl

from rdoc_package.analyze.tasks import (
    ax_cpt_rdoc,
    cued_task_switching_rdoc,
    flanker_rdoc,
    go_nogo_rdoc,
    n_back_rdoc,
    operation_span_rdoc,
    simple_span_rdoc,
    spatial_cueing_rdoc,
    spatial_task_switching_rdoc,
    stop_signal_rdoc,
    stroop_rdoc,
    visual_search_rdoc,
    op_only_span_rdoc,
)

def get_function_mapping(task_dir):
    """Map task directory names to their corresponding analysis functions."""
    # Normalize task directory name to handle different variations
    normalized_dir = task_dir.lower().replace('-', '_')
    
    # Handle special cases
    if normalized_dir in ['opspan', 'op_only_span']:
        return operation_span_rdoc
    elif normalized_dir in ['oponly', 'oponlyspan']:
        return op_only_span_rdoc
    elif normalized_dir in ['simplespan']:
        return simple_span_rdoc
    elif normalized_dir in ['stopsignal']:
        return stop_signal_rdoc
    elif normalized_dir in ['gonogo']:
        return go_nogo_rdoc
    elif normalized_dir in ['axcpt']:
        return ax_cpt_rdoc
    
    # Standard mappings
    mappings = {
        'ax_cpt': ax_cpt_rdoc,
        'flanker': flanker_rdoc,
        'go_nogo': go_nogo_rdoc,
        'nback': n_back_rdoc,
        'simple_span': simple_span_rdoc,
        'spatialcueing': spatial_cueing_rdoc,
        'cuedts': cued_task_switching_rdoc,
        'stop_signal': stop_signal_rdoc,
        'visualsearch': visual_search_rdoc,
        'stroop': stroop_rdoc,
        'operation_span': operation_span_rdoc,
        'spatialts': spatial_task_switching_rdoc,
    }
    return mappings.get(normalized_dir, None)

def main():
    # Check if subject folder is provided
    if len(sys.argv) != 2:
        print("Error: Please provide a subject folder name")
        print("Usage: python calculate_metrics.py <subject_folder>")
        print("Example: python calculate_metrics.py sub-SK")
        sys.exit(1)

    subject_folder = sys.argv[1]
    
    # Configure logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

    # Set up input directory
    preprocessed_dir = os.path.join('preprocessed_data', subject_folder)

    # Check if directory exists
    if not os.path.exists(preprocessed_dir):
        logging.error(f'Directory {preprocessed_dir} does not exist')
        sys.exit(1)

    # Get all task directories
    task_dirs = [d for d in os.listdir(preprocessed_dir) 
                if os.path.isdir(os.path.join(preprocessed_dir, d))]

    # Prepare output directory
    output_dir = os.path.join('outputs', subject_folder)
    metrics_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Process each task
    for task_dir in task_dirs:
        # Convert task directory name to experiment name format
        exp_name = task_dir.lower().replace('-', '_') + '_rdoc'
        
        # Get the analysis function for this task
        analysis_func = get_function_mapping(task_dir)
        if analysis_func is None:
            logging.warning(f'Unknown task: {task_dir}')
            continue

        # Get all parquet files for this task
        task_path = os.path.join(preprocessed_dir, task_dir)
        parquet_files = [f for f in os.listdir(task_path) if f.endswith('.parquet')]

        if not parquet_files:
            logging.warning(f'No parquet files found in {task_path}')
            continue

        task_metrics_list = []
        # Process each file
        for parquet_file in parquet_files:
            file_path = os.path.join(task_path, parquet_file)
            try:
                # Read the parquet file
                df = pl.read_parquet(file_path)
                
                # Calculate metrics
                metrics = analysis_func(df)
                
                # Add task and file information
                metrics = metrics.with_columns([
                    pl.lit(task_dir).alias('task'),
                    pl.lit(parquet_file).alias('file')
                ])
                
                # Reorder columns to put metric first
                metrics = metrics.select(['metric', 'value', 'task', 'file'])
                
                task_metrics_list.append(metrics)
                
            except Exception as e:
                logging.error(f'Error processing {file_path}: {str(e)}')
                continue

        # Save all metrics for this task to CSV
        if task_metrics_list:
            task_metrics_df = pl.concat(task_metrics_list)
            # Special case for operation span to ensure lowercase 'o'
            if task_dir.lower() in ['opspan', 'op_only_span']:
                metrics_csv_path = os.path.join(metrics_dir, 'opSpan_metrics.csv')
            else:
                metrics_csv_path = os.path.join(metrics_dir, f'{task_dir}_metrics.csv')
            task_metrics_df.write_csv(metrics_csv_path)
            logging.info(f'Metrics for {task_dir} saved to {metrics_csv_path}')

if __name__ == '__main__':
    main() 