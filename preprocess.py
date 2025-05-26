"""Preprocesses raw behavioral JSON files from Expfactory Deploy and converts them to parquet format for analysis.

Usage:
    python preprocess.py <subject_folder>
    Example: python preprocess.py sub-sM

This script expects input files in:
    rdoc_dropbox:rdoc_fmri_behavior/output/archive/<subject_folder>/**/*.json
and outputs to:
    preprocessed_data/<subject_folder>/<task_name>/<parquet files>
"""

import json
import logging
import os
import sys
import subprocess
from typing import List, Dict, Set
from pathlib import Path

import polars as pl


def scan_parquet_files() -> Dict[str, Set[str]]:
    """Scan all parquet files in preprocessed_data to build a dictionary of required columns for each task.
    
    Returns:
        Dict[str, Set[str]]: Dictionary mapping task names to sets of required columns
    """
    task_columns = {}
    preprocessed_dir = 'preprocessed_data'
    
    # Walk through all subject directories
    for subject_dir in os.listdir(preprocessed_dir):
        subject_path = os.path.join(preprocessed_dir, subject_dir)
        if not os.path.isdir(subject_path):
            continue
            
        # Walk through all task directories
        for task_dir in os.listdir(subject_path):
            task_path = os.path.join(subject_path, task_dir)
            if not os.path.isdir(task_path):
                continue
                
            # Get all parquet files for this task
            parquet_files = [f for f in os.listdir(task_path) if f.endswith('.parquet')]
            if not parquet_files:
                continue
                
            # Read the first parquet file to get columns
            first_file = os.path.join(task_path, parquet_files[0])
            df = pl.read_parquet(first_file)
            
            # Store columns for this task
            task_columns[task_dir] = set(df.columns)
            
    return task_columns


# Global variable to store task columns
TASK_COLUMNS = scan_parquet_files()


def get_required_columns(task_name: str) -> Set[str]:
    """Get the required columns for a given task based on existing parquet files.
    
    Args:
        task_name (str): Name of the task
        
    Returns:
        Set[str]: Set of required column names
    """
    # Task-specific required columns based on parquet files
    task_columns = {
        'visualSearch': {
            'ITIParams', 'accuracy', 'block_duration', 'block_level_feedback', 'block_num', 
            'choices', 'condition', 'correct_response', 'correct_trial', 'design_perm', 
            'exp_stage', 'internal_node_id', 'motor_perm', 'num_stimuli', 
            'order_and_color_of_rectangles', 'response', 'rt', 'stimulus', 'stimulus_duration', 
            'success', 'target_present', 'target_rectangle_location', 'time_elapsed', 
            'trial_duration', 'trial_id', 'trial_index', 'trial_type'
        },
        'stroop': {
            'ITIParams', 'accuracy', 'block_duration', 'block_level_feedback', 'block_num', 
            'choices', 'condition', 'correct_response', 'correct_trial', 'design_perm', 
            'exp_stage', 'internal_node_id', 'motor_perm', 'response', 'rt', 'stim_color', 
            'stim_word', 'stimulus', 'stimulus_duration', 'success', 'time_elapsed', 
            'trial_duration', 'trial_id', 'trial_index', 'trial_type'
        },
        'flanker': {
            'ITIParams', 'accuracy', 'block_duration', 'block_level_feedback', 'block_num', 
            'center_letter', 'choices', 'condition', 'correct_response', 'correct_trial', 
            'design_perm', 'exp_stage', 'flanker', 'internal_node_id', 'motor_perm', 
            'response', 'rt', 'stimulus', 'stimulus_duration', 'success', 'time_elapsed', 
            'trial_duration', 'trial_id', 'trial_index', 'trial_type'
        },
        'spatialTS': {
            'ITIParams', 'accuracy', 'block_duration', 'block_level_feedback', 'block_num', 
            'choices', 'color', 'condition', 'correct_response', 'correct_trial', 
            'current_block', 'current_trial', 'design_perm', 'exp_stage', 'form', 
            'internal_node_id', 'motor_perm', 'response', 'rt', 'shape', 'spatial_cue', 
            'stimulus', 'stimulus_duration', 'success', 'time_elapsed', 'trial_duration', 
            'trial_id', 'trial_index', 'trial_type', 'whichQuadrant'
        },
        'spatialCueing': {
            'CTI_duration', 'ITIParams', 'accuracy', 'block_duration', 'block_level_feedback', 
            'block_num', 'choices', 'condition', 'correct_response', 'correct_trial', 
            'cue_location', 'design_perm', 'exp_stage', 'internal_node_id', 'response', 
            'rt', 'stim_location', 'stimulus', 'stimulus_duration', 'success', 
            'time_elapsed', 'trial_duration', 'trial_id', 'trial_index', 'trial_type'
        },
        'nBack': {
            'ITIParams', 'accuracy', 'block_duration', 'block_level_feedback', 'block_num', 
            'choices', 'condition', 'correct_response', 'correct_trial', 'delay', 
            'design_perm', 'exp_stage', 'internal_node_id', 'letter_case', 'motor_perm', 
            'probe', 'response', 'rt', 'stimulus', 'stimulus_duration', 'success', 
            'time_elapsed', 'trial_duration', 'trial_id', 'trial_index', 'trial_type'
        },
        'cuedTS': {
            'CTI', 'ITIParams', 'accuracy', 'block_duration', 'block_level_feedback', 
            'block_num', 'choices', 'condition', 'correct_response', 'correct_trial', 
            'cue', 'cue_condition', 'current_trial', 'design_perm', 'exp_stage', 
            'internal_node_id', 'motor_perm', 'response', 'rt', 'stim_number', 
            'stimulus', 'stimulus_duration', 'success', 'task', 'task_condition', 
            'time_elapsed', 'trial_duration', 'trial_id', 'trial_index', 'trial_type'
        },
        'axCPT': {
            'ITIParams', 'accuracy', 'block_duration', 'block_level_feedback', 'block_num', 
            'choices', 'condition', 'correct_response', 'correct_trial', 'cue_letter', 
            'delay_ms', 'design_perm', 'exp_stage', 'internal_node_id', 'motor_perm', 
            'probe_letter', 'response', 'rt', 'stimulus', 'stimulus_duration', 'success', 
            'time_elapsed', 'trial_duration', 'trial_id', 'trial_index', 'trial_type'
        },
        'goNogo': {
            'ITIParams', 'accuracy', 'block_duration', 'block_level_feedback', 'block_num', 
            'choices', 'condition', 'correct_response', 'correct_trial', 'design_perm', 
            'exp_stage', 'internal_node_id', 'response', 'rt', 'shape', 'stimulus', 
            'stimulus_duration', 'success', 'time_elapsed', 'trial_duration', 'trial_id', 
            'trial_index', 'trial_type'
        },
        'simpleSpan': {
            'ITIParams', 'accuracy', 'block_duration', 'block_level_feedback', 'block_num', 
            'cell_order_through_grid', 'choices', 'condition', 'correct_cell_order', 
            'correct_spatial_judgement_key', 'correct_trial', 'design_perm', 
            'duplicate_responses', 'duplicate_responses_timestamps', 'exp_stage', 
            'extra_responses', 'extra_responses_timestamps', 'internal_node_id', 
            'moving_through_grid_timestamps', 'response', 'rt', 'spatial_location', 
            'starting_cell', 'stimulus', 'stimulus_duration', 'success', 'time_elapsed', 
            'trial_duration', 'trial_id', 'trial_index', 'trial_type', 'valid_responses', 
            'valid_responses_timestamps'
        },
        'stopSignal': {
            'ITIParams', 'SSD', 'SS_duration', 'SS_stimulus', 'SS_trial_type', 'accuracy', 
            'block_duration', 'block_level_feedback', 'block_num', 'choices', 'condition', 
            'correct_response', 'correct_trial', 'current_trial', 'design_perm', 'exp_stage', 
            'internal_node_id', 'motor_perm', 'response', 'response_ends_trial', 'rt', 
            'stim', 'stimulus', 'stimulus_duration', 'success', 'time_elapsed', 
            'trial_duration', 'trial_id', 'trial_index', 'trial_type'
        },
        'OpSpan': {
            'ITIParams', 'accuracy', 'block_duration', 'block_level_feedback', 'block_num', 
            'cell_order_through_grid', 'choices', 'condition', 'correct_cell_order', 
            'correct_spatial_judgement_key', 'correct_trial', 'design_perm', 
            'duplicate_responses', 'duplicate_responses_timestamps', 'exp_stage', 
            'extra_responses', 'extra_responses_timestamps', 'grid_symmetry', 
            'internal_node_id', 'motor_perm', 'moving_through_grid_timestamps', 
            'order_and_color_of_processing_boxes', 'response', 'rt', 'spatial_location', 
            'starting_cell', 'stimulus', 'stimulus_duration', 'success', 'time_elapsed', 
            'trial_duration', 'trial_id', 'trial_index', 'trial_type', 'valid_responses', 
            'valid_responses_timestamps'
        },
        'opOnly': {
            'accuracy', 'design_perm', 'internal_node_id', 'motor_perm', 'response', 
            'rt', 'time_elapsed', 'trial_index', 'trial_type'
        }
    }
    
    if task_name not in task_columns:
        logging.error(f"No column requirements found for task: {task_name}")
        return set()
        
    return task_columns[task_name]


def validate_columns(df: pl.DataFrame, task_name: str) -> bool:
    """Validate that all required columns are present in the DataFrame.
    
    Args:
        df (pl.DataFrame): DataFrame to validate
        task_name (str): Name of the task
        
    Returns:
        bool: True if all required columns are present, False otherwise
    """
    required_columns = get_required_columns(task_name)
    missing_columns = required_columns - set(df.columns)
    
    if missing_columns:
        print(f"ERROR: The data file labeled {task_name} is missing the expected columns for {task_name}. Please check that this file actually contains data for {task_name}.")
        return False
    
    return True


def convert_lists_to_strings(data: List[Dict]) -> List[Dict]:
    """Convert lists and complex objects to strings.

    This is necessary because the data is stored as a string in the JSON file.
    """
    for item in data:
        for key, value in item.items():
            if isinstance(value, (list, dict)):
                item[key] = str(value)
    return data


def load_json(fpath: str) -> Dict:
    """Load a JSON file.

    Args:
        fpath (str): Path to the JSON file.

    Returns:
        Dict: The loaded JSON data.
    """
    with open(fpath, 'r') as fp:
        return json.load(fp)


def get_trialdata_df(data: Dict) -> pl.DataFrame:
    """Convert trial data to a polars DataFrame.

    Args:
        data (Dict): Dictionary containing trial data.

    Returns:
        pl.DataFrame: DataFrame containing the trial data.
    """
    # Handle case where trialdata is stored as a string
    if isinstance(data, dict) and 'trialdata' in data:
        trialdata = data['trialdata']
        if isinstance(trialdata, str):
            trialdata = json.loads(trialdata)
        data = trialdata
    
    # Ensure data is a list of dictionaries
    if not isinstance(data, list):
        raise ValueError("Data must be a list of dictionaries")
    
    # Convert lists and complex objects to strings
    data = convert_lists_to_strings(data)
    return pl.from_dicts(data)


def main():
    # Check if subject folder is provided
    if len(sys.argv) != 2:
        print("Error: Please provide a subject folder name")
        print("Usage: python preprocess.py <subject_folder>")
        print("Example: python preprocess.py sub-sM")
        sys.exit(1)

    subject_folder = sys.argv[1]
    
    # Configure logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    # Set up directories
    raw_data_dir = os.path.join('raw_data', subject_folder)
    preprocessed_dir = os.path.join('preprocessed_data', subject_folder)

    # Create preprocessed directory
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Get all JSON files in the raw_data directory and its subdirectories
    json_files = []
    for root, _, files in os.walk(raw_data_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))

    total_files = len(json_files)
    logging.info(f'Found {total_files} JSON files to process')

    for index, filepath in enumerate(json_files, start=1):
        logging.info(f'Processing file {index} of {total_files}: {filepath}')

        # Extract task name and date_time from filename
        # Filename format: sub-SK_ses-1_task-{task_name}_dateTime-{timestamp}_run-1.json
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        task_name = None
        for part in parts:
            if part.startswith('task-'):
                task_name = part[5:]  # Remove 'task-' prefix
                break
        
        if task_name is None:
            logging.error(f'Could not extract task name from {filename}')
            continue

        # Create output directory structure
        task_dir = os.path.join(preprocessed_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        # Create output filename
        outname = f'{subject_folder}_task-{task_name}_{filename[:-5]}.parquet'  # Remove .json extension
        outpath = os.path.join(task_dir, outname)

        # Skip if file already exists
        if os.path.isfile(outpath):
            logging.info(f'Skipping {outpath} because it already exists')
            continue

        try:
            # Load and process the JSON data
            data = load_json(filepath)
            df = get_trialdata_df(data)
            
            # Validate required columns
            if not validate_columns(df, task_name):
                logging.error(f'Skipping {filepath} due to missing required columns')
                continue
            
            # Save as parquet
            df.write_parquet(outpath)
            logging.info(f'Saved preprocessed data to {outpath}')
            
        except Exception as e:
            logging.error(f'Error processing {filepath}: {str(e)}')
            continue


if __name__ == '__main__':
    main() 