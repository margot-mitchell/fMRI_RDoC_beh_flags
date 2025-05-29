"""Preprocesses raw behavioral JSON files and converts them to parquet format for analysis.

Usage:
    python preprocess.py <subject_folder>
    Example: python preprocess.py sub-sM

This script expects input files in:
    output/raw/<subject_folder>/**/*.json
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
        'opSpan': {
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
            try:
                # First try to parse the trialdata string as JSON
                trialdata = json.loads(trialdata)
            except json.JSONDecodeError:
                # If that fails, try to parse it as a list of JSON objects
                trialdata = [json.loads(item) for item in trialdata.strip('[]').split(',')]
        data = trialdata
    
    # Ensure data is a list of dictionaries
    if not isinstance(data, list):
        raise ValueError("Data must be a list of dictionaries")
    
    # Convert lists and complex objects to strings
    data = convert_lists_to_strings(data)
    return pl.from_dicts(data)


def process_file(filepath: str, output_dir: str) -> None:
    """Process a single JSON file and save it as parquet.
    
    Args:
        filepath (str): Path to the input JSON file
        output_dir (str): Directory to save the output parquet file
    """
    filename = os.path.basename(filepath)
    parts = filename.split('_')
    task_name = None
    task_parts = []
    found_task = False
    
    # Extract task name from filename
    for part in parts:
        if part.startswith('task-'):
            found_task = True
            task_parts.append(part[5:])  # Remove 'task-' prefix
        elif found_task and not part.startswith(('ses-', 'run-', 'fmri', 'dateTime')):
            task_parts.append(part)
        elif found_task and (part.startswith(('ses-', 'run-', 'fmri', 'dateTime')) or part == 'practice'):
            break
    
    if task_parts:
        task_name = '_'.join(task_parts)
    else:
        logging.error(f'Could not extract task name from {filepath}')
        return

    # Task name mapping
    task_mapping = {
        # AX-CPT variations
        'ax_cpt_rdoc': 'axCPT',
        'ax_cpt': 'axCPT',
        'axCPT': 'axCPT',
        
        # Cued Task Switching variations
        'cued_task_switching_rdoc': 'cuedTS',
        'cued_task_switching': 'cuedTS',
        'cuedTS': 'cuedTS',
        
        # Flanker variations
        'flanker_rdoc': 'flanker',
        'flanker': 'flanker',
        
        # Go/No-Go variations
        'go_nogo_rdoc': 'goNogo',
        'go_nogo': 'goNogo',
        'goNogo': 'goNogo',
        
        # Spatial Task Switching variations
        'spatial_task_switching_rdoc': 'spatialTS',
        'spatial_task_switching': 'spatialTS',
        'spatialTS': 'spatialTS',
        
        # Stop Signal variations
        'stop_signal_rdoc': 'stopSignal',
        'stop_signal': 'stopSignal',
        'stopSignal': 'stopSignal',
        
        # Stroop variations
        'stroop_rdoc': 'stroop',
        'stroop': 'stroop',
        
        # Simple Span variations
        'simple_span_rdoc': 'simpleSpan',
        'simple_span': 'simpleSpan',
        'simpleSpan': 'simpleSpan',
        
        # Operation Span variations
        'operation_span_rdoc': 'opSpan',
        'operation_span': 'opSpan',
        'opSpan': 'opSpan',
        'OpSpan': 'opSpan',
        
        # Operation Only variations
        'operation_only_rdoc': 'opOnly',
        'operation_only': 'opOnly',
        'opOnly': 'opOnly',
        
        # N-Back variations
        'n_back_rdoc': 'nBack',
        'n_back': 'nBack',
        'nBack': 'nBack',
        
        # Spatial Cueing variations
        'spatial_cueing_rdoc': 'spatialCueing',
        'spatial_cueing': 'spatialCueing',
        'spatialCueing': 'spatialCueing',
        
        # Visual Search variations
        'visual_search_rdoc': 'visualSearch',
        'visual_search': 'visualSearch',
        'visualSearch': 'visualSearch',
        'visual': 'visualSearch'
    }

    # Determine subfolder name
    subfolder_name = None
    if 'pretouch' in filename.lower() or 'practice' in filename.lower():
        # Handle practice files
        task_name_lower = task_name.lower()
        if task_name in task_mapping:
            subfolder_name = task_mapping[task_name] + "_practice"
        elif task_name_lower in task_mapping:
            subfolder_name = task_mapping[task_name_lower] + "_practice"
        else:
            task_name_with_rdoc = task_name_lower + '_rdoc'
            if task_name_with_rdoc in task_mapping:
                subfolder_name = task_mapping[task_name_with_rdoc] + "_practice"
            else:
                for key, mapped_name in task_mapping.items():
                    if key in task_name_lower:
                        subfolder_name = mapped_name + "_practice"
                        break
    else:
        # Handle non-practice files
        if task_name in task_mapping:
            subfolder_name = task_mapping[task_name]
        else:
            task_name_lower = task_name.lower()
            if task_name_lower in task_mapping:
                subfolder_name = task_mapping[task_name_lower]
            else:
                task_name_with_rdoc = task_name_lower + '_rdoc'
                if task_name_with_rdoc in task_mapping:
                    subfolder_name = task_mapping[task_name_with_rdoc]
                else:
                    for key, mapped_name in task_mapping.items():
                        if key in task_name_lower:
                            subfolder_name = mapped_name
                            break

    if subfolder_name is None:
        logging.error(f'Could not determine subfolder name for {filepath}')
        return

    # Create output directory structure
    task_dir = os.path.join(output_dir, subfolder_name)
    os.makedirs(task_dir, exist_ok=True)

    # Create output filename
    run_num = None
    for part in parts:
        if part.startswith('run-'):
            run_num = part[4:]
            break
    
    if run_num:
        outname = f'{os.path.basename(output_dir)}_task-{task_name}_run-{run_num}.parquet'
    else:
        outname = f'{os.path.basename(output_dir)}_task-{task_name}.parquet'
        
    outpath = os.path.join(task_dir, outname)

    # Skip if file already exists
    if os.path.isfile(outpath):
        logging.info(f'Skipping {outpath} because it already exists')
        return

    try:
        # Load and process the JSON data
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        df = get_trialdata_df(data)
        
        # Validate columns for non-practice tasks
        if not ('pretouch' in filename.lower() or 'practice' in filename.lower()):
            if not validate_columns(df, subfolder_name):
                logging.error(f'Skipping {filepath} due to missing required columns')
                return
        
        # Save as parquet
        df.write_parquet(outpath)
        logging.info(f'Saved preprocessed data to {outpath}')
    except Exception as e:
        logging.error(f'Error processing {filepath}: {str(e)}')


def main():
    # Configure logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    if len(sys.argv) == 2:
        # Single subject mode
        subject_folders = [sys.argv[1]]
    elif len(sys.argv) == 1:
        # All subjects mode
        raw_dir = os.path.join('output', 'raw')
        subject_folders = [f for f in os.listdir(raw_dir)
                           if os.path.isdir(os.path.join(raw_dir, f)) and f.startswith('sub-')]
        if not subject_folders:
            print(f"No subject folders found in {raw_dir}")
            sys.exit(1)
        print(f"Processing all subjects: {', '.join(subject_folders)}")
    else:
        print("Usage: python preprocess.py <subject_folder>")
        print("Example: python preprocess.py sub-sM")
        print("Or run without arguments to process all subjects.")
        sys.exit(1)

    for subject_folder in subject_folders:
        input_dir = os.path.join('output', 'raw', subject_folder)
        output_dir = os.path.join('preprocessed_data', subject_folder)
        os.makedirs(output_dir, exist_ok=True)
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.json'):
                    input_file = os.path.join(root, file)
                    process_file(input_file, output_dir)


if __name__ == '__main__':
    main() 