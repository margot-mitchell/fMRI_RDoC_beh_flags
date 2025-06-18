"""Utility functions for handling file naming and task identification in RDOC data.

This module provides functions for:
1. Extracting exp_id from JSON files
2. Mapping exp_ids to canonical task names
3. Validating exp_ids against expected task names
4. Renaming files based on task names
"""

import json
import logging
import os
from typing import Dict, Optional

# Task name mapping based on exp_id patterns
TASK_MAPPING = {
    'ax_cpt': 'axCPT',
    'axcpt': 'axCPT',
    'cued_task_switching': 'cuedTS',
    'cuedts': 'cuedTS',
    'flanker': 'flanker',
    'go_nogo': 'goNogo',
    'gonogo': 'goNogo',
    'spatial_task_switching': 'spatialTS',
    'spatialts': 'spatialTS',
    'stop_signal': 'stopSignal',
    'stopsignal': 'stopSignal',
    'stroop': 'stroop',
    'simple_span': 'simpleSpan',
    'simplespan': 'simpleSpan',
    'operation_span': 'opSpan',
    'opspan': 'opSpan',
    'operation_only': 'opOnly',
    'oponly': 'opOnly',
    'n_back': 'nBack',
    'nback': 'nBack',
    'spatial_cueing': 'spatialCueing',
    'spatialcueing': 'spatialCueing',
    'visual_search': 'visualSearch',
    'visualsearch': 'visualSearch'
}

def get_exp_id(filepath: str) -> Optional[str]:
    """Get the exp_id from a JSON file.
    
    Args:
        filepath (str): Path to the JSON file
        
    Returns:
        Optional[str]: The exp_id from the file, or None if not found
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Look for exp_id in the data
        if isinstance(data, dict):
            if 'exp_id' in data:
                return data['exp_id']
            elif 'trialdata' in data:
                # Try to find exp_id in trialdata
                trialdata = data['trialdata']
                if isinstance(trialdata, str):
                    try:
                        trialdata = json.loads(trialdata)
                    except json.JSONDecodeError:
                        return None
                if isinstance(trialdata, list) and len(trialdata) > 0:
                    for trial in trialdata:
                        if isinstance(trial, dict) and 'exp_id' in trial:
                            return trial['exp_id']
        elif isinstance(data, list):
            # Search all trials for exp_id
            for trial in data:
                if isinstance(trial, dict) and 'exp_id' in trial:
                    return trial['exp_id']
    except Exception as e:
        logging.error(f"Error reading {filepath}: {str(e)}")
    return None

def get_task_from_exp_id(exp_id: str) -> Optional[str]:
    """Get the canonical task name from exp_id.
    
    Args:
        exp_id (str): The exp_id from the data
        
    Returns:
        Optional[str]: The canonical task name, or None if not found
    """
    if not exp_id:
        return None
        
    # Convert to lowercase for comparison
    exp_id = exp_id.lower()
    
    # Remove _practice__fmri suffix if present
    if exp_id.endswith('_practice__fmri'):
        exp_id = exp_id[:-15]  # Remove _practice__fmri
    
    # Try exact match first
    if exp_id in TASK_MAPPING:
        return TASK_MAPPING[exp_id]
        
    # Try partial match
    for key, value in TASK_MAPPING.items():
        if key in exp_id:
            return value
            
    return None

def validate_exp_id(exp_id: str, task_name: str, is_practice: bool) -> bool:
    """Validate that the exp_id matches the expected task name for practice files.
    
    Args:
        exp_id (str): The exp_id extracted from the file
        task_name (str): Expected task name
        is_practice (bool): Whether this is a practice file
        
    Returns:
        bool: True if exp_id matches or if not a practice file, False otherwise
    """
    # Only validate practice files
    if not is_practice:
        return True
    if not exp_id:
        logging.error(f"Missing exp_id in practice data")
        return False

    # For practice files, exp_id should contain the task name and practice
    task_base = task_name.replace('_practice', '')  # Remove _practice suffix if present
    
    # Convert task name to lowercase for comparison
    task_base = task_base.lower()
    exp_id = exp_id.lower()
    
    # Clean up task_base for comparison
    task_base = task_base.replace('_rdoc', '').replace('_practice', '')
    
    # Check if the task name is contained in the exp_id
    if task_base not in exp_id:
        # Try alternative task names
        alt_names = {
            'axcpt': 'ax_cpt',
            'gong': 'go_nogo',
            'spatialcueing': 'spatial_cueing',
            'visualsearch': 'visual_search',
            'stopsignal': 'stop_signal',
            'simplespan': 'simple_span',
            'opspan': 'operation_span',
            'oponly': 'operation_only',
            'nback': 'n_back',
            'cuedts': 'cued_task_switching',
            'spatialts': 'spatial_task_switching',
            # Add practice-specific variations
            'operation_span_rdoc': 'opspan',
            'operation_only_rdoc': 'oponly',
            'operation_span': 'opspan',
            'operation_only': 'oponly',
            'span_rdoc': 'opspan',
            'only_rdoc': 'oponly'
        }
        
        # Try alternative names
        for alt, standard in alt_names.items():
            if alt in exp_id and standard in task_base:
                return True
            # Also try matching with _practice suffix
            if alt in exp_id and f"{standard}_practice" in task_base:
                return True
                
        # Try matching just the first part of the task name
        task_first_part = task_base.split('_')[0]
        exp_id_first_part = exp_id.split('_')[0]
        if task_first_part in exp_id_first_part or exp_id_first_part in task_first_part:
            return True
            
        logging.error(f"Practice file exp_id {exp_id} does not match expected task {task_name}")
        return False
        
    return True

def rename_file_task_part(filepath: str, new_task: str) -> None:
    """Rename a file by replacing its task part with the new task name.
    
    Args:
        filepath (str): Path to the file to rename
        new_task (str): New task name to use
    """
    filename = os.path.basename(filepath)
    parts = filename.split('_')
    
    # Find the task part
    for i, part in enumerate(parts):
        if part.startswith('task-'):
            # Replace the task part
            parts[i] = f'task-{new_task}'
            break
    
    # Create new filename
    new_filename = '_'.join(parts)
    new_filepath = os.path.join(os.path.dirname(filepath), new_filename)
    
    # Rename the file
    os.rename(filepath, new_filepath)
    logging.info(f'Renamed {filepath} to {new_filepath} (task: {parts[i][5:]} -> {new_task})') 