"""Preprocess behavioral data from JSON to parquet format."""

import glob
import json
import os
from typing import Optional

import polars as pl


def load_json_data(file_path: str) -> Optional[pl.DataFrame]:
    """Load JSON data from a file.

    Args:
        file_path (str): Path to JSON file.

    Returns:
        Optional[pl.DataFrame]: DataFrame containing the data, or None if loading fails.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Convert trial data to DataFrame
        if 'trial_data' in data:
            df = pl.DataFrame(data['trial_data'])
        else:
            df = pl.DataFrame(data)
            
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def process_subject_folder(subject_folder: str) -> None:
    """Process all files in a subject folder.
    
    Args:
        subject_folder (str): Path to subject folder containing task data.
    """
    # Get list of task folders that exist
    task_folders = [f for f in os.listdir(subject_folder) 
                   if os.path.isdir(os.path.join(subject_folder, f))]
    
    for task_folder in task_folders:
        task_files = glob.glob(os.path.join(subject_folder, task_folder, "*.json"))
        if not task_files:
            continue
            
        # Process each task file
        for task_file in task_files:
            # Convert to parquet
            df = load_json_data(task_file)
            if df is not None:
                output_file = task_file.replace('.json', '.parquet')
                df.write_parquet(output_file)


def main():
    """Main function to process all subject folders."""
    # Get list of subject folders
    subject_folders = [f for f in os.listdir('preprocessed_data') 
                      if os.path.isdir(os.path.join('preprocessed_data', f))]
    
    for subject_folder in subject_folders:
        print(f"Processing {subject_folder}...")
        process_subject_folder(os.path.join('preprocessed_data', subject_folder))


if __name__ == '__main__':
    main() 