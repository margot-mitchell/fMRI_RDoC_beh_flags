"""Preprocesses raw behavioral JSON files from Expfactory Deploy and converts them to parquet format for analysis.

Usage:
    python preprocess.py <subject_folder>
    Example: python preprocess.py sub-sK

This script expects input files in:
    ./raw_data/<subject_folder>/**/*.json
and outputs to:
    preprocessed_data/<subject_folder>/<task_name>/<parquet files>
"""

import json
import logging
import os
import sys
from typing import List, Dict
from pathlib import Path

import polars as pl


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
        print("Example: python preprocess.py sub-sK")
        sys.exit(1)

    subject_folder = sys.argv[1]
    
    # Configure logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    # Set up input and output directories
    raw_data_dir = os.path.join('raw_data', subject_folder)
    preprocessed_dir = os.path.join('preprocessed_data', subject_folder)

    # Create preprocessed_data directory if it doesn't exist
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
            
            # Save as parquet
            df.write_parquet(outpath)
            logging.info(f'Saved preprocessed data to {outpath}')
            
        except Exception as e:
            logging.error(f'Error processing {filepath}: {str(e)}')
            continue


if __name__ == '__main__':
    main() 