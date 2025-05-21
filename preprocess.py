"""Preprocesses raw data fetched from Expfactory Deploy.

Before running this script, you must download the data from the Experiments Factory API.
To do so, run `download.sh` in the root directory.

Usage:
    python preprocess.py <subject_folder>
    Example: python preprocess.py sub-SK
"""

import json
import logging
import os
import sys
import pandas as pd

from rdoc_package.preprocess.feedback import add_feedback

def load_json(fpath: str) -> dict:
    """Load a JSON file.

    This function loads a JSON file and returns a dictionary.
    """
    with open(fpath, 'r') as fp:
        return json.load(fp)

def get_trialdata_df(data: list) -> pd.DataFrame:
    """Get the trialdata dataframe.

    This function takes a list of trial data and converts it to a pandas dataframe.
    """
    return pd.DataFrame(data)

def main():
    # Check if subject folder is provided
    if len(sys.argv) != 2:
        print("Error: Please provide a subject folder name")
        print("Usage: python preprocess.py <subject_folder>")
        print("Example: python preprocess.py sub-SK")
        sys.exit(1)

    subject_folder = sys.argv[1]
    
    # Configure logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    # Set up input and output directories
    raw_data_dir = os.path.expanduser(f'~/Desktop/{subject_folder}')
    output_dir = os.path.join('preprocessed_data', subject_folder)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.json')]
    total_files = len(json_files)

    for index, filename in enumerate(json_files):
        logging.info(f'Processing file {index + 1} of {total_files}')

        # Extract metadata from filename
        parts = filename.split('_')
        subject = parts[0].split('-')[1]  # sub-SK -> SK
        exp_name = parts[2].split('-')[1]  # task-cuedTS -> cuedTS
        date_time = parts[3].split('-')[1]  # dateTime-1746661495 -> 1746661495

        fpath = os.path.join(raw_data_dir, filename)
        data = load_json(fpath)
        
        # Handle case where data is a dict with 'trialdata' as a string
        if isinstance(data, dict) and 'trialdata' in data:
            trialdata = data['trialdata']
            if isinstance(trialdata, str):
                import json as _json
                trialdata = _json.loads(trialdata)
            data = trialdata
        
        if not isinstance(data, list):
            logging.warning(f'Skipping {filename}: JSON root is not a list or valid trialdata')
            continue
        
        # Process data
        df = get_trialdata_df(data)

        # Add feedback information
        df = add_feedback(df, exp_name)

        # Compile outpath
        outname = f'sub-{subject}_exp-{exp_name}_dateTime-{date_time}.csv'
        exp_dir = os.path.join(output_dir, exp_name)
        outpath = os.path.join(exp_dir, outname)

        # Create directory if it doesn't exist
        os.makedirs(exp_dir, exist_ok=True)

        # Save as CSV
        df.to_csv(outpath, index=False)
        logging.info(f'Saved preprocessed data to {outpath}')

if __name__ == '__main__':
    main() 