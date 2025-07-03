"""Corrects raw file names based on their exp_id to ensure consistent task naming.

Usage:
    python correct_raw_file_names.py <subject_folder>
    Example: python correct_raw_file_names.py sub-sM

This script expects input files in:
    output/raw/<subject_folder>/**/*.json

The script:
1. Finds all JSON files in the specified subject folder
2. Reads their exp_id from the JSON
3. Renames files to match the canonical task name from exp_id
4. Reports any files missing exp_id or with unmappable task names
"""

import logging
import os
import sys
from pathlib import Path

# Import task renaming functions from the new module
from rdoc_package.utils.file_naming import (
    get_exp_id,
    get_task_from_exp_id,
    rename_file_task_part
)

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
        print("Usage: python correct_raw_file_names.py <subject_folder>")
        print("Example: python correct_raw_file_names.py sub-sM")
        print("Or run without arguments to process all subjects.")
        sys.exit(1)

    for subject_folder in subject_folders:
        input_dir = os.path.join('output', 'raw', subject_folder)
        print(f"Checking task names for {subject_folder}...")
        missing_exp_id = []
        total_files = 0
        renamed_files = 0
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.json'):
                    total_files += 1
                    filepath = os.path.join(root, file)
                    exp_id = get_exp_id(filepath)
                    if not exp_id:
                        missing_exp_id.append(filepath)
                        continue
                    canonical_task = get_task_from_exp_id(exp_id)
                    if not canonical_task:
                        logging.error(f"Could not determine task from exp_id {exp_id} in {filepath}")
                        continue
                    # Find the task part in the filename
                    parts = file.split('_')
                    task_index = None
                    for i, part in enumerate(parts):
                        if part.startswith('task-'):
                            task_index = i
                            break
                    if task_index is None:
                        logging.error(f"Could not find task part in {file}")
                        continue
                    old_task = parts[task_index][5:]
                    if old_task != canonical_task:
                        rename_file_task_part(filepath, canonical_task)
                        renamed_files += 1

        print(f"\nTask name check summary for {subject_folder}:")
        print(f"Total files checked: {total_files}")
        print(f"Files renamed: {renamed_files}")
        if missing_exp_id:
            print(f"Files missing exp_id: {len(missing_exp_id)}")
            for f in missing_exp_id:
                print(f"  {f}")
        else:
            print("All files had exp_id.")


if __name__ == '__main__':
    main() 