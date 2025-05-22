#!/bin/bash

# Exit on error
set -e

# Check if subject name is provided
if [ -z "$1" ]; then
    echo "Error: Please provide a subject name"
    echo "Usage: ./process_subject.sh <subject_name>"
    echo "Example: ./process_subject.sh sub-s2"
    exit 1
fi

SUBJECT=$1
echo "Processing subject: $SUBJECT"

# Check if subject exists in Dropbox
if ! rclone lsd "rdoc_dropbox:rdoc_fmri_behavior/output/raw/$SUBJECT" &>/dev/null; then
    echo "Error: Subject $SUBJECT not found in Dropbox"
    echo "Available subjects:"
    rclone lsd rdoc_dropbox:rdoc_fmri_behavior/output/raw/
    exit 1
fi

# Create necessary directories
mkdir -p preprocessed_data
mkdir -p outputs
mkdir -p raw_data

# Sync subject data from Dropbox
echo "Syncing data from Dropbox..."
rclone copy "rdoc_dropbox:rdoc_fmri_behavior/output/raw/$SUBJECT" "./raw_data/$SUBJECT"

# Check if data was copied successfully
if [ ! -d "./raw_data/$SUBJECT" ]; then
    echo "Error: Failed to copy data from Dropbox"
    exit 1
fi

# Run preprocessing
echo "Running preprocessing..."
python preprocess.py "$SUBJECT"

# Run analysis
echo "Running analysis..."
python analyze_behavioral_data.py "$SUBJECT"

# Upload results to Dropbox
echo "Uploading results to Dropbox..."
timestamp=$(date +%Y%m%d_%H%M%S)

# Create output directories in Dropbox if they don't exist
rclone mkdir "rdoc_dropbox:rdoc_fmri_behavior/output/preprocessed_data"
rclone mkdir "rdoc_dropbox:rdoc_fmri_behavior/output/analysis_outputs"
rclone mkdir "rdoc_dropbox:rdoc_fmri_behavior/output/archive/raw_${timestamp}"

# Upload preprocessed data
echo "Uploading preprocessed data..."
rclone copy "./preprocessed_data/$SUBJECT" "rdoc_dropbox:rdoc_fmri_behavior/output/preprocessed_data/$SUBJECT"

# Upload analysis outputs
echo "Uploading analysis outputs..."
rclone copy "./outputs/$SUBJECT" "rdoc_dropbox:rdoc_fmri_behavior/output/analysis_outputs/$SUBJECT"

# Archive the raw data
echo "Archiving raw data..."
rclone copy "./raw_data/$SUBJECT" "rdoc_dropbox:rdoc_fmri_behavior/output/archive/raw_${timestamp}/$SUBJECT"

# Clean up
echo "Cleaning up..."
rm -rf "./raw_data/$SUBJECT"

echo "Processing complete for $SUBJECT"
echo "Results are available in:"
echo "- Preprocessed data: rdoc_fmri_behavior/output/preprocessed_data/$SUBJECT"
echo "- Analysis outputs: rdoc_fmri_behavior/output/analysis_outputs/$SUBJECT"
echo "- Archived raw data: rdoc_fmri_behavior/output/archive/raw_${timestamp}/$SUBJECT" 