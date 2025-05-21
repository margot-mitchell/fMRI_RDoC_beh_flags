# fMRI RDoC Behavioral Quality Control

This project provides tools for preprocessing and analyzing behavioral data from fMRI RDoC studies. It includes quality control checks for various cognitive tasks and generates summary statistics and quality control flags.

## Setup

### Environment Setup
1. Create a new Python virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install required dependencies:
```bash
pip install pandas numpy
```

### Data Preparation
1. Create a folder on your Desktop with the subject's data files. The folder should be named according to the subject ID (e.g., `sub-s1`).
2. Place the raw behavioral data files in this folder.
3. Alternatively, you can modify the input path in the scripts:
   - In `preprocess.py`: Change `os.path.expanduser(f'~/Desktop/{subject_folder}')` to your desired input path
   - In `analyze_behavioral_data.py`: Change `PREPROCESSED_DIR` to your desired input path

## Usage

### 1. Preprocessing
Run the preprocessing script to prepare the behavioral data:
```bash
python preprocess.py <subject_folder>
```
Example:
```bash
python preprocess.py sub-s1
```

This will:
- Read JSON files from the input folder
- Process the data into CSV format
- Save preprocessed files in `preprocessed_data/<subject_folder>/`

### 2. Analysis
Run the analysis script to perform quality control checks:
```bash
python analyze_behavioral_data.py <subject_folder>
```
Example:
```bash
python analyze_behavioral_data.py sub-s1
```

This will:
- Read preprocessed CSV files
- Calculate summary statistics
- Perform quality control checks
- Generate output files in `outputs/<subject_folder>/`

## Outputs

### 1. Behavioral Summary (`behavioral_summary.csv`)
- Contains summary statistics for each task and condition
- Includes accuracy and omission rates
- Organized by task file and condition

### 2. Quality Control Flags (`<subject_folder>_flags.csv`)
- Lists any quality control flags found during analysis
- Empty file created if no flags are found
- Columns:
  - task_file: Name of the task file
  - task: Task name
  - condition: Condition within the task
  - metric: Metric being checked
  - value: Actual value
  - threshold: Threshold value
  - status: Flag status (exclude)

## Quality Control Thresholds

All quality control thresholds are defined in `thresholds_config.py`. The following tasks are supported:

### General Thresholds
- Accuracy: 55%
- Omission Rate: 50%

### Task-Specific Thresholds

#### Stop Signal Task
- Stop Accuracy: 25-75%
- Go Accuracy: 55%
- Go RT: 750ms

#### AX-CPT Task
- Accuracy: 55%

#### Go/NoGo Task
- Go Accuracy: 85.7%
- NoGo Accuracy: 14.3%
- Mean Accuracy: 55%
- Go Omission Rate: 50%

#### Flanker Task
- Accuracy: 55%
- Omission Rate: 50%

#### Operation Span Task
- 8x8 Asymmetric Grid Accuracy: 55%
- 8x8 Symmetric Grid Accuracy: 55%
- 4x4 Grid Accuracy: 25%
- Order Difference: 40%

#### Simple Span Task
- 8x8 Asymmetric Grid Accuracy: 55%
- 8x8 Symmetric Grid Accuracy: 55%
- 4x4 Grid Accuracy: 55%
- Order Difference: 20%

#### Cued Task Switching
- Accuracy: 55%
- Omission Rate: 50%

#### N-Back Task
- Weighted Accuracy: 55%
- Match Accuracy: 20%
- Mismatch Accuracy: 80%
- Match Weight: 20%
- Mismatch Weight: 80%

#### Spatial Cueing Task
- Accuracy: 55%
- Omission Rate: 50%

#### Spatial Task Switching
- Accuracy: 55%
- Omission Rate: 50%

#### Stroop Task
- Accuracy: 55%
- Omission Rate: 50%

#### Visual Search Task
- Accuracy: 55%
- Omission Rate: 50%

## Directory Structure
```
.
├── preprocess.py
├── analyze_behavioral_data.py
├── thresholds_config.py
├── preprocessed_data/
│   └── <subject_folder>/
│       └── <task_files>.csv
└── outputs/
    └── <subject_folder>/
        ├── behavioral_summary.csv
        └── <subject_folder>_flags.csv
```

## Notes
- All thresholds can be modified in `thresholds_config.py`
- The scripts automatically create necessary directories
- Quality control flags are determined by comparing metrics against thresholds
- Empty flag files are created when no quality issues are found 