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
pip install -r requirements.txt
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
- Download data from Dropbox (if configured)
- Read JSON files from `raw_data/<subject_id>/`
- Check that JSON files contain the expected columns for the task name of the file
- Process the data into parquet format
- Save preprocessed files in `preprocessed_data/<subject_id>/<task_name>/`

### 2. Metrics Calculation
Run the metrics calculation script:
```bash
python calculate_metrics.py <subject_id>
```
This will:
- Read preprocessed parquet files
- Calculate summary statistics and metrics for each task
- Save metrics in `outputs/<subject_id>/metrics/<task>_metrics.csv`

### 3. Flag Generation
Run the flag generation script:
```bash
python generate_flags.py <subject_id>
```
This will:
- Read metrics files from `outputs/<subject_id>/metrics/`
- Compare metrics to thresholds
- Generate `flags.csv` and `all_metrics_checked.csv` in `outputs/<subject_id>/`

## Outputs

- `outputs/<subject_id>/metrics/<task>_metrics.csv`: Task-specific metrics
- `outputs/<subject_id>/flags.csv`: Quality control flags (if any)
- `outputs/<subject_id>/all_metrics_checked.csv`: All metrics and their thresholds

## Quality Control Thresholds & Metrics
All thresholds are defined in `thresholds_config.py` and mapped in `generate_flags.py`.

### Stop Signal Task
- `go_accuracy`: min: 0.55
- `go_rt`: max: 750
- `go_omission_rate`: max: 0.5
- `stop_accuracy`: range: 0.25–0.75 (flag if <0.25 or >0.75)

### AX-CPT Task
- `AX_accuracy`, `BX_accuracy`, `AY_accuracy`, `BY_accuracy`: min: 0.55
- `AX_omission_rate`, `BX_omission_rate`, `AY_omission_rate`, `BY_omission_rate`: max: 0.5

### Go/NoGo Task
- `go_accuracy`: min: 0.857
- `nogo_accuracy`: min: 0.143
- `mean_accuracy`: min: 0.55
- `go_omission_rate`: max: 0.5

### Operation Span Task
- `8x8_grid_asymmetric_accuracy`: min: 0.55
- `8x8_grid_symmetric_accuracy`: min: 0.55
- `mean_4x4_grid_accuracy_irrespective_of_order`: min: 0.25
- `mean_4x4_grid_accuracy_respective_of_order`: max: 0.4 (order diff)

### Operation Only Span Task
- `8x8_asymmetric_accuracy`: min: 0.55
- `8x8_symmetric_accuracy`: min: 0.55

### Simple Span Task
- `mean_4x4_grid_accuracy_irrespective_of_order`: min: 0.55
- `mean_4x4_grid_accuracy_respective_of_order`: max: 0.2 (order diff)

### N-Back Task
- `weighted_2back_accuracy`: min: 0.55
- `weighted_1back_accuracy`: min: 0.55
- `match_2_accuracy`: min: 0.2
- `mismatch_2_accuracy`: min: 0.8
- `match_1_accuracy`: min: 0.2
- `mismatch_1_accuracy`: min: 0.8

### Cued Task Switching
- `task_stay_cue_stay_accuracy`, `task_stay_cue_switch_accuracy`, `task_switch_cue_switch_accuracy`: min: 0.55
- `task_stay_cue_stay_omission_rate`, `task_stay_cue_switch_omission_rate`, `task_switch_cue_switch_omission_rate`: max: 0.5
- `parity_accuracy`, `magnitude_accuracy`: min: 0.55

### Spatial Cueing Task
- `doublecue_accuracy`, `invalid_accuracy`, `nocue_accuracy`, `valid_accuracy`: min: 0.55
- `doublecue_omission_rate`, `invalid_omission_rate`, `nocue_omission_rate`, `valid_omission_rate`: max: 0.5

### Spatial Task Switching
- `task_stay_cue_switch_accuracy`, `task_switch_cue_stay_accuracy`, `task_switch_cue_switch_accuracy`, `color_accuracy`, `form_accuracy`: min: 0.55
- `task_stay_cue_switch_omission_rate`, `task_switch_cue_stay_omission_rate`, `task_switch_cue_switch_omission_rate`: max: 0.5

### Stroop Task
- `congruent_accuracy`, `incongruent_accuracy`: min: 0.55
- `congruent_omission_rate`, `incongruent_omission_rate`: max: 0.5

### Visual Search Task
- `conjunction_24_accuracy`, `conjunction_8_accuracy`, `feature_24_accuracy`, `feature_8_accuracy`: min: 0.55
- `conjunction_24_omission_rate`, `conjunction_8_omission_rate`, `feature_24_omission_rate`, `feature_8_omission_rate`: max: 0.5

### Flanker Task
- `congruent_accuracy`, `incongruent_accuracy`: min: 0.55
- `congruent_omission_rate`, `incongruent_omission_rate`: max: 0.5

## Directory Structure
```
.
├── preprocess.py
├── calculate_metrics.py
├── generate_flags.py
├── thresholds_config.py
├── preprocessed_data/
│   └── <subject_id>/
│       └── <task_name>/<parquet files>
├── raw_data/
│   └── <subject_id>/
│       └── <raw JSON files>
└── outputs/
    └── <subject_id>/
        ├── metrics/
        │   └── <task>_metrics.csv
        ├── flags.csv
        └── all_metrics_checked.csv
```

## Notes
- All thresholds can be modified in `thresholds_config.py`
- The scripts automatically create necessary directories
- Quality control flags are determined by comparing metrics against thresholds
- If no flags are found, `flags.csv` will be empty and a message will be printed
- The codebase supports both local and Dropbox-based workflows

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