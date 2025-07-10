# fMRI RDoC Behavioral Quality Control

This project provides tools for preprocessing and analyzing behavioral data from fMRI RDoC studies. It includes quality control checks for various cognitive tasks and generates summary statistics and quality control flags.

## Features

- **Automated Processing**: GitHub Actions workflow for batch processing
- **Session-Specific Processing**: Process individual sessions or all sessions for a subject
- **Selective Data Sync**: Only download specific session data from Dropbox
- **Quality Control Flags**: Automated flagging of problematic performance
- **Gmail SMTP Notifications**: Reliable email notifications using Gmail SMTP

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

### Dropbox Configuration (for automated workflow)
1. Configure rclone with your Dropbox credentials
2. Set up the `RCLONE_CONFIG` secret in your GitHub repository
3. Ensure your data is organized in Dropbox as: `rdoc_fmri_behavior/output/raw/<subject>/<session>/`

### GitHub Secrets Configuration

The automated workflows require several GitHub secrets to be configured in your repository:

#### Required Secrets

1. **RCLONE_CONFIG** (for Dropbox access)
   - Base64-encoded rclone configuration
   - Contains Dropbox authentication credentials

2. **Gmail SMTP Configuration** (for automated notifications)
   - `GMAIL_USERNAME`: Your Gmail address
   - `GMAIL_APP_PASSWORD`: Gmail app password (not your regular password)
   - `EMAIL_RECIPIENTS`: Comma-separated list of recipient email addresses

#### Gmail SMTP Setup

The pipeline uses Gmail SMTP for reliable email notifications. To set this up:

1. **Enable 2-Step Verification** (if not already enabled):
   - Go to your Google Account settings
   - Security → 2-Step Verification → Turn on

2. **Generate App Password**:
   - Go to your Google Account settings
   - Security → 2-Step Verification → App passwords
   - Select "Mail" and generate a password
   - Copy the generated 16-character password

3. **Configure GitHub Secrets**:
   - `GMAIL_USERNAME`: Your Gmail address (e.g., `your-email@gmail.com`)
   - `GMAIL_APP_PASSWORD`: The 16-character app password you generated
   - `EMAIL_RECIPIENTS`: Comma-separated list of recipient emails

#### Setting Up Secrets

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"** for each required secret
4. Enter the secret name and value
5. Click **"Add secret"**

For detailed setup instructions, see `AUTOMATION_SETUP.md`.

## Usage

### GitHub Actions Workflow (Recommended)

The project includes GitHub Actions workflows that can process data in several ways:

#### Automated Weekly Processing
- **Schedule**: Runs automatically every Sunday at 5:00 PM UTC
- **Function**: Detects new data from the past week and processes it automatically
- **Features**: 
  - Excludes prescan sessions from processing
  - Sends email notifications with processing summaries via Gmail SMTP
  - Compiles results into organized artifacts
- **Manual Trigger**: Can also be triggered manually via GitHub Actions

#### Manual Processing Options

**1. Process All Subjects**
- Set `process_all: true`
- Processes all available subjects in parallel

**2. Process Specific Subjects**
- Set `specific_subjects: "sub-s1,sub-s2,sub-s3"`
- Processes only the specified subjects

**3. Process Single Subject (All Sessions)**
- Set `subject_folder: "sub-s1"`
- Processes all sessions for the specified subject

**4. Process Single Subject (Specific Sessions)**
- Set `subject_folder: "sub-s1"` and `session_names: "ses-1,ses-2"`
- Processes only the specified sessions for the subject

### Local Processing

#### 1. Preprocessing
```bash
# Process all sessions for a subject
python preprocess.py <subject_folder>

# Process specific session
python preprocess.py <subject_folder> --session <session_name>
```

#### 2. Metrics Calculation
```bash
# Calculate metrics for all sessions
python calculate_metrics.py <subject_folder>

# Calculate metrics for specific session
python calculate_metrics.py <subject_folder> --session <session_name>
```

#### 3. Flag Generation
```bash
# Generate flags for all sessions
python generate_flags.py <subject_folder>

# Generate flags for specific session
python generate_flags.py <subject_folder> --session <session_name>
```

## Output Structure

### Artifact Organization
Results are organized in clean artifact structures:

```
<subject>_<session>_results/
├── preprocessed_data/
│   ├── <subject>_<session>_task-<task>_run-1.parquet
│   └── ...
└── results/
    ├── metrics/
    │   ├── <task>_metrics.csv
    │   └── ...
    └── flags/
        ├── <task>_flags.csv
        └── ...
```

### Local Output Structure
```
results/
├── metrics/
│   └── <subject>/
│       └── <task>_metrics.csv
└── flags/
    └── <subject>/
        └── <task>_flags.csv

preprocessed_data/
└── <subject>/
    └── <task>_<session>_run-1.parquet
```

## Quality Control & Testing

### Metrics Completeness Test
The workflow includes an automated test (`test_metrics_completeness.py`) that verifies:
- All expected tasks have metrics files
- All metrics files contain the expected metrics for each task
- All metrics are checked against appropriate thresholds
- No metrics are missing or have invalid values

The test runs automatically after flag generation and will fail the workflow if any issues are detected.

### Quality Control Thresholds & Metrics

All thresholds are defined in `thresholds_config.py` and mapped in `generate_flags.py`.

### Stop Signal Task
- `go_accuracy`: min: 0.75
- `go_rt`: max: 750
- `go_omission_rate`: max: 0.5
- `stop_accuracy`: range: 0.35–0.65 

### AX-CPT Task
- `AX_accuracy`, `BX_accuracy`, `AY_accuracy`, `BY_accuracy`: min: 0.55
- `AX_omission_rate`, `BX_omission_rate`, `AY_omission_rate`, `BY_omission_rate`: max: 0.5

### Go/NoGo Task
- `go_accuracy`: min: 0.75
- `nogo_accuracy`: min: 0.35 

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

## Important Notes

### Accuracy Calculations
- **Go/NoGo `nogo_accuracy`**: Calculated as the proportion of nogo trials with no response (successful inhibition)
- **Stop Signal `stop_accuracy`**: Calculated as the proportion of stop trials with no response (successful inhibition)
- **Accuracy rates**: Calculated as the proportion of correct test trials out of trials with a response
- **Omission rates**: Calculated as proportion of test trials with no response within each condition

### Data Organization
- Raw data files are not tracked in git (added to `.gitignore`)
- Workflow syncs fresh data from Dropbox for each run
- Results are organized in clean, navigable artifact structures

### Quality Control
- Flags are generated when metrics fall outside acceptable ranges
- Empty flag files indicate no quality issues detected
- The metrics completeness test ensures all metrics are calculated and checked against thresholds

## Repository Structure
```
.
├── .github/workflows/
│   ├── process_data.yml                    # Manual processing workflow
│   └── weekly_automated_processing.yml     # Automated weekly workflow
├── rdoc_package/
│   ├── analyze/
│   │   └── tasks/
│   │       ├── tasks.py                    # Task analysis functions
│   │       └── utils.py                    # Utility functions
│   │
│   └── ...
├── preprocess.py                           # Data preprocessing script
├── calculate_metrics.py                    # Metrics calculation script
├── generate_flags.py                       # Quality control flag generation
├── test_metrics_completeness.py           # Metrics completeness testing
├── test_automation_logic.py               # Local automation testing
├── thresholds_config.py                    # Quality control thresholds
├── requirements.txt                        # Python dependencies
├── AUTOMATION_SETUP.md                     # Automated workflow setup guide
└── README.md                               # This file
```

## Support

For issues or questions, please check the GitHub repository or contact the development team. 