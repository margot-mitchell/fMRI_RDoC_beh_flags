# fMRI RDoC Behavioral Quality Control

This project provides tools for preprocessing, analyzing, and QCing behavioral data from fMRI RDoC sessions. It calculates task metrics and generates quality control flags for perfomance that does not meet thresholds (defined in thresholds_config.py).

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
   - `GMAIL_PASSWORD`: Gmail app password (not your regular password)
   - `EMAIL_RECIPIENTS`: Comma-separated list of recipient email addresses

#### Gmail SMTP Setup

The pipeline uses Gmail SMTP for reliable email notifications.
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
│   ├── <task>/
│   │   └── <subject>_<session>_task-<task>_run-1.parquet
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
│       └── <session>/
│           ├── <task>_metrics.csv
│           └── ...
└── flags/
    └── <subject>/
        └── <session>/
            ├── <task>_flags.csv
            └── ...

preprocessed_data/
└── <subject>/
    ├── <task>/
    │   └── <subject>_<session>_task-<task>_run-1.parquet
    └── ...
```

The test runs automatically after flag generation and will fail the workflow if any issues are detected.


## Important Notes

### Accuracy Calculations
- **Go/NoGo `nogo_accuracy`**: Calculated as the proportion of nogo trials with no response (successful inhibition)
- **Stop Signal `stop_accuracy`**: Calculated as the proportion of stop trials with no response (successful inhibition)
- **All other accuracy rates**: Calculated as the proportion of correct test trials out of trials with a response

### Data Organization
- Raw data files are not tracked in git (added to `.gitignore`)
- Workflow syncs fresh data from Dropbox for each run
- Results are organized in clean, navigable artifact structures

### Quality Control
- Flags are generated when metrics fall outside acceptable ranges (defined in thresholds_config.py)
- Empty flag files indicate no quality issues detected

## Repository Structure
```

├── .github/workflows/
│   ├── process_data.yml                    # Manual processing workflow
│   └── weekly_automated_processing.yml     # Automated weekly workflow
├── src/
│   ├── preprocess.py                       # Data preprocessing script
│   ├── calculate_metrics.py                # Metrics calculation script
│   └── generate_flags.py                   # Quality control flag generation
├── rdoc_package/
│   ├── __init__.py
│   ├── config.py                           # Configuration settings
│   ├── preprocess/                         # Preprocessing modules
│   ├── analyze/
│   │   ├── __init__.py
│   │   ├── preprocess.py                   # Analysis preprocessing
│   │   ├── calculate_metrics.py            # Analysis metrics calculation
│   │   ├── generate_flags.py               # Analysis flag generation
│   │   └── tasks/
│   │       ├── __init__.py
│   │       ├── tasks.py                    # Task analysis functions
│   │       └── utils.py                    # Utility functions
│   └── utils/                              # Utility modules
├── preprocessed_data/                       # Preprocessed data storage
├── results/                                 # Results storage
│   ├── metrics/                            # Calculated metrics
│   └── flags/                              # Quality control flags
├── raw_data/                               # Raw data storage
├── logs/                                   # Log files
├── thresholds_config.py                    # Quality control thresholds
├── requirements.txt                        # Python dependencies
├── pyproject.toml                         # Project configuration
├── AUTOMATION_SETUP.md                     # Automated workflow setup guide
└── README.md                               # This file
```
