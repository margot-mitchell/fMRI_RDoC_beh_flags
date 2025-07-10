#!/bin/bash

# Calculate flag count and breakdown by subject/session
FLAG_COUNT=$(find compiled-results -path "*/flags/*" -type f | wc -l)

# Create detailed flag breakdown
FLAG_BREAKDOWN=""
if [ $FLAG_COUNT -gt 0 ]; then
  for subject_dir in compiled-results/*_results/; do
    if [ -d "$subject_dir" ]; then
      subject_name=$(basename "$subject_dir" | sed 's/_results$//')
      if [ -d "$subject_dir/results" ]; then
        for session_dir in "$subject_dir/results"/*/; do
          if [ -d "$session_dir" ]; then
            session_name=$(basename "$session_dir")
            if [ -d "$session_dir/flags" ]; then
              session_flag_count=$(find "$session_dir/flags" -type f | wc -l)
              if [ $session_flag_count -gt 0 ]; then
                if [ -z "$FLAG_BREAKDOWN" ]; then
                  FLAG_BREAKDOWN="$subject_name/$session_name: $session_flag_count flags"
                else
                  FLAG_BREAKDOWN="$FLAG_BREAKDOWN
$subject_name/$session_name: $session_flag_count flags"
                fi
              fi
            fi
          fi
        done
      fi
    fi
  done
fi

# If no breakdown was created, use total count
if [ -z "$FLAG_BREAKDOWN" ]; then
  FLAG_BREAKDOWN="$FLAG_COUNT total flags"
fi

# Calculate date range (same logic as detect step)
if [ "$GITHUB_EVENT_NAME" = "workflow_dispatch" ]; then
  # Manual run: check last 7 days
  END_DATE=$(date +%Y-%m-%d)
  START_DATE=$(date -d "7 days ago" +%Y-%m-%d)
else
  # Scheduled run (Sunday): check from last Sunday to this Sunday
  END_DATE=$(date +%Y-%m-%d)
  START_DATE=$(date -d "last Sunday" +%Y-%m-%d)
fi

# Create email body
cat > email_body.txt << EOF
Weekly fMRI Behavioral QC Processing Report

Processing Period: $START_DATE to $END_DATE
Workflow Run: $GITHUB_RUN_ID

Subjects Processed:
$SUBJECTS_PROCESSED

Sessions Processed:
$SESSIONS_PROCESSED

Quality flags found: $FLAG_BREAKDOWN

📁 DOWNLOAD RESULTS:
• Weekly Compiled Results and Individual Session Results can be found in the workflow artifacts
• Go to the Run URL below and scroll down to the "Artifacts" section

Repository: $GITHUB_REPOSITORY
Run URL: $GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID

---
This is an automated message from the fMRI Behavioral QC Pipeline.
Sent via GitHub Actions workflow.
EOF 