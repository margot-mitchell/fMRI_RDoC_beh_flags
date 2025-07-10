#!/bin/bash

# Debug: show the structure of compiled-results
echo "Debug: Contents of compiled-results directory:"
find compiled-results -type d | head -20

# Calculate flag count and breakdown by subject/session
FLAG_COUNT=$(find compiled-results -path "*/flags/*" -type f | wc -l)
echo "Debug: Total flag count: $FLAG_COUNT"

# Create detailed flag breakdown
FLAG_BREAKDOWN=""
if [ $FLAG_COUNT -gt 0 ]; then
  echo "Debug: Creating flag breakdown..."
  
  # Look for flags in the actual structure - each artifact contains one subject-session
  for artifact_dir in compiled-results/*_results/; do
    if [ -d "$artifact_dir" ]; then
      artifact_name=$(basename "$artifact_dir")
      echo "Debug: Checking artifact: $artifact_name"
      
      # Extract subject and session from artifact name (format: sub-s4-ses-2_results)
      subject_session=$(echo "$artifact_name" | sed 's/_results$//')
      echo "Debug: Subject-session from artifact: $subject_session"
      
      # Look for flags in the results structure
      if [ -d "$artifact_dir/results" ]; then
        echo "Debug: Found results directory in $artifact_name"
        
        # Find the flags directory
        flags_dir=$(find "$artifact_dir/results" -name "flags" -type d | head -1)
        if [ -n "$flags_dir" ]; then
          session_flag_count=$(find "$flags_dir" -type f | wc -l)
          echo "Debug: Found $session_flag_count flags in $subject_session"
          if [ $session_flag_count -gt 0 ]; then
            if [ -z "$FLAG_BREAKDOWN" ]; then
              FLAG_BREAKDOWN="$subject_session: $session_flag_count flags"
            else
              FLAG_BREAKDOWN="$FLAG_BREAKDOWN
$subject_session: $session_flag_count flags"
            fi
          fi
        else
          echo "Debug: No flags directory found in $artifact_name"
        fi
      else
        echo "Debug: No results directory found in $artifact_name"
      fi
    fi
  done
fi

echo "Debug: Final flag breakdown: $FLAG_BREAKDOWN"

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