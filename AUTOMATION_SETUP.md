# Automated Weekly Processing Setup

This document explains how to configure the automated weekly fMRI behavioral QC processing workflow that runs every Sunday at 5:00 PM UTC.

## Overview

The automated workflow (`weekly_automated_processing.yml`) performs the following tasks:

1. **Detects new data**: Scans the Dropbox `/output/raw` folder for data modified in the past week
2. **Processes new data**: Runs the full pipeline (preprocessing, metrics calculation, flagging) on new subjects/sessions
3. **Compiles results**: Creates a summary report and compiles all artifacts
4. **Sends email notifications**: Automatically emails results to specified recipients

## Required GitHub Secrets

You need to configure the following secrets in your GitHub repository:

### 1. RCLONE_CONFIG (Already configured)
- Base64-encoded rclone configuration for Dropbox access
- Should already be set up from your manual workflow

### 2. Email Configuration Secrets

Add these secrets in your GitHub repository settings:

#### SMTP_SERVER
- Your SMTP server address (e.g., `smtp.gmail.com`, `smtp.office365.com`)

#### SMTP_PORT
- SMTP server port (usually `587` for TLS or `465` for SSL)

#### SMTP_USERNAME
- Your email address used for sending notifications

#### SMTP_PASSWORD
- Your email password or app-specific password
- **Note**: For Gmail, you may need to use an "App Password" instead of your regular password

#### EMAIL_RECIPIENTS
- Comma-separated list of email addresses to receive notifications
- Example: `researcher1@university.edu,researcher2@university.edu`

## Setting Up Email Notifications

### For Gmail:
1. Enable 2-factor authentication on your Google account
2. Generate an "App Password":
   - Go to Google Account settings
   - Security → 2-Step Verification → App passwords
   - Generate a password for "Mail"
3. Use the generated password as `SMTP_PASSWORD`
4. Set `SMTP_SERVER=smtp.gmail.com` and `SMTP_PORT=587`

### For Office 365:
1. Set `SMTP_SERVER=smtp.office365.com`
2. Set `SMTP_PORT=587`
3. Use your Office 365 email and password

### For Other Providers:
- Check your email provider's SMTP settings
- Common ports: 587 (TLS), 465 (SSL), 25 (unencrypted)

## Workflow Schedule

The workflow runs automatically:
- **When**: Every Sunday at 5:00 PM UTC
- **What**: Scans for data modified in the past 7 days
- **Manual trigger**: Can also be run manually via GitHub Actions

## What the Workflow Does

### 1. Data Detection
- Syncs with Dropbox to check for new data
- Identifies subjects and sessions with files modified in the past week
- Only processes data that has actually changed

### 2. Processing
- Runs the complete pipeline on each new subject/session
- Preprocessing → Metrics calculation → Flagging
- Tests metrics completeness
- Organizes results into artifacts

### 3. Results Compilation
- Downloads all artifacts from parallel processing
- Creates a summary report with:
  - List of processed subjects
  - Total number of artifacts
  - Count of quality flags found
- Uploads compiled results as a single artifact

### 4. Email Notification
- Sends a summary email to all recipients
- Includes:
  - Processing summary
  - Number of subjects/sessions processed
  - Quality flag count
  - Links to GitHub Actions run and artifacts

## Email Content

The automated email includes:
- Date and workflow run ID
- Summary of processed subjects and sessions
- Count of quality control flags found
- Links to detailed results in GitHub Actions
- Repository information

## Monitoring and Troubleshooting

### Check Workflow Status
1. Go to your GitHub repository
2. Click "Actions" tab
3. Look for "Weekly Automated Data Processing" workflow
4. Check run history and logs

### Common Issues

#### No new data detected
- The workflow will skip processing if no data was modified in the past week
- Check the "detect-new-data" job logs to see what was scanned

#### Email delivery issues
- Verify SMTP settings and credentials
- Check if your email provider requires app-specific passwords
- Test with a simple email first

#### Processing failures
- Check individual job logs for specific errors
- Common issues: missing dependencies, data format problems, network issues

## Customization

### Change Schedule
Edit the cron expression in the workflow:
```yaml
schedule:
  - cron: '0 17 * * 0'  # Sunday 5:00 PM UTC
```

### Modify Email Content
Edit the `body` section in the `send-email-notification` job to customize the email message.

### Add More Recipients
Update the `EMAIL_RECIPIENTS` secret with additional email addresses (comma-separated).

## Security Notes

- All secrets are encrypted and only accessible to repository administrators
- Email credentials are stored securely in GitHub Secrets
- Consider using app-specific passwords for email accounts
- Regularly rotate email passwords for security

## Testing

Before relying on the automated workflow:
1. Test the manual workflow to ensure it works correctly
2. Test email notifications with a small dataset
3. Verify all secrets are configured correctly
4. Check that the schedule works for your timezone (UTC conversion) 