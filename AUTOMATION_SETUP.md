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

#### GMAIL_USERNAME
- Your Gmail address (e.g., `your-email@gmail.com`)

#### GMAIL_PASSWORD
- Your Gmail app password (recommended) or regular password
- **Note**: For better security, use an "App Password" (see detailed instructions below)

#### EMAIL_RECIPIENTS
- Comma-separated list of email addresses to receive notifications
- Example: `researcher1@university.edu,researcher2@university.edu`

## Setting Up Email Notifications

### Gmail Setup (Recommended)

We use Gmail SMTP for reliable email notifications. This approach is simpler than Gmail API and doesn't require Google Cloud project creation.

#### Option 1: App Password (Recommended - More Secure)

##### 1. Enable 2-Step Verification

1. Go to your [Google Account settings](https://myaccount.google.com/)
2. Click on **Security** in the left sidebar
3. Find **2-Step Verification** and click **Get started**
4. Follow the prompts to enable 2-Step Verification
5. You'll need your phone to complete this step

##### 2. Generate App Password

1. Go to your [Google Account settings](https://myaccount.google.com/)
2. Click on **Security** in the left sidebar
3. Under **2-Step Verification**, click **App passwords**
4. Click **Generate** next to "Mail"
5. Copy the 16-character password that appears
6. **Important**: Save this password securely - you won't be able to see it again

**Note**: If you don't see "App passwords" after enabling 2-Step Verification, try:
- Waiting 10-15 minutes
- Going directly to: https://myaccount.google.com/apppasswords
- Checking if your account type allows app passwords

##### 3. Configure GitHub Secrets

Add these secrets:

1. **GMAIL_USERNAME**
   - Value: Your Gmail address (e.g., `your-email@gmail.com`)

2. **GMAIL_PASSWORD**
   - Value: The 16-character app password you generated

3. **EMAIL_RECIPIENTS**
   - Value: Comma-separated list of recipient email addresses

#### Option 2: Regular Password (Fallback - Less Secure)

If app passwords are not available (e.g., work/school account restrictions):

##### 1. Configure GitHub Secrets

Add these secrets:

1. **GMAIL_USERNAME**
   - Value: Your Gmail address (e.g., `your-email@gmail.com`)

2. **GMAIL_PASSWORD**
   - Value: Your regular Gmail password

3. **EMAIL_RECIPIENTS**
   - Value: Comma-separated list of recipient email addresses

**Note**: Using your regular password is less secure and may not work if 2-Step Verification is enabled.

### Testing the Setup

1. Manually trigger the workflow in GitHub Actions
2. Check the logs in the "Send email notification" step
3. Verify emails are received

## Workflow Schedule

The workflow runs automatically:
- **When**: Every Sunday at 5:00 PM UTC
- **What**: Scans for data modified in the past 7 days
- **Manual trigger**: Can also be run manually via GitHub Actions

## What the Workflow Does

### 1. Data Detection
- Syncs with Dropbox to check for new data
- Identifies subjects and sessions with files modified in the past week
- **Excludes prescan sessions** (sessions containing "prescan" in the name)
- Only processes data that has actually changed

### 2. Processing
- Runs the complete pipeline on each new subject/session
- Preprocessing → Metrics calculation → Flagging
- Tests metrics completeness
- Organizes results into artifacts with flatter structure

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
  - Processing period and workflow run ID
  - Summary of processed subjects and sessions
  - Quality flag breakdown by subject-session
  - Links to GitHub Actions run and artifacts

## Email Content

The automated email includes:
- Processing period (date range)
- Workflow run ID
- Clean list of processed subjects and sessions (no JSON formatting)
- Quality flag breakdown with each subject-session on its own line
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

**Common Gmail Issues:**

1. **"Invalid credentials"**
   - Make sure you're using the correct password (app password or regular password)
   - Verify 2-Step Verification is enabled (for app passwords)
   - Regenerate the app password if needed

2. **"Authentication failed"**
   - Check that `GMAIL_USERNAME` is your complete Gmail address
   - Ensure `GMAIL_PASSWORD` is correct
   - Try regenerating the app password if using Option 1

3. **"Connection timeout"**
   - Gmail SMTP is very reliable, this shouldn't happen
   - Check your internet connection
   - Verify the secrets are set correctly

4. **"App passwords not showing up"**
   - Wait 10-15 minutes after enabling 2-Step Verification
   - Try the direct link: https://myaccount.google.com/apppasswords
   - Check if your account type allows app passwords
   - Use Option 2 (regular password) as fallback

**Debugging Steps:**
1. Check GitHub Actions logs for detailed error messages
2. Verify all secrets are set correctly in GitHub repository settings
3. Test with a simple email client first
4. Regenerate the app password if needed

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
The email content is generated by `generate_email_body.sh` script. Edit this script to customize the email message.

### Add More Recipients
Update the `EMAIL_RECIPIENTS` secret with additional email addresses (comma-separated).

## Security Notes

- All secrets are encrypted and only accessible to repository administrators
- Email credentials are stored securely in GitHub Secrets
- **App passwords** are specific to this application and more secure
- **Regular passwords** are less secure but work as fallback
- You can revoke app passwords anytime from your Google Account settings
- Never share your passwords
- Store them securely in GitHub Secrets
- You can generate multiple app passwords for different applications
- Regularly rotate email passwords for security

## Testing

Before relying on the automated workflow:
1. Test the manual workflow to ensure it works correctly
2. Test email notifications with a small dataset
3. Verify all secrets are configured correctly
4. Check that the schedule works for your timezone (UTC conversion)

## Artifact Structure

The workflow creates artifacts with a clean, flat structure:
```
sub-s4-ses-3_results/
├── metrics/
├── flags/
└── preprocessed_data/
```

This makes it easy to download and navigate the results.

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your 2-Step Verification is enabled (for app passwords)
3. Regenerate the app password if needed
4. Check GitHub Actions workflow logs
5. Test with a simple email first before running the full pipeline 