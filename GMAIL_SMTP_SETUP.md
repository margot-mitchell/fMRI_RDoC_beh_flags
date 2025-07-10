# Gmail SMTP Setup Guide

This guide explains how to set up Gmail SMTP for email notifications in the fMRI Behavioral QC Pipeline.

## Overview

We're using Gmail SMTP for reliable email notifications. This approach is simpler than Gmail API and doesn't require Google Cloud project creation.

## Setup Steps

### Option 1: App Password (Recommended - More Secure)

#### 1. Enable 2-Step Verification

1. Go to your [Google Account settings](https://myaccount.google.com/)
2. Click on **Security** in the left sidebar
3. Find **2-Step Verification** and click **Get started**
4. Follow the prompts to enable 2-Step Verification
5. You'll need your phone to complete this step

#### 2. Generate App Password

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

#### 3. Configure GitHub Secrets

Add these secrets:

1. **GMAIL_USERNAME**
   - Value: Your Gmail address (e.g., `your-email@gmail.com`)

2. **GMAIL_PASSWORD**
   - Value: The 16-character app password you generated

3. **EMAIL_RECIPIENTS**
   - Value: Comma-separated list of recipient email addresses

### Option 2: Regular Password (Fallback - Less Secure)

If app passwords are not available (e.g., work/school account restrictions):

#### 1. Configure GitHub Secrets

Add these secrets:

1. **GMAIL_USERNAME**
   - Value: Your Gmail address (e.g., `your-email@gmail.com`)

2. **GMAIL_PASSWORD**
   - Value: Your regular Gmail password

3. **EMAIL_RECIPIENTS**
   - Value: Comma-separated list of recipient email addresses

**Note**: Using your regular password is less secure and may not work if 2-Step Verification is enabled.

### 4. Test the Setup

1. Manually trigger the workflow in GitHub Actions
2. Check the logs in the "Send email notification" step
3. Verify emails are received

## Troubleshooting

### Common Issues

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

### Debugging Steps

1. Check GitHub Actions logs for detailed error messages
2. Verify all secrets are set correctly in GitHub repository settings
3. Test with a simple email client first
4. Regenerate the app password if needed

## Migration from Outlook SMTP

### What Changed

- **Server**: From Outlook SMTP to Gmail SMTP
- **Authentication**: From regular password to app password (or regular password as fallback)
- **Reliability**: Gmail SMTP is more reliable than Outlook SMTP

### Benefits

- More reliable email delivery
- Better error reporting
- No connection timeout issues
- Simpler setup than Gmail API
- No need for Google Cloud project creation

## Security Notes

- **App passwords** are specific to this application and more secure
- **Regular passwords** are less secure but work as fallback
- You can revoke app passwords anytime from your Google Account settings
- Never share your passwords
- Store them securely in GitHub Secrets
- You can generate multiple app passwords for different applications

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your 2-Step Verification is enabled (for app passwords)
3. Regenerate the app password if needed
4. Check GitHub Actions workflow logs
5. Test with a simple email first before running the full pipeline

## Alternative: If You Still Want Gmail API

If you later get access to create Google Cloud projects, you can switch to Gmail API by:

1. Following the instructions in `GMAIL_API_SETUP.md`
2. Updating the workflow to use Gmail API instead of SMTP
3. The Gmail API provides even better reliability and features 