#!/usr/bin/env python3
"""
Test script for the automated workflow logic.
This simulates the key components of the weekly automation workflow.
"""

import os
import sys
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

def test_date_detection():
    """Test the date range logic for detecting new data."""
    print("=== Testing Date Detection Logic ===")
    
    # Simulate the date calculation from the workflow
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"End date: {end_date.strftime('%Y-%m-%d')}")
    print(f"Start date: {start_date.strftime('%Y-%m-%d')}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Test the find command logic (if we have test data)
    if os.path.exists('output/raw'):
        print("\nTesting file detection logic...")
        for subject_dir in Path('output/raw').glob('sub-*'):
            if subject_dir.is_dir():
                print(f"Found subject: {subject_dir.name}")
                for session_dir in subject_dir.glob('ses-*'):
                    if session_dir.is_dir():
                        print(f"  Found session: {session_dir.name}")
    
    return True

def test_matrix_generation():
    """Test the matrix generation logic for GitHub Actions."""
    print("\n=== Testing Matrix Generation Logic ===")
    
    # Simulate the subjects and sessions that would be found
    test_subjects = ['sub-s1', 'sub-s2', 'sub-s3']
    test_sessions = ['sub-s1/ses-1', 'sub-s1/ses-2', 'sub-s2/ses-1']
    
    # Convert to JSON format (like the workflow does)
    subjects_json = json.dumps(test_subjects)
    sessions_json = json.dumps(test_sessions)
    
    print(f"Subjects JSON: {subjects_json}")
    print(f"Sessions JSON: {sessions_json}")
    
    # Test parsing logic
    print("\nTesting session parsing logic:")
    for session_str in test_sessions:
        subject, session = session_str.split('/')
        print(f"  '{session_str}' -> subject: '{subject}', session: '{session}'")
    
    return True

def test_artifact_naming():
    """Test the artifact naming logic."""
    print("\n=== Testing Artifact Naming Logic ===")
    
    test_cases = [
        ('sub-s1', 'ses-1'),
        ('sub-s2', 'ses-2'),
        ('sub-s3', 'ses-pretouch'),
        ('sub-s4', 'ses-1 '),  # Test with trailing space
        ('sub-s5', ' ses-2'),  # Test with leading space
    ]
    
    for subject, session in test_cases:
        # Simulate the cleaning logic from the workflow
        clean_session = session.strip().replace(' ', '').replace('-', '_')
        artifact_name = f"{subject}-{clean_session}_results"
        artifact_name = artifact_name.replace(' ', '').replace('__', '_')
        
        print(f"  Subject: '{subject}', Session: '{session}' -> Artifact: '{artifact_name}'")
    
    return True

def test_email_content():
    """Test the email content generation."""
    print("\n=== Testing Email Content Generation ===")
    
    # Simulate workflow outputs
    workflow_run_id = "123456789"
    new_subjects = ['sub-s1', 'sub-s2']
    new_sessions = ['sub-s1/ses-1', 'sub-s1/ses-2', 'sub-s2/ses-1']
    flag_count = 3
    
    email_body = f"""
Weekly fMRI Behavioral QC Processing Report

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Workflow Run: {workflow_run_id}

Processing Summary:
- Subjects processed: {new_subjects}
- Sessions processed: {new_sessions}
- Quality flags found: {flag_count}

The processing has completed successfully. Please check the GitHub Actions artifacts for detailed results.

Repository: test-repo
Run URL: https://github.com/test-repo/actions/runs/{workflow_run_id}

This is an automated message from the fMRI Behavioral QC Pipeline.
"""
    
    print("Generated email content:")
    print(email_body)
    
    return True

def test_workflow_structure():
    """Test that the workflow file is valid YAML."""
    print("\n=== Testing Workflow File Structure ===")
    
    workflow_file = '.github/workflows/weekly_automated_processing.yml'
    
    if not os.path.exists(workflow_file):
        print(f"ERROR: Workflow file {workflow_file} not found!")
        return False
    
    try:
        import yaml
        with open(workflow_file, 'r') as f:
            workflow_data = yaml.safe_load(f)
        
        print("✓ Workflow file is valid YAML")
        
        # Check required sections
        required_keys = ['name', 'on', 'jobs']
        for key in required_keys:
            if key in workflow_data:
                print(f"✓ Contains '{key}' section")
            else:
                print(f"✗ Missing '{key}' section")
                return False
        
        # Check jobs
        jobs = workflow_data.get('jobs', {})
        required_jobs = ['detect-new-data', 'process-new-data', 'compile-results', 'send-email-notification']
        for job in required_jobs:
            if job in jobs:
                print(f"✓ Contains '{job}' job")
            else:
                print(f"✗ Missing '{job}' job")
                return False
        
        print("✓ All required workflow components present")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to parse workflow file: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Automated Workflow Logic")
    print("=" * 40)
    
    tests = [
        test_date_detection,
        test_matrix_generation,
        test_artifact_naming,
        test_email_content,
        test_workflow_structure,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"ERROR in {test.__name__}: {e}")
            results.append((test.__name__, False))
    
    print("\n" + "=" * 40)
    print("TEST RESULTS:")
    print("=" * 40)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ ALL TESTS PASSED - Workflow logic appears correct")
        print("\nNext steps:")
        print("1. Configure GitHub secrets (see AUTOMATION_SETUP.md)")
        print("2. Push to GitHub and test manually")
        print("3. Monitor the first automated run")
    else:
        print("✗ SOME TESTS FAILED - Please fix issues before pushing")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 