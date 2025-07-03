#!/usr/bin/env python3
"""
Script to discover available sessions for a given subject.
"""

import os
import sys
import json
from pathlib import Path


def discover_sessions(subject_id):
    """
    Discover available sessions for a given subject.
    
    Args:
        subject_id (str): Subject ID (e.g., 'sub-sM')
        
    Returns:
        list: List of available sessions
    """
    raw_dir = Path('output/raw')
    subject_dir = raw_dir / subject_id
    
    if not subject_dir.exists():
        print(f"Subject directory {subject_dir} does not exist")
        return []
    
    sessions = set()
    
    # Look for session directories
    for item in subject_dir.iterdir():
        if item.is_dir() and item.name.startswith('ses-'):
            sessions.add(item.name)
    
    # Look for session files directly in subject directory
    for item in subject_dir.iterdir():
        if item.is_file() and item.suffix == '.json':
            # Extract session from filename
            parts = item.stem.split('_')
            for part in parts:
                if part.startswith('ses-'):
                    sessions.add(part)
                    break
    
    return sorted(list(sessions))


def main():
    if len(sys.argv) != 2:
        print("Usage: python discover_sessions.py <subject_id>")
        print("Example: python discover_sessions.py sub-sM")
        sys.exit(1)
    
    subject_id = sys.argv[1]
    sessions = discover_sessions(subject_id)
    
    if sessions:
        print(f"Available sessions for {subject_id}:")
        for session in sessions:
            print(f"  - {session}")
    else:
        print(f"No sessions found for {subject_id}")


if __name__ == '__main__':
    main() 