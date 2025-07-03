#!/usr/bin/env python3
"""
Test script to verify that all metrics are calculated successfully for every task
and mapped against their appropriate thresholds.

This script checks:
1. All expected tasks have metrics files
2. All metrics files contain the expected metrics for each task
3. All metrics are checked against appropriate thresholds
4. No metrics are missing or have invalid values
"""

import os
import sys
import logging
import polars as pl
import pandas as pd
from typing import Dict, List, Set, Tuple
from generate_flags import TASK_NAME_MAP, THRESHOLDS, get_all_metrics_and_thresholds

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def get_expected_metrics_for_task(task_name: str) -> Set[str]:
    """Get the set of expected metrics for a given task."""
    expected_metrics = set()
    
    # Get metrics from get_all_metrics_and_thresholds
    metrics_thresholds = get_all_metrics_and_thresholds(task_name)
    for metric_name, _ in metrics_thresholds:
        if metric_name != 'order_difference':  # Skip calculated metrics
            expected_metrics.add(metric_name)
    
    # Add RT metrics that might be dynamically found
    if task_name == 'stop_signal':
        expected_metrics.update(['stop_signal_go_rt'])
    elif task_name == 'gonogo':
        expected_metrics.update(['gonogo_go_rt'])
    elif task_name == 'ax_cpt':
        expected_metrics.update(['AX_rt', 'BX_rt', 'AY_rt', 'BY_rt'])
    elif task_name == 'nback':
        expected_metrics.update(['match_1_rt', 'match_2_rt', 'mismatch_1_rt', 'mismatch_2_rt'])
    elif task_name == 'flanker':
        expected_metrics.update(['congruent_rt', 'incongruent_rt'])
    elif task_name == 'stroop':
        expected_metrics.update(['congruent_rt', 'incongruent_rt'])
    elif task_name == 'visual_search':
        expected_metrics.update(['conjunction_24_rt', 'conjunction_8_rt', 'feature_24_rt', 'feature_8_rt'])
    elif task_name == 'cued_ts':
        expected_metrics.update(['task_stay_cue_stay_rt', 'task_stay_cue_switch_rt', 'task_switch_cue_switch_rt'])
    elif task_name == 'spatial_cueing':
        expected_metrics.update(['doublecue_rt', 'invalid_rt', 'nocue_rt', 'valid_rt'])
    elif task_name == 'spatial_ts':
        expected_metrics.update(['task_stay_cue_switch_rt', 'task_switch_cue_stay_rt', 'task_switch_cue_switch_rt'])
    
    # Add proportion_feedback for all tasks
    expected_metrics.add('proportion_feedback')
    
    return expected_metrics

def get_calculated_metrics_for_task(task_name: str) -> Set[str]:
    """Get the set of calculated metrics for a given task."""
    calculated_metrics = set()
    
    if task_name == 'gonogo':
        calculated_metrics.add('mean_accuracy')
    elif task_name == 'nback':
        calculated_metrics.update(['weighted_2back_accuracy', 'weighted_1back_accuracy'])
    elif task_name in ['operation_span', 'simple_span']:
        calculated_metrics.add('order_difference')
    
    return calculated_metrics

def test_metrics_completeness(subject_folder: str = None, session: str = None) -> Dict[str, List[str]]:
    """
    Test that all metrics are calculated and checked against thresholds.
    
    Args:
        subject_folder: Specific subject to test (e.g., 'sub-SK')
        session: Specific session to test (e.g., 'ses-1')
    
    Returns:
        Dictionary with test results and any issues found
    """
    issues = {
        'missing_tasks': [],
        'missing_metrics': [],
        'invalid_values': [],
        'unchecked_metrics': [],
        'task_mapping_issues': []
    }
    
    # Determine which subjects to test
    if subject_folder:
        subject_folders = [subject_folder]
    else:
        metrics_dir = os.path.join('results', 'metrics')
        if not os.path.exists(metrics_dir):
            logger.error(f"Metrics directory {metrics_dir} does not exist")
            return issues
        
        subject_folders = [f for f in os.listdir(metrics_dir)
                          if os.path.isdir(os.path.join(metrics_dir, f)) and f.startswith('sub-')]
    
    logger.info(f"Testing {len(subject_folders)} subjects: {', '.join(subject_folders)}")
    
    for subject in subject_folders:
        logger.info(f"\nTesting subject: {subject}")
        
        metrics_dir = os.path.join('results', 'metrics', subject)
        if not os.path.exists(metrics_dir):
            logger.error(f"Subject metrics directory {metrics_dir} does not exist")
            continue
        
        # Get all task directories
        task_dirs = [d for d in os.listdir(metrics_dir)
                    if os.path.isdir(os.path.join(metrics_dir, d))]
        
        for task_dir in task_dirs:
            logger.info(f"  Testing task: {task_dir}")
            
            # Check task name mapping
            task_name = TASK_NAME_MAP.get(task_dir)
            if task_name is None:
                issues['task_mapping_issues'].append(f"{subject}/{task_dir}: No mapping found in TASK_NAME_MAP")
                logger.warning(f"    No task mapping found for {task_dir}")
                continue
            
            # Get expected metrics for this task
            expected_metrics = get_expected_metrics_for_task(task_name)
            calculated_metrics = get_calculated_metrics_for_task(task_name)
            
            # Get all CSV files for this task
            task_path = os.path.join(metrics_dir, task_dir)
            csv_files = [f for f in os.listdir(task_path) if f.endswith('.csv')]
            
            if not csv_files:
                issues['missing_tasks'].append(f"{subject}/{task_dir}: No metrics files found")
                logger.warning(f"    No metrics files found")
                continue
            
            # Filter files by session if specified
            if session:
                session_files = [f for f in csv_files if session in f]
                if not session_files:
                    logger.warning(f"    No files found for session {session}")
                    continue
                csv_files = session_files
            
            # Process each metrics file
            for metrics_file in csv_files:
                logger.info(f"    Testing file: {metrics_file}")
                
                metrics_path = os.path.join(task_path, metrics_file)
                try:
                    task_metrics_df = pl.read_csv(metrics_path)
                except Exception as e:
                    issues['invalid_values'].append(f"{subject}/{task_dir}/{metrics_file}: Failed to read CSV - {e}")
                    logger.error(f"      Failed to read CSV: {e}")
                    continue
                
                # Get actual metrics from the file
                actual_metrics = set(task_metrics_df['metric'].to_list())
                
                # Check for missing expected metrics
                missing_metrics = expected_metrics - actual_metrics
                if missing_metrics:
                    issues['missing_metrics'].append(
                        f"{subject}/{task_dir}/{metrics_file}: Missing metrics - {', '.join(sorted(missing_metrics))}"
                    )
                    logger.warning(f"      Missing metrics: {', '.join(sorted(missing_metrics))}")
                
                # Check for invalid values (NaN, None, etc.)
                for metric in actual_metrics:
                    filtered_df = task_metrics_df.filter(pl.col('metric') == metric)
                    if len(filtered_df) > 0:
                        value = filtered_df['value'].item()
                        if pd.isna(value) or value is None:
                            issues['invalid_values'].append(
                                f"{subject}/{task_dir}/{metrics_file}: Invalid value for {metric} - {value}"
                            )
                            logger.warning(f"      Invalid value for {metric}: {value}")
                
                # Check that all metrics have appropriate thresholds
                for metric in actual_metrics:
                    # Skip calculated metrics as they don't have direct thresholds
                    if metric in calculated_metrics:
                        continue
                    
                    # Check if metric has a threshold defined
                    threshold_found = False
                    
                    # Check METRIC_TO_THRESHOLD mapping
                    if metric in METRIC_TO_THRESHOLD:
                        threshold_found = True
                    else:
                        # Check if it's an RT metric
                        if 'rt' in metric.lower():
                            threshold_found = True  # Uses general RT_THRESHOLD
                        else:
                            # Check if threshold exists in THRESHOLDS
                            metric_upper = metric.upper()
                            task_upper = task_name.upper()
                            threshold_var = f'{task_upper}_{metric_upper}'
                            if threshold_var in THRESHOLDS or metric_upper in THRESHOLDS:
                                threshold_found = True
                    
                    if not threshold_found:
                        issues['unchecked_metrics'].append(
                            f"{subject}/{task_dir}/{metrics_file}: No threshold found for {metric}"
                        )
                        logger.warning(f"      No threshold found for {metric}")
    
    return issues

def print_test_results(issues: Dict[str, List[str]]) -> None:
    """Print the test results in a formatted way."""
    print("\n" + "="*80)
    print("METRICS COMPLETENESS TEST RESULTS")
    print("="*80)
    
    total_issues = sum(len(issue_list) for issue_list in issues.values())
    
    if total_issues == 0:
        print("✅ ALL TESTS PASSED!")
        print("All metrics are calculated successfully and checked against appropriate thresholds.")
    else:
        print(f"❌ FOUND {total_issues} ISSUES:")
        
        for issue_type, issue_list in issues.items():
            if issue_list:
                print(f"\n{issue_type.upper().replace('_', ' ')} ({len(issue_list)} issues):")
                for issue in issue_list:
                    print(f"  • {issue}")
    
    print("\n" + "="*80)

def main():
    """Main function to run the metrics completeness test."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test metrics completeness and threshold checking')
    parser.add_argument('subject_folder', nargs='?', help='Specific subject to test (e.g., sub-SK)')
    parser.add_argument('--session', help='Specific session to test (e.g., ses-1)')
    args = parser.parse_args()
    
    print("Running metrics completeness test...")
    issues = test_metrics_completeness(args.subject_folder, args.session)
    print_test_results(issues)
    
    # Exit with error code if issues found
    total_issues = sum(len(issue_list) for issue_list in issues.values())
    if total_issues > 0:
        sys.exit(1)

if __name__ == '__main__':
    main() 