"""Generate behavioral flags based on task metrics."""

import os
from typing import Dict, List, Tuple

import polars as pl


def get_flag_rules() -> Dict[str, List[Tuple[str, float, str]]]:
    """Get rules for generating flags for each task.
    
    Returns:
        Dict[str, List[Tuple[str, float, str]]]: Dictionary mapping task names to lists of
            (metric_name, threshold, flag_name) tuples.
    """
    return {
        'axCPT': [
            ('accuracy', 0.5, 'low_accuracy'),
            ('rt', 2000, 'slow_rt'),
        ],
        'cuedTS': [
            ('accuracy', 0.5, 'low_accuracy'),
            ('rt', 2000, 'slow_rt'),
        ],
        'flanker': [
            ('accuracy', 0.5, 'low_accuracy'),
            ('rt', 2000, 'slow_rt'),
        ],
        'goNogo': [
            ('accuracy', 0.5, 'low_accuracy'),
            ('rt', 2000, 'slow_rt'),
        ],
        'nBack': [
            ('accuracy', 0.5, 'low_accuracy'),
            ('rt', 2000, 'slow_rt'),
        ],
        'OpSpan': [
            ('mean_4x4_grid_accuracy_entirely_correct', 0.5, 'low_accuracy'),
            ('mean_4x4_grid_accuracy_respective_of_order', 0.5, 'low_accuracy'),
        ],
        'opOnly': [
            ('mean_4x4_grid_accuracy_entirely_correct', 0.5, 'low_accuracy'),
            ('mean_4x4_grid_accuracy_respective_of_order', 0.5, 'low_accuracy'),
        ],
        'simpleSpan': [
            ('mean_4x4_grid_accuracy_entirely_correct', 0.5, 'low_accuracy'),
            ('mean_4x4_grid_accuracy_respective_of_order', 0.5, 'low_accuracy'),
        ],
        'spatialCueing': [
            ('accuracy', 0.5, 'low_accuracy'),
            ('rt', 2000, 'slow_rt'),
        ],
        'spatialTS': [
            ('accuracy', 0.5, 'low_accuracy'),
            ('rt', 2000, 'slow_rt'),
        ],
        'stopSignal': [
            ('accuracy', 0.5, 'low_accuracy'),
            ('rt', 2000, 'slow_rt'),
        ],
        'stroop': [
            ('accuracy', 0.5, 'low_accuracy'),
            ('rt', 2000, 'slow_rt'),
        ],
        'visualSearch': [
            ('accuracy', 0.5, 'low_accuracy'),
            ('rt', 2000, 'slow_rt'),
        ],
    }


def apply_flag_rules(metrics: pl.DataFrame, rules: List[Tuple[str, float, str]]) -> pl.DataFrame:
    """Apply flag rules to metrics.
    
    Args:
        metrics (pl.DataFrame): DataFrame containing task metrics.
        rules (List[Tuple[str, float, str]]): List of (metric_name, threshold, flag_name) tuples.
        
    Returns:
        pl.DataFrame: DataFrame containing flags.
    """
    flags = []
    for metric_name, threshold, flag_name in rules:
        if metric_name in metrics['metric']:
            metric_value = metrics.filter(pl.col('metric') == metric_name)['value'].item()
            flag_value = 1 if metric_value < threshold else 0
            flags.append({
                'metric': flag_name,
                'value': flag_value,
                'task': metrics['task'].iloc[0],
                'file': metrics['file'].iloc[0],
            })
    
    return pl.DataFrame(flags)


def generate_subject_flags(subject_folder: str) -> None:
    """Generate flags for all tasks in a subject folder.
    
    Args:
        subject_folder (str): Path to subject folder containing metrics.
    """
    # Check if metrics file exists
    metrics_file = os.path.join(subject_folder, 'behavioral_summary.csv')
    if not os.path.exists(metrics_file):
        print(f"No metrics file found in {subject_folder}")
        return
        
    # Read metrics
    metrics = pl.read_csv(metrics_file)
    
    # Get list of tasks that have metrics
    available_tasks = metrics['task'].unique().to_list()
    
    # Get flag rules for available tasks
    flag_rules = get_flag_rules()
    available_rules = {
        task: rules for task, rules in flag_rules.items()
        if task in available_tasks
    }
    
    if not available_rules:
        print(f"No flag rules available for tasks in {subject_folder}")
        return
        
    # Generate flags for each available task
    all_flags = []
    for task, rules in available_rules.items():
        task_metrics = metrics.filter(pl.col('task') == task)
        flags = apply_flag_rules(task_metrics, rules)
        all_flags.append(flags)
    
    if all_flags:
        # Combine all flags
        combined_flags = pl.concat(all_flags)
        # Save to CSV
        output_file = os.path.join(subject_folder, 'behavioral_flags.csv')
        combined_flags.write_csv(output_file)
        print(f"Saved flags to {output_file}")
    else:
        print(f"No flags generated for {subject_folder}")


def main():
    """Main function to generate flags for all subjects."""
    # Get list of subject folders
    subject_folders = [f for f in os.listdir('preprocessed_data') 
                      if os.path.isdir(os.path.join('preprocessed_data', f))]
    
    for subject_folder in subject_folders:
        print(f"Generating flags for {subject_folder}...")
        generate_subject_flags(os.path.join('preprocessed_data', subject_folder))


if __name__ == '__main__':
    main() 