"""
Configuration and utility functions for RDOC package.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Callable, Any

from .analyze.tasks import (
    ax_cpt_rdoc,
    cued_task_switching_rdoc,
    flanker_rdoc,
    go_nogo_rdoc,
    n_back_rdoc,
    operation_span_rdoc,
    simple_span_rdoc,
    spatial_cueing_rdoc,
    stop_signal_rdoc,
    stroop_rdoc,
    visual_search_rdoc,
)

def get_function_mapping(exp_name: str) -> Optional[Callable]:
    """Get the appropriate analysis function for a given experiment name.
    
    Args:
        exp_name (str): Name of the experiment.
        
    Returns:
        Optional[Callable]: The analysis function for the experiment, or None if not found.
    """
    mappings = {
        'ax_cpt_rdoc': ax_cpt_rdoc,
        'flanker_rdoc': flanker_rdoc,
        'go_nogo_rdoc': go_nogo_rdoc,
        'n_back_rdoc': n_back_rdoc,
        'simple_span_rdoc': simple_span_rdoc,
        'stop_signal_rdoc': stop_signal_rdoc,
        'spatial_cueing_rdoc': spatial_cueing_rdoc,
        'cued_task_switching_rdoc': cued_task_switching_rdoc,
        'operation_span_rdoc': operation_span_rdoc,
        'stroop_rdoc': stroop_rdoc,
        'visual_search_rdoc': visual_search_rdoc,
    }
    return mappings.get(exp_name, None)

def get_metrics(df: pd.DataFrame, group_by: Optional[List[str]] = None) -> pd.DataFrame:
    """Calculate accuracy, RT, and omission rate metrics.

    Args:
        df (pd.DataFrame): Input dataframe containing task data.
        group_by (Optional[List[str]]): Columns to group by. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with metrics.
    """
    if group_by is None:
        return df.groupby('condition').agg({
            'correct_trial': 'mean',  # Accuracy
            'rt': lambda x: x[df['correct_trial'] == 1].mean(),  # RT for correct trials
            'rt': lambda x: x.isna().mean(),  # Omission rate
        }).reset_index()
    
    return df.groupby(group_by).agg({
        'correct_trial': 'mean',
        'rt': lambda x: x[df['correct_trial'] == 1].mean(),
        'rt': lambda x: x.isna().mean(),
    }).reset_index()

def organize_metrics(
    metrics: pd.DataFrame, group_by: Optional[List[str]] = None
) -> pd.DataFrame:
    """Organize metrics into a standardized format.

    Args:
        metrics (pd.DataFrame): Input dataframe containing metrics.
        group_by (Optional[List[str]]): Columns used for grouping. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with 'metric' and 'value' columns.
    """
    if group_by is None:
        group_by = ['condition']

    valid_group_by_options = {
        ('condition', 'num_stimuli'),
        ('condition', 'delay'),
        ('grid_symmetry',),
    }

    if tuple(group_by) not in valid_group_by_options and group_by != ['condition']:
        raise ValueError(f'Unsupported group_by: {group_by}')

    # Melt the dataframe to get long format
    melted = pd.melt(metrics, id_vars=group_by, var_name='variable', value_name='value')
    
    # Create metric column by concatenating group_by columns and variable
    melted['metric'] = melted[group_by + ['variable']].apply(
        lambda x: '_'.join(x.astype(str)), axis=1
    )
    
    # Select and sort final columns
    result = melted[['metric', 'value']].sort_values('metric')
    
    return result 