"""
AX-CPT task analysis functions.
"""

import pandas as pd
import numpy as np

def ax_cpt_rdoc_time_resolved(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze AX-CPT task data with time-resolved metrics.
    
    Args:
        df (pd.DataFrame): Input dataframe containing AX-CPT task data.
        
    Returns:
        pd.DataFrame: DataFrame with calculated metrics.
    """
    # Filter for test trials
    test_trials = df[df['trial_id'] == 'test_trial']
    
    # Group by trial_type and calculate metrics
    metrics = test_trials.groupby('trial_type').agg({
        'correct_trial': 'mean',  # Accuracy
        'rt': lambda x: x[test_trials['correct_trial'] == 1].mean(),  # RT for correct trials
        'rt': lambda x: x.isna().mean(),  # Omission rate
    }).reset_index()
    
    # Rename columns for clarity
    metrics.columns = ['condition', 'accuracy', 'rt', 'omission_rate']
    
    return metrics 