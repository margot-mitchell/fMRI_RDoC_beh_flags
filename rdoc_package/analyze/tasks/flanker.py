"""
Flanker task analysis functions.
"""

import pandas as pd
import numpy as np

def flanker_rdoc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze Flanker task data.
    
    Args:
        df (pd.DataFrame): Input dataframe containing Flanker task data.
        
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