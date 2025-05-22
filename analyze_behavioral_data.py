"""Analyzes preprocessed behavioral data from fMRI RDoC studies.

This script reads preprocessed parquet files and calculates various metrics
for quality control and analysis.

Usage:
    python analyze_behavioral_data.py <subject_folder>
    Example: python analyze_behavioral_data.py sub-SK
"""

import os
import sys
import logging
import ast
from typing import Callable, List, Optional, Dict

import numpy as np
import polars as pl

from thresholds_config import (
    # Stop Signal thresholds
    STOP_SIGNAL_ACCURACY_MIN,
    STOP_SIGNAL_ACCURACY_MAX,
    STOP_SIGNAL_GO_ACCURACY,
    STOP_SIGNAL_GO_RT,
    STOP_SIGNAL_OMISSION_RATE,
    # AX-CPT thresholds
    AX_CPT_ACCURACY,
    AX_CPT_OMISSION_RATE,
    # Go/NoGo thresholds
    GONOGO_GO_ACCURACY,
    GONOGO_NOGO_ACCURACY,
    GONOGO_MEAN_ACCURACY,
    GONOGO_OMISSION_RATE,
    # Flanker thresholds
    FLANKER_ACCURACY,
    FLANKER_OMISSION_RATE,
    # Operation Span thresholds
    OP_SPAN_ASYMMETRIC_ACCURACY,
    OP_SPAN_SYMMETRIC_ACCURACY,
    OP_SPAN_4X4_ACCURACY,
    OP_SPAN_ORDER_DIFF,
    # Simple Span thresholds
    SIMPLE_SPAN_4X4_ACCURACY,
    SIMPLE_SPAN_ORDER_DIFF,
    # N-Back thresholds
    NBACK_MATCH_WEIGHT,
    NBACK_MISMATCH_WEIGHT,
    NBACK_MATCH_ACCURACY,
    NBACK_MISMATCH_ACCURACY,
    NBACK_WEIGHTED_ACCURACY,
    # Cued TS thresholds
    CUED_TS_ACCURACY,
    CUED_TS_OMISSION_RATE,
    # Spatial cueing thresholds
    SPATIAL_CUEING_ACCURACY,
    SPATIAL_CUEING_OMISSION_RATE,
    # Spatial TS thresholds
    SPATIAL_TS_ACCURACY,
    SPATIAL_TS_OMISSION_RATE,
    # Stroop thresholds
    STROOP_ACCURACY,
    STROOP_OMISSION_RATE,
    # Visual search thresholds
    VISUAL_SEARCH_ACCURACY,
    VISUAL_SEARCH_OMISSION_RATE,
    # Add THRESHOLDS dictionary
    THRESHOLDS
)

def get_metrics(df: pl.DataFrame, group_by: List[str] | None = None) -> pl.DataFrame:
    """Calculate accuracy, RT, and omission rate metrics.

    Args:
        df (pl.DataFrame): Input dataframe containing task data.
        group_by (list[str] | None, optional): Columns to group by.
        Defaults to None.

    Returns:
        pl.DataFrame: DataFrame with metrics.
    """
    if group_by is None:
        # Try different possible condition columns
        condition_col = None
        for col in ['condition', 'trial_type']:
            if col in df.columns:
                condition_col = col
                break
        if condition_col is None:
            raise ValueError('No condition column found in dataframe')

        return df.group_by(condition_col).agg(
            accuracy=pl.col('correct_trial').mean(),
            rt=pl.col('rt').filter(pl.col('correct_trial') == 1).mean(),
            omission_rate=pl.col('rt').is_null().mean(),
        )
    return df.group_by(group_by).agg(
        accuracy=pl.col('correct_trial').mean(),
        rt=pl.col('rt').filter(pl.col('correct_trial') == 1).mean(),
        omission_rate=pl.col('rt').is_null().mean(),
    )

def get_stop_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate stop-signal metrics.

    Args:
        df (pl.DataFrame): Input dataframe containing stop-signal task data.

    Returns:
        pl.DataFrame: DataFrame with 'metric' and 'value' columns.
    """
    return (
        df.select(
            min_SSD=pl.col('SSD').min(),
            max_SSD=pl.col('SSD').max(),
            mean_SSD=pl.col('SSD').mean(),
            final_SSD=pl.col('SSD').last(),
        )
        .unpivot(index=None)
        .rename({'variable': 'metric'})
    )

def get_span_metrics(test_trials: pl.DataFrame):
    """Calculate span task metrics.

    Args:
        test_trials (pl.DataFrame): DataFrame containing test trial data.

    Returns:
        Dict: Dictionary containing span task metrics.
    """
    metrics = {}
    
    # Check if RT columns exist
    has_rt_columns = 'rt_each_spatial_location_response_grid' in test_trials.columns
    if not has_rt_columns:
        print('WARNING: Column rt_each_spatial_location_response_grid is missing. Skipping RT-related metrics.')
    else:
        def rename_rt_cols(col: str) -> str:
            return col.replace(
                'rt_each_spatial_location_response_grid', '4x4_grid_response_time'
            ).replace(
                'rt_moving_each_spatial_location_response_grid', '4x4_grid_movement_time'
            )

        def mean_of_differences(values: List[float]) -> float:
            if not values:
                return 0.0
            # Calculate the sum of the subtracted values
            total = values[0] + sum(
                values[i] - values[i - 1] for i in range(1, len(values))
            )
            # Return the mean
            return total / len(values)

        def extract_row_rts(
            rt_cols: List[str], func: Callable[[List[float]], float]
        ) -> dict:
            rts = {}
            for col in rt_cols:
                series = test_trials.select(pl.col(col)).to_series()
                rts[col] = [
                    func(ast.literal_eval(item))
                    for item in series
                    if item is not None and ast.literal_eval(item)
                ]
            return rts

        ## RESPONSE TIMES
        rt_cols = [
            'rt_each_spatial_location_response_grid',
            'rt_moving_each_spatial_location_response_grid',
        ]

        first_rts = extract_row_rts(rt_cols, lambda x: x[0])
        mean_rts = extract_row_rts(rt_cols, mean_of_differences)

        mean_first_rts = {}
        mean_mean_rts = {}
        for col in rt_cols:
            mean_first_rts[col] = np.mean(first_rts[col])
            mean_mean_rts[col] = np.mean(mean_rts[col])

        # Make into single dict and prepend "mean_" to the keys
        mean_rts = {
            f'mean_{rename_rt_cols(col)}': mean_mean_rts[col] for col in mean_mean_rts
        }
        # Make into single dict and prepend "first_" to the keys
        first_rts = {
            f'first_{rename_rt_cols(col)}': mean_first_rts[col] for col in mean_first_rts
        }
        
        metrics.update({k: float(v) for k, v in mean_rts.items()})
        metrics.update({k: float(v) for k, v in first_rts.items()})

    def extract_row_accuracies(test_trials: pl.DataFrame) -> Dict[str, List[float]]:
        def get_irrespective_of_order_accuracy(
            response: List[str], correct: List[str]
        ) -> float:
            # Convert to sets for intersection operation
            response_set = set(response)
            correct_set = set(correct)
            matches = response_set.intersection(correct_set)
            return len(matches) / len(correct_set)

        def get_with_respect_to_order_accuracy(
            response: List[str], correct: List[str]
        ) -> float:
            matches = sum(1 for r, c in zip(response, correct) if r == c)
            return matches / len(correct)

        # Responses
        response_series = test_trials.select(pl.col('valid_responses')).to_series()
        responses = [
            ast.literal_eval(item) for item in response_series if item is not None
        ]
        # Correct responses
        correct_series = test_trials.select(pl.col('correct_cell_order')).to_series()
        correct_responses = [
            ast.literal_eval(item) for item in correct_series if item is not None
        ]

        accuracy_irrespective_of_order = []
        accuracy_with_respect_to_order = []

        for response, correct in zip(responses, correct_responses):
            # Irrespective of order calculation
            accuracy_irrespective_of_order.append(
                get_irrespective_of_order_accuracy(response, correct)
            )
            accuracy_with_respect_to_order.append(
                get_with_respect_to_order_accuracy(response, correct)
            )

        return {
            'accuracy_irrespective_of_order': accuracy_irrespective_of_order,
            'accuracy_respective_of_order': accuracy_with_respect_to_order,
        }

    def get_mean_number_of_responses(test_trials: pl.DataFrame) -> float:
        responses = test_trials.select(pl.col('valid_responses')).to_series()

        def safe_len(x):
            try:
                val = ast.literal_eval(x) if x is not None else []
                return len(val) if isinstance(val, list) else 0
            except Exception:
                return 0

        response_lengths = [safe_len(x) for x in responses]
        # Only include non-empty responses in the mean
        non_empty_lengths = [length for length in response_lengths if length > 0]
        mean = np.mean(non_empty_lengths) if non_empty_lengths else 0.0
        return float(mean)

    def get_4x4_grid_omission_rate(test_trials: pl.DataFrame) -> float:
        responses = test_trials.select(pl.col('valid_responses')).to_series()

        def safe_len(x):
            try:
                val = ast.literal_eval(x) if x is not None else []
                return len(val) if isinstance(val, list) else 0
            except Exception:
                return 0

        response_lengths = [safe_len(x) for x in responses]
        num_empty = sum(1 for length in response_lengths if length == 0)
        total = len(response_lengths)
        omission_rate = num_empty / total if total > 0 else 0.0
        return float(omission_rate)

    def get_time_remaining_after_last_response(test_trials: pl.DataFrame) -> float:
        if not has_rt_columns:
            return 0.0
            
        responses = test_trials.select(
            pl.col('rt_each_spatial_location_response_grid')
        ).to_series()
        test_trial_durations = test_trials.select(pl.col('trial_duration')).to_series()
        # Assert that all test_trial_durations are the same
        assert test_trial_durations.n_unique() == 1
        # take first value from the trial_duration column
        # - should be the same for all rows, which is now 7000ms
        test_trial_duration = test_trial_durations.item(0)
        last_response_times = responses.map_elements(
            lambda x: ast.literal_eval(x)[-1]
            if x is not None and len(ast.literal_eval(x)) > 0
            else 0,
            return_dtype=pl.Float64,
        )
        time_remaining = test_trial_duration - last_response_times
        return time_remaining.mean()

    ## ACCURACIES
    # Accuracy irrespective of spatial sequence order
    accuracies = extract_row_accuracies(test_trials)

    mean_accuracy_irrespective_of_order = np.mean(
        accuracies['accuracy_irrespective_of_order']
    )
    mean_accuracy_respective_of_order = np.mean(
        accuracies['accuracy_respective_of_order']
    )

    # Get number of responses and time remaining after last response
    mean_num_responses = get_mean_number_of_responses(test_trials)
    omission_rate = get_4x4_grid_omission_rate(test_trials)
    time_remaining_after_last_response = get_time_remaining_after_last_response(
        test_trials
    )

    metrics.update({
        'mean_4x4_grid_accuracy_irrespective_of_order': float(mean_accuracy_irrespective_of_order),
        'mean_4x4_grid_accuracy_respective_of_order': float(mean_accuracy_respective_of_order),
        'mean_number_of_responses': mean_num_responses,
        '4x4_grid_omission_rate': omission_rate,
        'mean_time_remaining_after_last_response': time_remaining_after_last_response,
    })

    return metrics

def organize_metrics(
    metrics: pl.DataFrame, group_by: List[str] | None = None
) -> pl.DataFrame:
    """Organize metrics into a standardized format.

    Args:
        metrics (pl.DataFrame): Input dataframe containing metrics.
        group_by (list[str] | None, optional): Columns used for grouping.
        Defaults to None.

    Returns:
        pl.DataFrame: DataFrame with 'metric' and 'value' columns.
    """

    def create_metric_column(group_by_cols):
        return pl.concat_str(
            [pl.col(col).cast(pl.Utf8) for col in group_by_cols] + [pl.col('variable')],
            separator='_',
        ).alias('metric')

    if group_by is None:
        group_by = ['condition']

    valid_group_by_options = {
        ('condition', 'num_stimuli'),
        ('condition', 'delay'),
        ('grid_symmetry',),
        ('task_condition', 'cue_condition'),
        ('trial_type', 'cue_condition'),
        ('condition', 'num_stimuli'),
        ('task', 'cue_condition'),
    }

    if tuple(group_by) not in valid_group_by_options and group_by != ['condition']:
        raise ValueError(f'Unsupported group_by: {group_by}')

    return (
        metrics.unpivot(index=group_by)
        .with_columns(create_metric_column(group_by))
        .select(pl.col('metric'), pl.col('value'))
        .sort('metric')
    )

def ax_cpt_rdoc(df: pl.DataFrame) -> pl.DataFrame:
    """Process AX-CPT task data.

    Args:
        df (pl.DataFrame): Input dataframe containing task data.

    Returns:
        pl.DataFrame: DataFrame with 'metric' and 'value' columns.
    """
    # Filter dataframe for test trials only
    trial_ids = df.select('trial_id').unique().to_series().to_list()

    if 'test_trial' in trial_ids:
        test_trials = df.filter(pl.col('trial_id') == 'test_trial')
    else:
        test_trials = df.filter(pl.col('trial_id') == 'test_probe')

    if 'success' not in test_trials.columns and 'correct_trial' in test_trials.columns:
        test_trials = test_trials.with_columns(success=pl.col('correct_trial'))

    # Group by condition and calculate metrics
    metrics = get_metrics(test_trials)
    return organize_metrics(metrics)

def ax_cpt_rdoc_time_resolved(df: pl.DataFrame) -> pl.DataFrame:
    """Process AX-CPT task data with time-resolved metrics.

    Args:
        df (pl.DataFrame): Input dataframe containing task data.

    Returns:
        pl.DataFrame: DataFrame with 'metric' and 'value' columns.
    """
    # Filter dataframe for test trials only
    trial_ids = df.select('trial_id').unique().to_series().to_list()

    if 'test_trial' in trial_ids:
        test_trials = df.filter(pl.col('trial_id') == 'test_trial')
    else:
        test_trials = df.filter(pl.col('trial_id') == 'test_probe')

    if 'success' not in test_trials.columns and 'correct_trial' in test_trials.columns:
        test_trials = test_trials.with_columns(success=pl.col('correct_trial'))

    # Calculate proportion of trials with any fixation/cue responses
    # First, get all trials that could have fixation/cue responses
    fixation_cue_trials = df.filter(
        (pl.col('trial_id').is_in(['test_fixation', 'test_cue']))
        | (
            (pl.col('trial_id') == 'test_inter-stimulus')
            & (pl.col('stimulus').str.contains('fixation'))
        )
    )

    if fixation_cue_trials.height > 0:
        # Get all test trial indices
        test_indices = test_trials.select('trial_index').to_series().to_list()

        # For each test trial, find the immediate preceding fixation/cue trials
        responses = []
        for test_idx in test_indices:
            # Find the fixation/cue trials that immediately precede this test trial
            # A test trial sequence is: fixation -> cue -> inter-stimulus -> probe
            # So we look at the 3 trials immediately before the test trial
            prev_trials = fixation_cue_trials.filter(
                (pl.col('trial_index') < test_idx)
                & (
                    pl.col('trial_index') >= test_idx - 3
                )  # Only look at the 3 trials before
            )

            # If any of these trials had a response, count it
            has_response = prev_trials.filter(pl.col('rt').is_not_null()).height > 0
            responses.append(1 if has_response else 0)

        # Calculate proportion as number of test trials with responses
        # divided by total test trials
        total_responses = sum(responses)
        total_test_trials = len(test_indices)

        # Ensure we don't exceed 1.0
        fixation_cue_responses = min(total_responses / total_test_trials, 1.0)
    else:
        fixation_cue_responses = 0.0  # Default to 0 if no fixation/cue trials

    # Group by condition and calculate metrics
    metrics = get_metrics(test_trials)

    # Create a DataFrame for the fixation/cue response proportion
    fixation_metric = pl.DataFrame(
        {
            'metric': ['proportion_cue/fixation_responses'],
            'value': [fixation_cue_responses],
        }
    )

    # Get the organized metrics and combine with the fixation metric
    organized_metrics = organize_metrics(metrics)
    return pl.concat([organized_metrics, fixation_metric])

def cued_task_switching_rdoc(df: pl.DataFrame) -> pl.DataFrame:
    """Process cued task switching task data.

    Args:
        df (pl.DataFrame): Input dataframe containing task data.

    Returns:
        pl.DataFrame: DataFrame with 'metric' and 'value' columns.
    """
    # Filter dataframe for test trials only
    test_trials = df.filter(pl.col('trial_id') == 'test_trial')

    # Calculate success based on response matching correct_response
    test_trials = test_trials.with_columns(
        success=pl.when(pl.col('response') == pl.col('correct_response'))
        .then(True)
        .when(pl.col('response').is_null())
        .then(None)
        .otherwise(False)
    )

    # Calculate magnitude judgment accuracy (collapsed across trial type and cue)
    magnitude_trials = test_trials.filter(
        pl.col('task').str.to_lowercase().str.contains('magnitude|mag', strict=False)
    )
    magnitude_accuracy = (
        magnitude_trials.select(pl.col('success').mean()).item()
        if magnitude_trials.height > 0
        else None
    )

    # Calculate parity judgment accuracy (collapsed across trial type and cue)
    parity_trials = test_trials.filter(
        pl.col('task').str.to_lowercase().str.contains('parity|par', strict=False)
    )
    parity_accuracy = (
        parity_trials.select(pl.col('success').mean()).item()
        if parity_trials.height > 0
        else None
    )

    # Get base metrics grouped by task_condition and cue_condition
    metrics = get_metrics(test_trials, group_by=['task_condition', 'cue_condition'])
    melted = organize_metrics(metrics, group_by=['task_condition', 'cue_condition'])

    # Filter out task_na metrics and na_na metrics
    filtered_melted = melted.filter(
        (~pl.col('metric').str.starts_with('task_na'))
        & (
            ~pl.col('metric').is_in(
                ['na_na_accuracy', 'na_na_omission_rate', 'na_na_rt']
            )
        )
    )

    # Add magnitude and parity accuracy as separate rows
    new_metrics = pl.DataFrame(
        {
            'metric': ['magnitude_accuracy', 'parity_accuracy'],
            'value': [magnitude_accuracy, parity_accuracy],
        }
    )

    # Add attention check metrics as separate rows
    attention_check_trials = df.filter(pl.col('trial_id') == 'test_attention_check')
    attention_check_mean_rt = (
        attention_check_trials.select(pl.col('rt').mean()).item()
        if attention_check_trials.height > 0
        else None
    )
    attention_check_mean_accuracy = (
        attention_check_trials.select(pl.col('correct_trial').mean()).item()
        if attention_check_trials.height > 0
        else None
    )
    attention_metrics = pl.DataFrame(
        {
            'metric': ['attention_check_mean_rt', 'attention_check_mean_accuracy'],
            'value': [attention_check_mean_rt, attention_check_mean_accuracy],
        }
    )

    # Rename metrics as requested
    rename_map = {
        'stay_stay_accuracy': 'task_stay_cue_stay_accuracy',
        'stay_stay_omission_rate': 'task_stay_cue_stay_omission_rate',
        'stay_stay_rt': 'task_stay_cue_stay_rt',
        'stay_switch_accuracy': 'task_stay_cue_switch_accuracy',
        'stay_switch_omission_rate': 'task_stay_cue_switch_omission_rate',
        'stay_switch_rt': 'task_stay_cue_switch_rt',
        'switch_switch_accuracy': 'task_switch_cue_switch_accuracy',
        'switch_switch_omission_rate': 'task_switch_cue_switch_omission_rate',
        'switch_switch_rt': 'task_switch_cue_switch_rt',
    }
    return pl.concat([filtered_melted, new_metrics, attention_metrics]).with_columns(
        pl.when(pl.col('metric').is_in(list(rename_map.keys())))
        .then(pl.col('metric').replace(rename_map))
        .otherwise(pl.col('metric'))
        .alias('metric')
    )

def flanker_rdoc(df: pl.DataFrame) -> pl.DataFrame:
    """Process flanker task data.

    Args:
        df (pl.DataFrame): Input dataframe containing task data.

    Returns:
        pl.DataFrame: DataFrame with 'metric' and 'value' columns.
    """
    # Filter dataframe for test trials only
    test_trials = df.filter(pl.col('trial_id') == 'test_trial')

    # Group by condition and calculate metrics
    metrics = get_metrics(test_trials)
    return organize_metrics(metrics)

def go_nogo_rdoc(df: pl.DataFrame) -> pl.DataFrame:
    """Process go/no-go task data.

    Args:
        df (pl.DataFrame): Input dataframe containing task data.

    Returns:
        pl.DataFrame: DataFrame with 'metric' and 'value' columns.
    """
    # Filter dataframe for test trials only
    test_trials = df.filter(pl.col('trial_id') == 'test_trial')
    # Group by condition and calculate metrics
    metrics = get_metrics(test_trials)
    melted = organize_metrics(metrics)

    # drop nogo_omission_rate and nogo_rt
    melted = melted.filter(
        ~(pl.col('metric') == 'nogo_omission_rate') & ~(pl.col('metric') == 'nogo_rt')
    )

    # NOTE: Should probably change
    # how this is being done, but for now
    # just recalculate nogo_omission_rate
    nogo_trials = df.filter(pl.col('condition') == 'nogo')
    nogo_rt_value = nogo_trials.select(
        pl.col('rt').filter(pl.col('correct_trial') == 0).mean()
    ).item()

    # Create a properly formatted DataFrame with metric and value columns
    nogo_rt_df = pl.DataFrame({'metric': ['nogo_rt'], 'value': [nogo_rt_value]})
    return melted.vstack(nogo_rt_df)

def n_back_rdoc(df: pl.DataFrame) -> pl.DataFrame:
    """Process n-back task data.

    Args:
        df (pl.DataFrame): Input dataframe containing task data.

    Returns:
        pl.DataFrame: DataFrame with 'metric' and 'value' columns.
    """
    # Add n-back processing logic
    test_trials = df.filter(pl.col('trial_id') == 'test_trial')
    metrics = get_metrics(test_trials, group_by=['condition', 'delay'])
    melted = organize_metrics(metrics, group_by=['condition', 'delay'])
    
    # Calculate weighted accuracy for match_1 and mismatch_1
    match_1_accuracy = melted.filter(pl.col('metric') == 'match_1_accuracy')['value'].item()
    mismatch_1_accuracy = melted.filter(pl.col('metric') == 'mismatch_1_accuracy')['value'].item()
    weighted_accuracy_1 = 0.2 * match_1_accuracy + 0.8 * mismatch_1_accuracy
    
    # Calculate weighted accuracy for match_2 and mismatch_2
    match_2_accuracy = melted.filter(pl.col('metric') == 'match_2_accuracy')['value'].item()
    mismatch_2_accuracy = melted.filter(pl.col('metric') == 'mismatch_2_accuracy')['value'].item()
    weighted_accuracy_2 = 0.2 * match_2_accuracy + 0.8 * mismatch_2_accuracy
    
    # Add weighted accuracy metrics
    weighted_metrics = pl.DataFrame({
        'metric': ['weighted_accuracy_1', 'weighted_accuracy_2'],
        'value': [weighted_accuracy_1, weighted_accuracy_2]
    })
    
    # Combine with original metrics
    return melted.vstack(weighted_metrics)

def calculate_processing_metrics(processing_trials: pl.DataFrame, full_df: pl.DataFrame) -> pl.DataFrame:
    grid_trials = full_df.filter(
        (pl.col('trial_id') == 'test_trial')
        & (pl.col('internal_node_id').str.contains('0.0-3.0'))
    )
    grid_indices = grid_trials.select('trial_index').to_series().to_list()
    processing_after_grid = []
    for grid_idx in grid_indices:
        next_trial = (
            full_df.filter(
                (pl.col('trial_index') > grid_idx)
                & (pl.col('trial_id') == 'processing_trial')
            )
            .select('trial_index')
            .min()
        )
        if next_trial is not None:
            processing_after_grid.append(next_trial.item())
    relevant_trials = processing_trials.filter(
        pl.col('trial_index').is_in(processing_after_grid)
    )
    overall_omission_rate = relevant_trials.select(
        pl.col('rt').is_null().mean()
    ).item()
    metrics = get_metrics(processing_trials, group_by=['grid_symmetry'])
    melted = organize_metrics(metrics, group_by=['grid_symmetry'])
    melted = melted.filter(~pl.col('metric').str.ends_with('_omission_rate'))
    melted = melted.with_columns(
        pl.col('metric').str.replace('^', '8x8_grid_').alias('metric')
    )
    symmetric_count = processing_trials.filter(
        pl.col('grid_symmetry') == 'symmetric'
    ).height
    asymmetric_count = processing_trials.filter(
        pl.col('grid_symmetry') == 'asymmetric'
    ).height
    additional_metrics_df = pl.DataFrame(
        {
            'metric': [
                '8x8_grid_omission_rate',
                'symmetric_trial_count',
                'asymmetric_trial_count',
            ],
            'value': [overall_omission_rate, float(symmetric_count), float(asymmetric_count)],
        }
    )
    return melted.vstack(additional_metrics_df)

def operation_span_rdoc(df: pl.DataFrame) -> pl.DataFrame:
    """Process operation span task data.

    Args:
        df (pl.DataFrame): Input dataframe containing task data.

    Returns:
        pl.DataFrame: DataFrame with 'metric' and 'value' columns.
    """
    # Filter dataframe for test trials only
    test_trials = df.filter(pl.col('trial_id') == 'test_trial')
    processing_trials = df.filter(pl.col('trial_id') == 'processing_trial')

    # Get span metrics
    span_metrics = get_span_metrics(test_trials)
    
    # Convert span metrics dictionary to DataFrame
    if span_metrics:
        span_metrics_df = pl.DataFrame({
            'metric': list(span_metrics.keys()),
            'value': list(span_metrics.values())
        })
    else:
        span_metrics_df = pl.DataFrame({'metric': pl.Series([], pl.String), 'value': pl.Series([], pl.Float64)})

    # Get processing metrics
    processing_metrics = calculate_processing_metrics(processing_trials, df)
    
    # Combine metrics
    all_metrics = pl.concat([span_metrics_df, processing_metrics])
    
    return all_metrics

def simple_span_rdoc(df: pl.DataFrame) -> pl.DataFrame:
    """Process simple span task data.

    Args:
        df (pl.DataFrame): Input dataframe containing task data.

    Returns:
        pl.DataFrame: DataFrame with 'metric' and 'value' columns.
    """
    # Filter dataframe for test trials only
    test_trials = df.filter(pl.col('trial_id') == 'test_trial')

    # Get span metrics
    span_metrics = get_span_metrics(test_trials)
    
    # Convert span metrics dictionary to DataFrame
    if span_metrics:
        span_metrics_df = pl.DataFrame({
            'metric': list(span_metrics.keys()),
            'value': list(span_metrics.values())
        })
    else:
        span_metrics_df = pl.DataFrame({'metric': pl.Series([], pl.String), 'value': pl.Series([], pl.Float64)})
    
    return span_metrics_df

def spatial_cueing_rdoc(df: pl.DataFrame) -> pl.DataFrame:
    """Process spatial cueing task data.

    Args:
        df (pl.DataFrame): Input dataframe containing task data.

    Returns:
        pl.DataFrame: DataFrame with 'metric' and 'value' columns.
    """
    # Add spatial cueing processing logic
    test_trials = df.filter(pl.col('trial_id') == 'test_trial')
    metrics = get_metrics(test_trials)
    return organize_metrics(metrics)

def spatial_task_switching_rdoc(df: pl.DataFrame) -> pl.DataFrame:
    """Process spatial task switching task data.

    Args:
        df (pl.DataFrame): Input dataframe containing task data.

    Returns:
        pl.DataFrame: DataFrame with 'metric' and 'value' columns.
    """
    # Add spatial task switching processing logic
    test_trials = df.filter(pl.col('trial_id') == 'test_trial')

    # Group by condition and calculate metrics
    if 'condition' in test_trials.columns:
        metrics = get_metrics(test_trials)
        melted = organize_metrics(metrics)
    else:
        metrics = get_metrics(test_trials, group_by=['task_condition', 'cue_condition'])
        melted = organize_metrics(metrics, group_by=['task_condition', 'cue_condition'])

    # Remove row with n/a metrics
    return melted.filter(~pl.col('metric').str.starts_with('task_na'))

def stop_signal_rdoc(df: pl.DataFrame) -> pl.DataFrame:
    """Process stop signal task data.

    Args:
        df (pl.DataFrame): Input dataframe containing task data.

    Returns:
        pl.DataFrame: DataFrame with 'metric' and 'value' columns.
    """
    # Add stop signal processing logic
    test_trials = df.filter(pl.col('trial_id') == 'test_trial')
    # Group by condition and calculate metrics
    metrics = get_metrics(test_trials)
    melted = organize_metrics(metrics)
    stop_metrics = get_stop_metrics(test_trials)
    # Remove row with metric "stop_omission_rate"
    melted = melted.filter(pl.col('metric') != 'stop_omission_rate')
    return melted.vstack(stop_metrics)

def stroop_rdoc(df: pl.DataFrame) -> pl.DataFrame:
    """Process Stroop task data.

    Args:
        df (pl.DataFrame): Input dataframe containing task data.

    Returns:
        pl.DataFrame: DataFrame with 'metric' and 'value' columns.
    """
    # Add Stroop processing logic
    test_trials = df.filter(pl.col('trial_id') == 'test_trial')
    metrics = get_metrics(test_trials)
    return organize_metrics(metrics)

def visual_search_rdoc(df: pl.DataFrame) -> pl.DataFrame:
    """Process visual search task data.

    Args:
        df (pl.DataFrame): Input dataframe containing task data.

    Returns:
        pl.DataFrame: DataFrame with 'metric' and 'value' columns.
    """
    # Filter dataframe for test trials only
    test_trials = df.filter(pl.col('trial_id') == 'test_trial')

    # Group by condition and calculate metrics
    metrics = get_metrics(test_trials, group_by=['condition', 'num_stimuli'])
    melted = organize_metrics(metrics, group_by=['condition', 'num_stimuli'])

    # Filter out task_na metrics
    return melted.filter(~pl.col('metric').str.starts_with('task_na'))

def check_thresholds_from_csv(task_metrics_df: pl.DataFrame, task_name: str) -> List[tuple]:
    """Check if any metrics violate thresholds by reading from metrics CSV.
    
    Args:
        task_metrics_df (pl.DataFrame): DataFrame containing task metrics
        task_name (str): Name of the task
        
    Returns:
        List[tuple]: List of violations (metric, value, threshold)
    """
    violations = []
    # Create metrics dictionary from the first row of each metric (in case there are multiple files)
    metrics_dict = {}
    for metric in task_metrics_df['metric'].unique():
        # Get the first value for each metric
        value = task_metrics_df.filter(pl.col('metric') == metric)['value'].item()
        metrics_dict[metric] = value

    if task_name == 'stopSignal':
        stop_accuracy = metrics_dict.get('stop_accuracy', 0)
        go_accuracy = metrics_dict.get('go_accuracy', 0)
        go_rt = metrics_dict.get('go_rt', 0)
        go_omission_rate = metrics_dict.get('go_omission_rate', 0)

        if stop_accuracy < STOP_SIGNAL_ACCURACY_MIN or stop_accuracy > STOP_SIGNAL_ACCURACY_MAX:
            violations.append(('stop_accuracy', stop_accuracy, STOP_SIGNAL_ACCURACY_MIN))
        if go_accuracy < STOP_SIGNAL_GO_ACCURACY:
            violations.append(('go_accuracy', go_accuracy, STOP_SIGNAL_GO_ACCURACY))
        if go_rt > STOP_SIGNAL_GO_RT:
            violations.append(('go_rt', go_rt, STOP_SIGNAL_GO_RT))
        if go_omission_rate > STOP_SIGNAL_OMISSION_RATE:
            violations.append(('go_omission_rate', go_omission_rate, STOP_SIGNAL_OMISSION_RATE))

    elif task_name == 'axCPT':
        ax_accuracy = metrics_dict.get('AX_accuracy', 0)
        bx_accuracy = metrics_dict.get('BX_accuracy', 0)
        ay_accuracy = metrics_dict.get('AY_accuracy', 0)
        by_accuracy = metrics_dict.get('BY_accuracy', 0)
        ax_omission_rate = metrics_dict.get('AX_omission_rate', 0)
        bx_omission_rate = metrics_dict.get('BX_omission_rate', 0)
        ay_omission_rate = metrics_dict.get('AY_omission_rate', 0)
        by_omission_rate = metrics_dict.get('BY_omission_rate', 0)

        if ax_accuracy < AX_CPT_ACCURACY:
            violations.append(('AX_accuracy', ax_accuracy, AX_CPT_ACCURACY))
        if bx_accuracy < AX_CPT_ACCURACY:
            violations.append(('BX_accuracy', bx_accuracy, AX_CPT_ACCURACY))
        if ay_accuracy < AX_CPT_ACCURACY:
            violations.append(('AY_accuracy', ay_accuracy, AX_CPT_ACCURACY))
        if by_accuracy < AX_CPT_ACCURACY:
            violations.append(('BY_accuracy', by_accuracy, AX_CPT_ACCURACY))
        if ax_omission_rate > AX_CPT_OMISSION_RATE:
            violations.append(('AX_omission_rate', ax_omission_rate, AX_CPT_OMISSION_RATE))
        if bx_omission_rate > AX_CPT_OMISSION_RATE:
            violations.append(('BX_omission_rate', bx_omission_rate, AX_CPT_OMISSION_RATE))
        if ay_omission_rate > AX_CPT_OMISSION_RATE:
            violations.append(('AY_omission_rate', ay_omission_rate, AX_CPT_OMISSION_RATE))
        if by_omission_rate > AX_CPT_OMISSION_RATE:
            violations.append(('BY_omission_rate', by_omission_rate, AX_CPT_OMISSION_RATE))

    elif task_name == 'goNogo':
        go_accuracy = metrics_dict.get('go_accuracy', 0)
        nogo_accuracy = metrics_dict.get('nogo_accuracy', 0)
        mean_accuracy = (go_accuracy + nogo_accuracy) / 2
        go_omission_rate = metrics_dict.get('go_omission_rate', 0)

        if go_accuracy < GONOGO_GO_ACCURACY and nogo_accuracy < GONOGO_NOGO_ACCURACY:
            violations.append(('go_accuracy', go_accuracy, GONOGO_GO_ACCURACY))
        if mean_accuracy < GONOGO_MEAN_ACCURACY:
            violations.append(('mean_accuracy', mean_accuracy, GONOGO_MEAN_ACCURACY))
        if go_omission_rate > GONOGO_OMISSION_RATE:
            violations.append(('go_omission_rate', go_omission_rate, GONOGO_OMISSION_RATE))

    elif task_name == 'cuedTS':
        stay_stay_accuracy = metrics_dict.get('task_stay_cue_stay_accuracy', 0)
        stay_switch_accuracy = metrics_dict.get('task_stay_cue_switch_accuracy', 0)
        switch_switch_accuracy = metrics_dict.get('task_switch_cue_switch_accuracy', 0)
        stay_stay_omission = metrics_dict.get('task_stay_cue_stay_omission_rate', 0)
        stay_switch_omission = metrics_dict.get('task_stay_cue_switch_omission_rate', 0)
        switch_switch_omission = metrics_dict.get('task_switch_cue_switch_omission_rate', 0)

        if stay_stay_accuracy < CUED_TS_ACCURACY:
            violations.append(('task_stay_cue_stay_accuracy', stay_stay_accuracy, CUED_TS_ACCURACY))
        if stay_switch_accuracy < CUED_TS_ACCURACY:
            violations.append(('task_stay_cue_switch_accuracy', stay_switch_accuracy, CUED_TS_ACCURACY))
        if switch_switch_accuracy < CUED_TS_ACCURACY:
            violations.append(('task_switch_cue_switch_accuracy', switch_switch_accuracy, CUED_TS_ACCURACY))
        if stay_stay_omission > CUED_TS_OMISSION_RATE:
            violations.append(('task_stay_cue_stay_omission_rate', stay_stay_omission, CUED_TS_OMISSION_RATE))
        if stay_switch_omission > CUED_TS_OMISSION_RATE:
            violations.append(('task_stay_cue_switch_omission_rate', stay_switch_omission, CUED_TS_OMISSION_RATE))
        if switch_switch_omission > CUED_TS_OMISSION_RATE:
            violations.append(('task_switch_cue_switch_omission_rate', switch_switch_omission, CUED_TS_OMISSION_RATE))

    elif task_name == 'flanker':
        congruent_accuracy = metrics_dict.get('congruent_accuracy', 0)
        incongruent_accuracy = metrics_dict.get('incongruent_accuracy', 0)
        congruent_omission = metrics_dict.get('congruent_omission_rate', 0)
        incongruent_omission = metrics_dict.get('incongruent_omission_rate', 0)

        if congruent_accuracy < FLANKER_ACCURACY:
            violations.append(('congruent_accuracy', congruent_accuracy, FLANKER_ACCURACY))
        if incongruent_accuracy < FLANKER_ACCURACY:
            violations.append(('incongruent_accuracy', incongruent_accuracy, FLANKER_ACCURACY))
        if congruent_omission > FLANKER_OMISSION_RATE:
            violations.append(('congruent_omission_rate', congruent_omission, FLANKER_OMISSION_RATE))
        if incongruent_omission > FLANKER_OMISSION_RATE:
            violations.append(('incongruent_omission_rate', incongruent_omission, FLANKER_OMISSION_RATE))

    elif task_name == 'nBack':
        match_1_accuracy = metrics_dict.get('match_1_accuracy', 0)
        mismatch_1_accuracy = metrics_dict.get('mismatch_1_accuracy', 0)
        match_2_accuracy = metrics_dict.get('match_2_accuracy', 0)
        mismatch_2_accuracy = metrics_dict.get('mismatch_2_accuracy', 0)

        weighted_accuracy_1 = (NBACK_MATCH_WEIGHT * match_1_accuracy + 
                             NBACK_MISMATCH_WEIGHT * mismatch_1_accuracy)
        weighted_accuracy_2 = (NBACK_MATCH_WEIGHT * match_2_accuracy + 
                             NBACK_MISMATCH_WEIGHT * mismatch_2_accuracy)

        if weighted_accuracy_1 < NBACK_WEIGHTED_ACCURACY:
            violations.append(('weighted_accuracy_1', weighted_accuracy_1, NBACK_WEIGHTED_ACCURACY))
        if weighted_accuracy_2 < NBACK_WEIGHTED_ACCURACY:
            violations.append(('weighted_accuracy_2', weighted_accuracy_2, NBACK_WEIGHTED_ACCURACY))
        if mismatch_1_accuracy < NBACK_MISMATCH_ACCURACY and match_1_accuracy < NBACK_MATCH_ACCURACY:
            violations.append(('mismatch_1_accuracy', mismatch_1_accuracy, NBACK_MISMATCH_ACCURACY))
        if mismatch_2_accuracy < NBACK_MISMATCH_ACCURACY and match_2_accuracy < NBACK_MATCH_ACCURACY:
            violations.append(('mismatch_2_accuracy', mismatch_2_accuracy, NBACK_MISMATCH_ACCURACY))

    elif task_name == 'spatialCueing':
        doublecue_accuracy = metrics_dict.get('doublecue_accuracy', 0)
        doublecue_omission = metrics_dict.get('doublecue_omission_rate', 0)
        invalid_accuracy = metrics_dict.get('invalid_accuracy', 0)
        invalid_omission = metrics_dict.get('invalid_omission_rate', 0)
        nocue_accuracy = metrics_dict.get('nocue_accuracy', 0)
        nocue_omission = metrics_dict.get('nocue_omission_rate', 0)
        valid_accuracy = metrics_dict.get('valid_accuracy', 0)
        valid_omission = metrics_dict.get('valid_omission_rate', 0)

        if doublecue_accuracy < SPATIAL_CUEING_ACCURACY:
            violations.append(('doublecue_accuracy', doublecue_accuracy, SPATIAL_CUEING_ACCURACY))
        if doublecue_omission > SPATIAL_CUEING_OMISSION_RATE:
            violations.append(('doublecue_omission_rate', doublecue_omission, SPATIAL_CUEING_OMISSION_RATE))
        if invalid_accuracy < SPATIAL_CUEING_ACCURACY:
            violations.append(('invalid_accuracy', invalid_accuracy, SPATIAL_CUEING_ACCURACY))
        if invalid_omission > SPATIAL_CUEING_OMISSION_RATE:
            violations.append(('invalid_omission_rate', invalid_omission, SPATIAL_CUEING_OMISSION_RATE))
        if nocue_accuracy < SPATIAL_CUEING_ACCURACY:
            violations.append(('nocue_accuracy', nocue_accuracy, SPATIAL_CUEING_ACCURACY))
        if nocue_omission > SPATIAL_CUEING_OMISSION_RATE:
            violations.append(('nocue_omission_rate', nocue_omission, SPATIAL_CUEING_OMISSION_RATE))
        if valid_accuracy < SPATIAL_CUEING_ACCURACY:
            violations.append(('valid_accuracy', valid_accuracy, SPATIAL_CUEING_ACCURACY))
        if valid_omission > SPATIAL_CUEING_OMISSION_RATE:
            violations.append(('valid_omission_rate', valid_omission, SPATIAL_CUEING_OMISSION_RATE))

    elif task_name == 'spatialTS':
        stay_stay_accuracy = metrics_dict.get('task_stay_cue_stay_accuracy', 0)
        stay_switch_accuracy = metrics_dict.get('task_stay_cue_switch_accuracy', 0)
        switch_switch_accuracy = metrics_dict.get('task_switch_cue_switch_accuracy', 0)
        stay_stay_omission = metrics_dict.get('task_stay_cue_stay_omission_rate', 0)
        stay_switch_omission = metrics_dict.get('task_stay_cue_switch_omission_rate', 0)
        switch_switch_omission = metrics_dict.get('task_switch_cue_switch_omission_rate', 0)

        if stay_stay_accuracy < SPATIAL_TS_ACCURACY:
            violations.append(('task_stay_cue_stay_accuracy', stay_stay_accuracy, SPATIAL_TS_ACCURACY))
        if stay_switch_accuracy < SPATIAL_TS_ACCURACY:
            violations.append(('task_stay_cue_switch_accuracy', stay_switch_accuracy, SPATIAL_TS_ACCURACY))
        if switch_switch_accuracy < SPATIAL_TS_ACCURACY:
            violations.append(('task_switch_cue_switch_accuracy', switch_switch_accuracy, SPATIAL_TS_ACCURACY))
        if stay_stay_omission > SPATIAL_TS_OMISSION_RATE:
            violations.append(('task_stay_cue_stay_omission_rate', stay_stay_omission, SPATIAL_TS_OMISSION_RATE))
        if stay_switch_omission > SPATIAL_TS_OMISSION_RATE:
            violations.append(('task_stay_cue_switch_omission_rate', stay_switch_omission, SPATIAL_TS_OMISSION_RATE))
        if switch_switch_omission > SPATIAL_TS_OMISSION_RATE:
            violations.append(('task_switch_cue_switch_omission_rate', switch_switch_omission, SPATIAL_TS_OMISSION_RATE))

    elif task_name == 'stroop':
        congruent_accuracy = metrics_dict.get('congruent_accuracy', 0)
        incongruent_accuracy = metrics_dict.get('incongruent_accuracy', 0)
        congruent_omission = metrics_dict.get('congruent_omission_rate', 0)
        incongruent_omission = metrics_dict.get('incongruent_omission_rate', 0)

        if congruent_accuracy < STROOP_ACCURACY:
            violations.append(('congruent_accuracy', congruent_accuracy, STROOP_ACCURACY))
        if incongruent_accuracy < STROOP_ACCURACY:
            violations.append(('incongruent_accuracy', incongruent_accuracy, STROOP_ACCURACY))
        if congruent_omission > STROOP_OMISSION_RATE:
            violations.append(('congruent_omission_rate', congruent_omission, STROOP_OMISSION_RATE))
        if incongruent_omission > STROOP_OMISSION_RATE:
            violations.append(('incongruent_omission_rate', incongruent_omission, STROOP_OMISSION_RATE))

    elif task_name == 'visualSearch':
        conjunction_24_accuracy = metrics_dict.get('conjunction_24_accuracy', 0)
        conjunction_8_accuracy = metrics_dict.get('conjunction_8_accuracy', 0)
        feature_24_accuracy = metrics_dict.get('feature_24_accuracy', 0)
        feature_8_accuracy = metrics_dict.get('feature_8_accuracy', 0)
        conjunction_24_omission = metrics_dict.get('conjunction_24_omission_rate', 0)
        conjunction_8_omission = metrics_dict.get('conjunction_8_omission_rate', 0)
        feature_24_omission = metrics_dict.get('feature_24_omission_rate', 0)
        feature_8_omission = metrics_dict.get('feature_8_omission_rate', 0)

        if conjunction_24_accuracy < VISUAL_SEARCH_ACCURACY:
            violations.append(('conjunction_24_accuracy', conjunction_24_accuracy, VISUAL_SEARCH_ACCURACY))
        if conjunction_8_accuracy < VISUAL_SEARCH_ACCURACY:
            violations.append(('conjunction_8_accuracy', conjunction_8_accuracy, VISUAL_SEARCH_ACCURACY))
        if feature_24_accuracy < VISUAL_SEARCH_ACCURACY:
            violations.append(('feature_24_accuracy', feature_24_accuracy, VISUAL_SEARCH_ACCURACY))
        if feature_8_accuracy < VISUAL_SEARCH_ACCURACY:
            violations.append(('feature_8_accuracy', feature_8_accuracy, VISUAL_SEARCH_ACCURACY))
        if conjunction_24_omission > VISUAL_SEARCH_OMISSION_RATE:
            violations.append(('conjunction_24_omission_rate', conjunction_24_omission, VISUAL_SEARCH_OMISSION_RATE))
        if conjunction_8_omission > VISUAL_SEARCH_OMISSION_RATE:
            violations.append(('conjunction_8_omission_rate', conjunction_8_omission, VISUAL_SEARCH_OMISSION_RATE))
        if feature_24_omission > VISUAL_SEARCH_OMISSION_RATE:
            violations.append(('feature_24_omission_rate', feature_24_omission, VISUAL_SEARCH_OMISSION_RATE))
        if feature_8_omission > VISUAL_SEARCH_OMISSION_RATE:
            violations.append(('feature_8_omission_rate', feature_8_omission, VISUAL_SEARCH_OMISSION_RATE))

    elif task_name in ['OpSpan', 'opOnly']:
        asymmetric_accuracy = metrics_dict.get('8x8_grid_asymmetric_accuracy', 0)
        symmetric_accuracy = metrics_dict.get('8x8_grid_symmetric_accuracy', 0)
        mean_4x4_accuracy = metrics_dict.get('mean_4x4_grid_accuracy_irrespective_of_order', 0)
        mean_4x4_accuracy_respective = metrics_dict.get('mean_4x4_grid_accuracy_respective_of_order', 0)

        if asymmetric_accuracy < OP_SPAN_ASYMMETRIC_ACCURACY:
            violations.append(('8x8_grid_asymmetric_accuracy', asymmetric_accuracy, OP_SPAN_ASYMMETRIC_ACCURACY))
        if symmetric_accuracy < OP_SPAN_SYMMETRIC_ACCURACY:
            violations.append(('8x8_grid_symmetric_accuracy', symmetric_accuracy, OP_SPAN_SYMMETRIC_ACCURACY))
        if task_name == 'OpSpan':
            if mean_4x4_accuracy < OP_SPAN_4X4_ACCURACY:
                violations.append(('mean_4x4_grid_accuracy_irrespective_of_order', mean_4x4_accuracy, OP_SPAN_4X4_ACCURACY))
            if mean_4x4_accuracy_respective > mean_4x4_accuracy - OP_SPAN_ORDER_DIFF:
                violations.append(('mean_4x4_grid_accuracy_respective_of_order', mean_4x4_accuracy_respective, mean_4x4_accuracy - OP_SPAN_ORDER_DIFF))

    elif task_name == 'simpleSpan':
        mean_4x4_accuracy = metrics_dict.get('mean_4x4_grid_accuracy_irrespective_of_order', 0)
        mean_4x4_accuracy_respective = metrics_dict.get('mean_4x4_grid_accuracy_respective_of_order', 0)

        if mean_4x4_accuracy < SIMPLE_SPAN_4X4_ACCURACY:
            violations.append(('mean_4x4_grid_accuracy_irrespective_of_order', mean_4x4_accuracy, SIMPLE_SPAN_4X4_ACCURACY))
        if mean_4x4_accuracy_respective > mean_4x4_accuracy - SIMPLE_SPAN_ORDER_DIFF:
            violations.append(('mean_4x4_grid_accuracy_respective_of_order', mean_4x4_accuracy_respective, mean_4x4_accuracy - SIMPLE_SPAN_ORDER_DIFF))

    return violations

# Map directory names to analysis functions
TASK_MAPPING = {
    'nBack': n_back_rdoc,
    'flanker': flanker_rdoc,
    'cuedTS': cued_task_switching_rdoc,
    'spatialCueing': spatial_cueing_rdoc,
    'spatialTS': spatial_task_switching_rdoc,
    'stroop': stroop_rdoc,
    'visualSearch': visual_search_rdoc,
    'axCPT': ax_cpt_rdoc,
    'goNogo': go_nogo_rdoc,
    'simpleSpan': simple_span_rdoc,
    'stopSignal': stop_signal_rdoc,
    'OpSpan': operation_span_rdoc,
    'opOnly': operation_span_rdoc,
}

def main():
    # Check if subject folder is provided
    if len(sys.argv) != 2:
        print("Error: Please provide a subject folder name")
        print("Usage: python analyze_behavioral_data.py <subject_folder>")
        print("Example: python analyze_behavioral_data.py sub-SK")
        sys.exit(1)

    subject_folder = sys.argv[1]
    
    # Configure logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    # Set up input directory
    preprocessed_dir = os.path.join('preprocessed_data', subject_folder)

    # Check if directory exists
    if not os.path.exists(preprocessed_dir):
        logging.error(f'Directory {preprocessed_dir} does not exist')
        sys.exit(1)

    # Get all task directories
    task_dirs = [d for d in os.listdir(preprocessed_dir) 
                if os.path.isdir(os.path.join(preprocessed_dir, d))]

    # Prepare output directory
    output_dir = os.path.join('outputs', subject_folder)
    os.makedirs(output_dir, exist_ok=True)

    # Process each task
    all_metrics = []
    all_flags = []
    for task_dir in task_dirs:
        if task_dir not in TASK_MAPPING:
            logging.warning(f'Unknown task: {task_dir}')
            continue

        # Get all parquet files for this task
        task_path = os.path.join(preprocessed_dir, task_dir)
        parquet_files = [f for f in os.listdir(task_path) if f.endswith('.parquet')]

        if not parquet_files:
            logging.warning(f'No parquet files found in {task_path}')
            continue

        task_metrics_list = []
        # Process each file
        for parquet_file in parquet_files:
            file_path = os.path.join(task_path, parquet_file)
            try:
                # Read the parquet file
                df = pl.read_parquet(file_path)
                
                # Get the analysis function for this task
                analysis_func = TASK_MAPPING[task_dir]
                
                # Calculate metrics
                metrics = analysis_func(df)
                
                # Add task and file information
                metrics = metrics.with_columns([
                    pl.lit(task_dir).alias('task'),
                    pl.lit(parquet_file).alias('file')
                ])
                
                all_metrics.append(metrics)
                task_metrics_list.append(metrics)
                
            except Exception as e:
                logging.error(f'Error processing {file_path}: {str(e)}')
                continue

        # Save all metrics for this task to CSV
        if task_metrics_list:
            task_metrics_df = pl.concat(task_metrics_list)
            metrics_csv_path = os.path.join(output_dir, f'{task_dir}_metrics.csv')
            task_metrics_df.write_csv(metrics_csv_path)
            logging.info(f'Metrics for {task_dir} saved to {metrics_csv_path}')

        # Check thresholds for this task using only its own metrics CSV
        metrics_csv_path = os.path.join(output_dir, f'{task_dir}_metrics.csv')
        if os.path.exists(metrics_csv_path):
            try:
                task_metrics_df = pl.read_csv(metrics_csv_path)
                violations = check_thresholds_from_csv(task_metrics_df, task_dir)
                if violations:
                    logging.warning(f'Threshold violations found for {task_dir}:')
                    print(violations)
                    # Add task name to each flag for clarity
                    all_flags.extend([(task_dir, metric, value, threshold) for metric, value, threshold in violations])
            except Exception as e:
                logging.error(f'Error checking thresholds for {task_dir}: {str(e)}')
                continue

    # Save flags
    if all_flags:
        # Ensure each violation has the expected number of elements
        valid_flags = [flag for flag in all_flags if len(flag) == 4]
        # Convert the list of violations into a DataFrame
        combined_flags = pl.DataFrame(
            [(str(task), str(metric), str(value), str(threshold)) for task, metric, value, threshold in valid_flags],
            schema=['task', 'metric', 'value', 'threshold']
        )
        # Save flags as CSV
        output_file = os.path.join(output_dir, f'{subject_folder}_flags.csv')
        combined_flags.write_csv(output_file)
        logging.info(f'Flags saved to {output_file}')
    else:
        # If no flags, create an empty DataFrame with headers
        empty_flags = pl.DataFrame({'task': [], 'metric': [], 'value': [], 'threshold': []})
        output_file = os.path.join(output_dir, f'{subject_folder}_flags.csv')
        empty_flags.write_csv(output_file)
        logging.info(f'No flags were generated. Empty flags file saved to {output_file}')

if __name__ == '__main__':
    main() 