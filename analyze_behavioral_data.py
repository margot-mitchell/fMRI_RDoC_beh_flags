import os
import glob
import sys
import pandas as pd
from thresholds_config import (
    # General thresholds
    ACCURACY_THRESHOLD,
    OMISSION_THRESHOLD,
    # Stop Signal thresholds
    STOP_SIGNAL_ACCURACY_MIN,
    STOP_SIGNAL_ACCURACY_MAX,
    STOP_SIGNAL_GO_ACCURACY,
    STOP_SIGNAL_GO_RT,
    # AX-CPT thresholds
    AX_CPT_ACCURACY,
    # Go/NoGo thresholds
    GONOGO_GO_ACCURACY,
    GONOGO_NOGO_ACCURACY,
    GONOGO_MEAN_ACCURACY,
    GONOGO_GO_OMISSION,
    # Flanker thresholds
    FLANKER_ACCURACY,
    FLANKER_OMISSION,
    # Operation Span thresholds
    OP_SPAN_ASYMMETRIC_ACCURACY,
    OP_SPAN_SYMMETRIC_ACCURACY,
    OP_SPAN_4X4_ACCURACY,
    OP_SPAN_ORDER_DIFF,
    # Simple Span thresholds
    SIMPLE_SPAN_ASYMMETRIC_ACCURACY,
    SIMPLE_SPAN_SYMMETRIC_ACCURACY,
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
    CUED_TS_OMISSION,
    # Spatial cueing thresholds
    SPATIAL_CUEING_ACCURACY,
    SPATIAL_CUEING_OMISSION,
    # Spatial TS thresholds
    SPATIAL_TS_ACCURACY,
    SPATIAL_TS_OMISSION,
    # Stroop thresholds
    STROOP_ACCURACY,
    STROOP_OMISSION,
    # Visual search thresholds
    VISUAL_SEARCH_ACCURACY,
    VISUAL_SEARCH_OMISSION
)

def check_span_metrics(df, task_name, flags_rows, csv_file):
    """Check span task specific metrics and thresholds."""
    # Get the relevant metrics
    if 'mean_4x4_grid_accuracy_irrespective_of_order' in df.columns:
        irresp_acc = df['mean_4x4_grid_accuracy_irrespective_of_order'].iloc[0]
        resp_acc = df['mean_4x4_grid_accuracy_respective_of_order'].iloc[0]
        
        if task_name == 'operationSpan':
            # Check operation span thresholds
            if irresp_acc < OP_SPAN_4X4_ACCURACY:
                flags_rows.append({
                    'task_file': os.path.basename(csv_file),
                    'task': task_name,
                    'condition': 'all',
                    'metric': '4x4_grid_accuracy_irrespective_of_order',
                    'value': irresp_acc,
                    'threshold': OP_SPAN_4X4_ACCURACY,
                    'status': 'exclude'
                })
            
            # Check order difference threshold
            order_diff = irresp_acc - resp_acc
            if order_diff > OP_SPAN_ORDER_DIFF:
                flags_rows.append({
                    'task_file': os.path.basename(csv_file),
                    'task': task_name,
                    'condition': 'all',
                    'metric': '4x4_grid_order_difference',
                    'value': order_diff,
                    'threshold': OP_SPAN_ORDER_DIFF,
                    'status': 'exclude'
                })
                
        elif task_name == 'simpleSpan':
            # Check simple span thresholds
            if irresp_acc < SIMPLE_SPAN_4X4_ACCURACY:
                flags_rows.append({
                    'task_file': os.path.basename(csv_file),
                    'task': task_name,
                    'condition': 'all',
                    'metric': '4x4_grid_accuracy_irrespective_of_order',
                    'value': irresp_acc,
                    'threshold': SIMPLE_SPAN_4X4_ACCURACY,
                    'status': 'exclude'
                })
            
            # Check order difference threshold
            order_diff = irresp_acc - resp_acc
            if order_diff > SIMPLE_SPAN_ORDER_DIFF:
                flags_rows.append({
                    'task_file': os.path.basename(csv_file),
                    'task': task_name,
                    'condition': 'all',
                    'metric': '4x4_grid_order_difference',
                    'value': order_diff,
                    'threshold': SIMPLE_SPAN_ORDER_DIFF,
                    'status': 'exclude'
                })
    
    return flags_rows

def check_task_specific_metrics(df, task_name, cond, group, flags_rows, csv_file):
    """Check task-specific metrics and thresholds."""
    if task_name == 'stopSignal':
        # Check stop accuracy thresholds
        if cond == 'stop':
            accuracy = group['correct_trial'].mean() if 'correct_trial' in group.columns else group['success'].mean()
            if accuracy < STOP_SIGNAL_ACCURACY_MIN or accuracy > STOP_SIGNAL_ACCURACY_MAX:
                flags_rows.append({
                    'task_file': os.path.basename(csv_file),
                    'task': task_name,
                    'condition': cond,
                    'metric': 'stop_trial_accuracy',
                    'value': accuracy,
                    'threshold': f'<{STOP_SIGNAL_ACCURACY_MIN} or >{STOP_SIGNAL_ACCURACY_MAX}',
                    'status': 'exclude'
                })
        # Check go accuracy threshold
        elif cond == 'go':
            accuracy = group['correct_trial'].mean() if 'correct_trial' in group.columns else group['success'].mean()
            if accuracy < STOP_SIGNAL_GO_ACCURACY:
                flags_rows.append({
                    'task_file': os.path.basename(csv_file),
                    'task': task_name,
                    'condition': cond,
                    'metric': 'go_trial_accuracy',
                    'value': accuracy,
                    'threshold': STOP_SIGNAL_GO_ACCURACY,
                    'status': 'exclude'
                })
            # Check go RT threshold
            if 'rt' in group.columns:
                mean_go_rt = group['rt'].mean()
                if mean_go_rt > STOP_SIGNAL_GO_RT:
                    flags_rows.append({
                        'task_file': os.path.basename(csv_file),
                        'task': task_name,
                        'condition': cond,
                        'metric': 'go_trial_rt',
                        'value': mean_go_rt,
                        'threshold': STOP_SIGNAL_GO_RT,
                        'status': 'exclude'
                    })
    
    elif task_name == 'axCpt':
        accuracy = group['correct_trial'].mean() if 'correct_trial' in group.columns else group['success'].mean()
        if accuracy < AX_CPT_ACCURACY:
            flags_rows.append({
                'task_file': os.path.basename(csv_file),
                'task': task_name,
                'condition': cond,
                'metric': f'{cond}_accuracy',  # e.g., 'AX_accuracy', 'BX_accuracy', etc.
                'value': accuracy,
                'threshold': AX_CPT_ACCURACY,
                'status': 'exclude'
            })
    
    elif task_name == 'gonogo':
        if cond == 'go':
            accuracy = group['correct_trial'].mean() if 'correct_trial' in group.columns else group['success'].mean()
            if accuracy < GONOGO_GO_ACCURACY:
                flags_rows.append({
                    'task_file': os.path.basename(csv_file),
                    'task': task_name,
                    'condition': cond,
                    'metric': 'go_trial_accuracy',
                    'value': accuracy,
                    'threshold': GONOGO_GO_ACCURACY,
                    'status': 'exclude'
                })
            # Check go omission rate
            if 'rt' in group.columns:
                omission_rate = group['rt'].isna().mean()
                if omission_rate > GONOGO_GO_OMISSION:
                    flags_rows.append({
                        'task_file': os.path.basename(csv_file),
                        'task': task_name,
                        'condition': cond,
                        'metric': 'go_trial_omission_rate',
                        'value': omission_rate,
                        'threshold': GONOGO_GO_OMISSION,
                        'status': 'exclude'
                    })
        elif cond == 'nogo':
            accuracy = group['correct_trial'].mean() if 'correct_trial' in group.columns else group['success'].mean()
            if accuracy > GONOGO_NOGO_ACCURACY:  # Note: for nogo, we want accuracy to be LOW
                flags_rows.append({
                    'task_file': os.path.basename(csv_file),
                    'task': task_name,
                    'condition': cond,
                    'metric': 'nogo_trial_accuracy',
                    'value': accuracy,
                    'threshold': GONOGO_NOGO_ACCURACY,
                    'status': 'exclude'
                })
        
        # Check mean accuracy across conditions
        if cond == 'all':
            mean_accuracy = df['correct_trial'].mean() if 'correct_trial' in df.columns else df['success'].mean()
            if mean_accuracy < GONOGO_MEAN_ACCURACY:
                flags_rows.append({
                    'task_file': os.path.basename(csv_file),
                    'task': task_name,
                    'condition': cond,
                    'metric': 'overall_mean_accuracy',
                    'value': mean_accuracy,
                    'threshold': GONOGO_MEAN_ACCURACY,
                    'status': 'exclude'
                })
    
    elif task_name == 'flanker':
        accuracy = group['correct_trial'].mean() if 'correct_trial' in group.columns else group['success'].mean()
        if accuracy < FLANKER_ACCURACY:
            flags_rows.append({
                'task_file': os.path.basename(csv_file),
                'task': task_name,
                'condition': cond,
                'metric': f'{cond}_accuracy',
                'value': accuracy,
                'threshold': FLANKER_ACCURACY,
                'status': 'exclude'
            })
        # Check omission rate
        if 'rt' in group.columns:
            omission_rate = group['rt'].isna().mean()
            if omission_rate > FLANKER_OMISSION:
                flags_rows.append({
                    'task_file': os.path.basename(csv_file),
                    'task': task_name,
                    'condition': cond,
                    'metric': f'{cond}_omission_rate',
                    'value': omission_rate,
                    'threshold': FLANKER_OMISSION,
                    'status': 'exclude'
                })
    
    elif task_name == 'cuedTS':
        accuracy = group['correct_trial'].mean() if 'correct_trial' in group.columns else group['success'].mean()
        if accuracy < CUED_TS_ACCURACY:
            flags_rows.append({
                'task_file': os.path.basename(csv_file),
                'task': task_name,
                'condition': cond,
                'metric': f'{cond}_accuracy',
                'value': accuracy,
                'threshold': CUED_TS_ACCURACY,
                'status': 'exclude'
            })
        if 'rt' in group.columns:
            omission_rate = group['rt'].isna().mean()
            if omission_rate > CUED_TS_OMISSION:
                flags_rows.append({
                    'task_file': os.path.basename(csv_file),
                    'task': task_name,
                    'condition': cond,
                    'metric': f'{cond}_omission_rate',
                    'value': omission_rate,
                    'threshold': CUED_TS_OMISSION,
                    'status': 'exclude'
                })
    
    elif task_name == 'nBack':
        # Check 1-back weighted accuracy
        if 'match_1_accuracy' in df.columns and 'mismatch_1_accuracy' in df.columns:
            match_1_acc = df['match_1_accuracy'].iloc[0]
            mismatch_1_acc = df['mismatch_1_accuracy'].iloc[0]
            weighted_1_acc = (match_1_acc * NBACK_MATCH_WEIGHT + 
                            mismatch_1_acc * NBACK_MISMATCH_WEIGHT)
            
            if weighted_1_acc < NBACK_WEIGHTED_ACCURACY:
                flags_rows.append({
                    'task_file': os.path.basename(csv_file),
                    'task': task_name,
                    'condition': '1back',
                    'metric': 'weighted_accuracy',
                    'value': weighted_1_acc,
                    'threshold': NBACK_WEIGHTED_ACCURACY,
                    'status': 'exclude'
                })
            
            if mismatch_1_acc < NBACK_MISMATCH_ACCURACY and match_1_acc < NBACK_MATCH_ACCURACY:
                flags_rows.append({
                    'task_file': os.path.basename(csv_file),
                    'task': task_name,
                    'condition': '1back',
                    'metric': 'match_mismatch_accuracy',
                    'value': f'match={match_1_acc}, mismatch={mismatch_1_acc}',
                    'threshold': f'match<{NBACK_MATCH_ACCURACY} AND mismatch<{NBACK_MISMATCH_ACCURACY}',
                    'status': 'exclude'
                })
        
        # Check 2-back weighted accuracy
        if 'match_2_accuracy' in df.columns and 'mismatch_2_accuracy' in df.columns:
            match_2_acc = df['match_2_accuracy'].iloc[0]
            mismatch_2_acc = df['mismatch_2_accuracy'].iloc[0]
            weighted_2_acc = (match_2_acc * NBACK_MATCH_WEIGHT + 
                            mismatch_2_acc * NBACK_MISMATCH_WEIGHT)
            
            if weighted_2_acc < NBACK_WEIGHTED_ACCURACY:
                flags_rows.append({
                    'task_file': os.path.basename(csv_file),
                    'task': task_name,
                    'condition': '2back',
                    'metric': 'weighted_accuracy',
                    'value': weighted_2_acc,
                    'threshold': NBACK_WEIGHTED_ACCURACY,
                    'status': 'exclude'
                })
            
            if mismatch_2_acc < NBACK_MISMATCH_ACCURACY and match_2_acc < NBACK_MATCH_ACCURACY:
                flags_rows.append({
                    'task_file': os.path.basename(csv_file),
                    'task': task_name,
                    'condition': '2back',
                    'metric': 'match_mismatch_accuracy',
                    'value': f'match={match_2_acc}, mismatch={mismatch_2_acc}',
                    'threshold': f'match<{NBACK_MATCH_ACCURACY} AND mismatch<{NBACK_MISMATCH_ACCURACY}',
                    'status': 'exclude'
                })
    
    elif task_name == 'spatialCueing':
        accuracy = group['correct_trial'].mean() if 'correct_trial' in group.columns else group['success'].mean()
        if accuracy < SPATIAL_CUEING_ACCURACY:
            flags_rows.append({
                'task_file': os.path.basename(csv_file),
                'task': task_name,
                'condition': cond,
                'metric': f'{cond}_accuracy',
                'value': accuracy,
                'threshold': SPATIAL_CUEING_ACCURACY,
                'status': 'exclude'
            })
        if 'rt' in group.columns:
            omission_rate = group['rt'].isna().mean()
            if omission_rate > SPATIAL_CUEING_OMISSION:
                flags_rows.append({
                    'task_file': os.path.basename(csv_file),
                    'task': task_name,
                    'condition': cond,
                    'metric': f'{cond}_omission_rate',
                    'value': omission_rate,
                    'threshold': SPATIAL_CUEING_OMISSION,
                    'status': 'exclude'
                })
    
    elif task_name == 'spatialTS':
        accuracy = group['correct_trial'].mean() if 'correct_trial' in group.columns else group['success'].mean()
        if accuracy < SPATIAL_TS_ACCURACY:
            flags_rows.append({
                'task_file': os.path.basename(csv_file),
                'task': task_name,
                'condition': cond,
                'metric': f'{cond}_accuracy',
                'value': accuracy,
                'threshold': SPATIAL_TS_ACCURACY,
                'status': 'exclude'
            })
        if 'rt' in group.columns:
            omission_rate = group['rt'].isna().mean()
            if omission_rate > SPATIAL_TS_OMISSION:
                flags_rows.append({
                    'task_file': os.path.basename(csv_file),
                    'task': task_name,
                    'condition': cond,
                    'metric': f'{cond}_omission_rate',
                    'value': omission_rate,
                    'threshold': SPATIAL_TS_OMISSION,
                    'status': 'exclude'
                })
    
    elif task_name == 'stroop':
        accuracy = group['correct_trial'].mean() if 'correct_trial' in group.columns else group['success'].mean()
        if accuracy < STROOP_ACCURACY:
            flags_rows.append({
                'task_file': os.path.basename(csv_file),
                'task': task_name,
                'condition': cond,
                'metric': f'{cond}_accuracy',
                'value': accuracy,
                'threshold': STROOP_ACCURACY,
                'status': 'exclude'
            })
        if 'rt' in group.columns:
            omission_rate = group['rt'].isna().mean()
            if omission_rate > STROOP_OMISSION:
                flags_rows.append({
                    'task_file': os.path.basename(csv_file),
                    'task': task_name,
                    'condition': cond,
                    'metric': f'{cond}_omission_rate',
                    'value': omission_rate,
                    'threshold': STROOP_OMISSION,
                    'status': 'exclude'
                })
    
    elif task_name == 'visualSearch':
        accuracy = group['correct_trial'].mean() if 'correct_trial' in group.columns else group['success'].mean()
        if accuracy < VISUAL_SEARCH_ACCURACY:
            flags_rows.append({
                'task_file': os.path.basename(csv_file),
                'task': task_name,
                'condition': cond,
                'metric': f'{cond}_accuracy',
                'value': accuracy,
                'threshold': VISUAL_SEARCH_ACCURACY,
                'status': 'exclude'
            })
        if 'rt' in group.columns:
            omission_rate = group['rt'].isna().mean()
            if omission_rate > VISUAL_SEARCH_OMISSION:
                flags_rows.append({
                    'task_file': os.path.basename(csv_file),
                    'task': task_name,
                    'condition': cond,
                    'metric': f'{cond}_omission_rate',
                    'value': omission_rate,
                    'threshold': VISUAL_SEARCH_OMISSION,
                    'status': 'exclude'
                })
    
    return flags_rows

def main():
    # Check if subject folder is provided
    if len(sys.argv) != 2:
        print("Error: Please provide a subject folder name")
        print("Usage: python analyze_behavioral_data.py <subject_folder>")
        print("Example: python analyze_behavioral_data.py sub-SK")
        sys.exit(1)

    subject_folder = sys.argv[1]
    
    # Directory containing preprocessed CSVs
    PREPROCESSED_DIR = os.path.join('preprocessed_data', subject_folder)

    # Create outputs directory if it doesn't exist
    OUTPUT_DIR = os.path.join('outputs', subject_folder)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define output file paths
    OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'behavioral_summary.csv')
    FLAGS_CSV = os.path.join(OUTPUT_DIR, f'{subject_folder}_flags.csv')

    # Possible condition columns to check for
    CONDITION_COLS = ['condition', 'trial_type', 'task_condition', 'cue_condition', 'delay', 'grid_symmetry', 'num_stimuli']

    summary_rows = []
    flags_rows = []

    for task_dir in glob.glob(os.path.join(PREPROCESSED_DIR, '*')):
        for csv_file in glob.glob(os.path.join(task_dir, '*.csv')):
            df = pd.read_csv(csv_file)
            task_name = os.path.basename(task_dir)
            
            # Try to find a condition column
            condition_col = None
            for col in CONDITION_COLS:
                if col in df.columns:
                    condition_col = col
                    break
            if condition_col is None:
                # If no condition column, treat all as one group
                df['__all__'] = 'all'
                condition_col = '__all__'
                
            # Compute metrics per condition
            for cond, group in df.groupby(condition_col):
                # Accuracy: mean of correct_trial (if present)
                if 'correct_trial' in group.columns:
                    accuracy = group['correct_trial'].mean()
                elif 'success' in group.columns:
                    accuracy = group['success'].mean()
                else:
                    accuracy = float('nan')
                    
                # Omission rate: mean of missing RTs (if rt column exists)
                if 'rt' in group.columns:
                    omission_rate = group['rt'].isna().mean()
                else:
                    omission_rate = float('nan')
                
                # Add to summary
                summary_rows.append({
                    'task_file': os.path.basename(csv_file),
                    'task': task_name,
                    'condition': cond,
                    'accuracy': accuracy,
                    'omission_rate': omission_rate
                })
                
                # Check task-specific metrics
                flags_rows = check_task_specific_metrics(df, task_name, cond, group, flags_rows, csv_file)
            
            # Check span task specific metrics
            if task_name in ['operationSpan', 'simpleSpan']:
                flags_rows = check_span_metrics(df, task_name, flags_rows, csv_file)

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows)
    # Pivot so each row is a file, and columns are accuracy/omission for each condition
    summary_pivot = summary_df.pivot_table(
        index=['task_file', 'task'],
        columns='condition',
        values=['accuracy', 'omission_rate']
    )
    # Flatten columns
    summary_pivot.columns = [f'{metric}_{cond}' for metric, cond in summary_pivot.columns]
    summary_pivot = summary_pivot.reset_index()
    # Save to CSV
    summary_pivot.to_csv(OUTPUT_CSV, index=False)
    print(f'Summary saved to {OUTPUT_CSV}')

    # Create and save flags DataFrame
    if flags_rows:
        flags_df = pd.DataFrame(flags_rows)
        flags_df.to_csv(FLAGS_CSV, index=False)
        print(f'Flags saved to {FLAGS_CSV}')
        print(f'Found {len(flags_rows)} quality control flags')
    else:
        # Create empty DataFrame with the same columns
        flags_df = pd.DataFrame(columns=[
            'task_file',
            'task',
            'condition',
            'metric',
            'value',
            'threshold',
            'status'
        ])
        flags_df.to_csv(FLAGS_CSV, index=False)
        print(f'No quality control flags found. Created empty flags file at {FLAGS_CSV}')

if __name__ == '__main__':
    main() 