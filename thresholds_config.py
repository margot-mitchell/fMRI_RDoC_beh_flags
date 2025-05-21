"""Configuration file containing thresholds for behavioral data quality control.

This module defines thresholds for flagging behavioral data based on various metrics
such as accuracy, reaction times, and omission rates for different tasks.
"""

# General thresholds
ACCURACY_THRESHOLD = 0.55  # General accuracy threshold
OMISSION_THRESHOLD = 0.50  # General omission rate threshold

# Stop Signal Task thresholds
STOP_SIGNAL_ACCURACY_MIN = 0.25  # Minimum stop accuracy
STOP_SIGNAL_ACCURACY_MAX = 0.75  # Maximum stop accuracy
STOP_SIGNAL_GO_ACCURACY = 0.55   # Minimum go accuracy
STOP_SIGNAL_GO_RT = 750          # Maximum go RT in milliseconds

# AX-CPT Task thresholds
AX_CPT_ACCURACY = 0.55           # Minimum accuracy for any condition

# Go/NoGo Task thresholds
GONOGO_GO_ACCURACY = 0.857       # Minimum go accuracy
GONOGO_NOGO_ACCURACY = 0.143     # Maximum nogo accuracy (to avoid false alarms)
GONOGO_MEAN_ACCURACY = 0.55      # Minimum mean accuracy
GONOGO_GO_OMISSION = 0.50        # Maximum go omission rate

# Flanker Task thresholds
FLANKER_ACCURACY = 0.55          # Minimum accuracy for any condition
FLANKER_OMISSION = 0.50          # Maximum omission rate for any condition

# Operation Span Task thresholds
OP_SPAN_ASYMMETRIC_ACCURACY = 0.55  # Minimum 8x8 asymmetric grid accuracy
OP_SPAN_SYMMETRIC_ACCURACY = 0.55   # Minimum 8x8 symmetric grid accuracy
OP_SPAN_4X4_ACCURACY = 0.25         # Minimum 4x4 grid accuracy irrespective of order
OP_SPAN_ORDER_DIFF = 0.40           # Maximum difference between order-dependent and order-independent accuracy

# Simple Span Task thresholds
SIMPLE_SPAN_ASYMMETRIC_ACCURACY = 0.55  # Minimum 8x8 asymmetric grid accuracy
SIMPLE_SPAN_SYMMETRIC_ACCURACY = 0.55   # Minimum 8x8 symmetric grid accuracy
SIMPLE_SPAN_4X4_ACCURACY = 0.55         # Minimum 4x4 grid accuracy irrespective of order
SIMPLE_SPAN_ORDER_DIFF = 0.20           # Maximum difference between order-dependent and order-independent accuracy

# Cued Task Switching thresholds
CUED_TS_ACCURACY = 0.55          # Minimum accuracy for any condition
CUED_TS_OMISSION = 0.50          # Maximum omission rate for any condition

# N-Back Task thresholds
NBACK_WEIGHTED_ACCURACY = 0.55   # Minimum weighted accuracy
NBACK_MATCH_ACCURACY = 0.20      # Minimum match accuracy
NBACK_MISMATCH_ACCURACY = 0.80   # Minimum mismatch accuracy
NBACK_MATCH_WEIGHT = 0.20        # Weight for match trials
NBACK_MISMATCH_WEIGHT = 0.80     # Weight for mismatch trials

# Spatial Cueing Task thresholds
SPATIAL_CUEING_ACCURACY = 0.55   # Minimum accuracy for any condition
SPATIAL_CUEING_OMISSION = 0.50   # Maximum omission rate for any condition

# Spatial Task Switching thresholds
SPATIAL_TS_ACCURACY = 0.55       # Minimum accuracy for any condition
SPATIAL_TS_OMISSION = 0.50       # Maximum omission rate for any condition

# Stroop Task thresholds
STROOP_ACCURACY = 0.55           # Minimum accuracy for any condition
STROOP_OMISSION = 0.50           # Maximum omission rate for any condition

# Visual Search Task thresholds
VISUAL_SEARCH_ACCURACY = 0.55    # Minimum accuracy for any condition
VISUAL_SEARCH_OMISSION = 0.50    # Maximum omission rate for any condition

# Dictionary of all thresholds for easy reference
THRESHOLDS = {
    'general': {
        'accuracy': ACCURACY_THRESHOLD,
        'omission_rate': OMISSION_THRESHOLD
    },
    'stop_signal': {
        'stop_accuracy_min': STOP_SIGNAL_ACCURACY_MIN,
        'stop_accuracy_max': STOP_SIGNAL_ACCURACY_MAX,
        'go_accuracy': STOP_SIGNAL_GO_ACCURACY,
        'go_rt': STOP_SIGNAL_GO_RT
    },
    'ax_cpt': {
        'accuracy': AX_CPT_ACCURACY
    },
    'gonogo': {
        'go_accuracy': GONOGO_GO_ACCURACY,
        'nogo_accuracy': GONOGO_NOGO_ACCURACY,
        'mean_accuracy': GONOGO_MEAN_ACCURACY,
        'go_omission': GONOGO_GO_OMISSION
    },
    'flanker': {
        'accuracy': FLANKER_ACCURACY,
        'omission_rate': FLANKER_OMISSION
    },
    'operation_span': {
        'asymmetric_accuracy': OP_SPAN_ASYMMETRIC_ACCURACY,
        'symmetric_accuracy': OP_SPAN_SYMMETRIC_ACCURACY,
        '4x4_accuracy': OP_SPAN_4X4_ACCURACY,
        'order_difference': OP_SPAN_ORDER_DIFF
    },
    'simple_span': {
        'asymmetric_accuracy': SIMPLE_SPAN_ASYMMETRIC_ACCURACY,
        'symmetric_accuracy': SIMPLE_SPAN_SYMMETRIC_ACCURACY,
        '4x4_accuracy': SIMPLE_SPAN_4X4_ACCURACY,
        'order_difference': SIMPLE_SPAN_ORDER_DIFF
    },
    'cued_ts': {
        'accuracy': CUED_TS_ACCURACY,
        'omission_rate': CUED_TS_OMISSION
    },
    'nback': {
        'weighted_accuracy': NBACK_WEIGHTED_ACCURACY,
        'match_accuracy': NBACK_MATCH_ACCURACY,
        'mismatch_accuracy': NBACK_MISMATCH_ACCURACY,
        'match_weight': NBACK_MATCH_WEIGHT,
        'mismatch_weight': NBACK_MISMATCH_WEIGHT
    },
    'spatial_cueing': {
        'accuracy': SPATIAL_CUEING_ACCURACY,
        'omission_rate': SPATIAL_CUEING_OMISSION
    },
    'spatial_ts': {
        'accuracy': SPATIAL_TS_ACCURACY,
        'omission_rate': SPATIAL_TS_OMISSION
    },
    'stroop': {
        'accuracy': STROOP_ACCURACY,
        'omission_rate': STROOP_OMISSION
    },
    'visual_search': {
        'accuracy': VISUAL_SEARCH_ACCURACY,
        'omission_rate': VISUAL_SEARCH_OMISSION
    }
} 