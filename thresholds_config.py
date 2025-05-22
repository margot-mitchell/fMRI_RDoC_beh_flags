"""Configuration file containing thresholds for behavioral data quality control.

This module defines thresholds for flagging behavioral data based on various metrics
such as accuracy, reaction times, and omission rates for different tasks.
"""

# Remove general thresholds
# ACCURACY_THRESHOLD = 0.55
# OMISSION_THRESHOLD = 0.5

# Stop Signal Task Thresholds
STOP_SIGNAL_ACCURACY_MIN = 0.25
STOP_SIGNAL_ACCURACY_MAX = 0.75
STOP_SIGNAL_GO_ACCURACY = 0.55
STOP_SIGNAL_GO_RT = 750
STOP_SIGNAL_OMISSION_RATE = 0.5

# AX-CPT Task Thresholds
AX_CPT_ACCURACY = 0.55
AX_CPT_OMISSION_RATE = 0.5

# Go/NoGo Task Thresholds
GONOGO_GO_ACCURACY = 0.857
GONOGO_NOGO_ACCURACY = 0.143
GONOGO_MEAN_ACCURACY = 0.55
GONOGO_OMISSION_RATE = 0.5

# Operation Span Task Thresholds
OP_SPAN_ASYMMETRIC_ACCURACY = 0.55
OP_SPAN_SYMMETRIC_ACCURACY = 0.55
OP_SPAN_4X4_ACCURACY = 0.25
OP_SPAN_ORDER_DIFF = 0.4

# Simple Span Task Thresholds
SIMPLE_SPAN_4X4_ACCURACY = 0.55
SIMPLE_SPAN_ORDER_DIFF = 0.2

# Cued Task Switching Thresholds
CUED_TS_ACCURACY = 0.55
CUED_TS_OMISSION_RATE = 0.5

# Flanker Task Thresholds
FLANKER_ACCURACY = 0.55
FLANKER_OMISSION_RATE = 0.5

# N-Back Task Thresholds
NBACK_MATCH_WEIGHT = 0.2
NBACK_MISMATCH_WEIGHT = 0.8
NBACK_WEIGHTED_ACCURACY = 0.55
NBACK_MATCH_ACCURACY = 0.2
NBACK_MISMATCH_ACCURACY = 0.8

# Spatial Cueing Task Thresholds
SPATIAL_CUEING_ACCURACY = 0.55
SPATIAL_CUEING_OMISSION_RATE = 0.5

# Spatial Task Switching Thresholds
SPATIAL_TS_ACCURACY = 0.55
SPATIAL_TS_OMISSION_RATE = 0.5

# Stroop Task Thresholds
STROOP_ACCURACY = 0.55
STROOP_OMISSION_RATE = 0.5

# Visual Search Task Thresholds
VISUAL_SEARCH_ACCURACY = 0.55
VISUAL_SEARCH_OMISSION_RATE = 0.5

# Add THRESHOLDS dictionary
THRESHOLDS = {
    'stop_signal': {
        'stop_accuracy_min': STOP_SIGNAL_ACCURACY_MIN,
        'stop_accuracy_max': STOP_SIGNAL_ACCURACY_MAX,
        'go_accuracy': STOP_SIGNAL_GO_ACCURACY,
        'go_rt': STOP_SIGNAL_GO_RT,
        'omission_rate': STOP_SIGNAL_OMISSION_RATE,
    },
    'ax_cpt': {
        'accuracy': AX_CPT_ACCURACY,
        'omission_rate': AX_CPT_OMISSION_RATE,
    },
    'gonogo': {
        'go_accuracy': GONOGO_GO_ACCURACY,
        'nogo_accuracy': GONOGO_NOGO_ACCURACY,
        'mean_accuracy': GONOGO_MEAN_ACCURACY,
        'omission_rate': GONOGO_OMISSION_RATE,
    },
    'flanker': {
        'accuracy': FLANKER_ACCURACY,
        'omission_rate': FLANKER_OMISSION_RATE,
    },
    'operation_span': {
        'asymmetric_accuracy': OP_SPAN_ASYMMETRIC_ACCURACY,
        'symmetric_accuracy': OP_SPAN_SYMMETRIC_ACCURACY,
        '4x4_accuracy': OP_SPAN_4X4_ACCURACY,
        'order_diff': OP_SPAN_ORDER_DIFF,
    },
    'simple_span': {
        '4x4_accuracy': SIMPLE_SPAN_4X4_ACCURACY,
        'order_diff': SIMPLE_SPAN_ORDER_DIFF,
    },
    'nback': {
        'match_weight': NBACK_MATCH_WEIGHT,
        'mismatch_weight': NBACK_MISMATCH_WEIGHT,
        'weighted_accuracy': NBACK_WEIGHTED_ACCURACY,
        'match_accuracy': NBACK_MATCH_ACCURACY,
        'mismatch_accuracy': NBACK_MISMATCH_ACCURACY,
    },
    'cued_ts': {
        'accuracy': CUED_TS_ACCURACY,
        'omission_rate': CUED_TS_OMISSION_RATE,
    },
    'spatial_cueing': {
        'accuracy': SPATIAL_CUEING_ACCURACY,
        'omission_rate': SPATIAL_CUEING_OMISSION_RATE,
    },
    'spatial_ts': {
        'accuracy': SPATIAL_TS_ACCURACY,
        'omission_rate': SPATIAL_TS_OMISSION_RATE,
    },
    'stroop': {
        'accuracy': STROOP_ACCURACY,
        'omission_rate': STROOP_OMISSION_RATE,
    },
    'visual_search': {
        'accuracy': VISUAL_SEARCH_ACCURACY,
        'omission_rate': VISUAL_SEARCH_OMISSION_RATE,
    },
} 