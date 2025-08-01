"""Configuration file containing threshold values for quality control flags.

This file contains all the threshold values used to determine if a task's
performance metrics are within acceptable ranges. These thresholds are used
by generate_flags.py to generate quality control flags.
"""

# General thresholds
PROPORTION_FEEDBACK_THRESHOLD = 0.2  
RT_THRESHOLD = 1000  


# Stop Signal thresholds
STOP_SIGNAL_GO_ACCURACY = 0.75   
STOP_SIGNAL_GO_RT = 700     
STOP_SIGNAL_OMISSION_RATE = 0.05 
## Conditional accuracy threshold: if stop accuracy is < 25% OR > 75%, flag (Waiting for stop signal; not following instructions)
STOP_SIGNAL_STOP_ACCURACY_MIN = 0.35
STOP_SIGNAL_STOP_ACCURACY_MAX = 0.65 

# AX-CPT thresholds
AX_CPT_AX_ACCURACY = 0.75     
AX_CPT_BX_ACCURACY = 0.75     
AX_CPT_AY_ACCURACY = 0.75     
AX_CPT_BY_ACCURACY = 0.75     
AX_CPT_AX_OMISSION_RATE = 0.05  
AX_CPT_BX_OMISSION_RATE = 0.05  
AX_CPT_AY_OMISSION_RATE = 0.05  
AX_CPT_BY_OMISSION_RATE = 0.05  
CUE_FIXATION_RESPONSE_PROPORTION = .1

# Go/NoGo thresholds
GONOGO_GO_ACCURACY_MIN = 0.857        
GONOGO_NOGO_ACCURACY_MIN = 0.143     
GONOGO_MEAN_ACCURACY = 0.75  
GONOGO_GO_OMISSION_RATE = 0.05     

# Operation Only Span thresholds
OP_ONLY_SPAN_ASYMMETRIC_ACCURACY = 0.75  
OP_ONLY_SPAN_SYMMETRIC_ACCURACY = 0.75  
OP_ONLY_SPAN_8X8_OMISSION_RATE = 0.05      

# Operation Span thresholds
OP_SPAN_ASYMMETRIC_ACCURACY = 0.75  
OP_SPAN_SYMMETRIC_ACCURACY = 0.75   
OP_SPAN_4X4_IRRESPECTIVE_ACCURACY = 0.4        
OP_SPAN_4X4_OMISSION_RATE = 0.05  
OP_SPAN_8X8_OMISSION_RATE = 0.05  
OP_SPAN_ORDER_DIFF = 0.4  # if irrespective accuracy is > 40% higher than respective, flag

# Simple Span thresholds
SIMPLE_SPAN_4X4_IRRESPECTIVE_ACCURACY = 0.6    
SIMPLE_SPAN_4X4_OMISSION_RATE = 0.05  
SIMPLE_SPAN_ORDER_DIFF = 0.2  # if irrespective accuracy is > 40% higher than respective, flag

# N-Back thresholds
NBACK_WEIGHTED_2BACK_ACCURACY = 0.75   
NBACK_WEIGHTED_1BACK_ACCURACY = 0.75  
## Weights for weighted accuracy calculation (based on the number of trials)
NBACK_MATCH_WEIGHT = 0.2           
NBACK_MISMATCH_WEIGHT = 0.8      
## Conditional accuracy threshold: if mismatch is >80% AND match >20% (in either 2-back or 1-back), flag
NBACK_MISMATCH_MIN_CONDITIONAL_ACCURACY = 0.8
NBACK_MATCH_MIN_CONDITIONAL_ACCURACY = 0.2
NBACK_OMISSION_RATE = 0.05  

# Cued TS thresholds
CUED_TS_SWITCH_STAY_ACCURACY = 0.75 
CUED_TS_STAY_SWITCH_ACCURACY = 0.75
CUED_TS_SWITCH_SWITCH_ACCURACY = 0.75
CUED_TS_SWITCH_STAY_OMISSION_RATE = 0.05 
CUED_TS_STAY_SWITCH_OMISSION_RATE = 0.05
CUED_TS_SWITCH_SWITCH_OMISSION_RATE = 0.05
CUED_TS_PARITY_ACCURACY = 0.75
CUED_TS_MAGNITUDE_ACCURACY = 0.75

# Spatial cueing thresholds
SPATIAL_CUEING_DOUBLE_CUE_ACCURACY = 0.75
SPATIAL_CUEING_DOUBLE_CUE_OMISSION_RATE = 0.05
SPATIAL_CUEING_INVALID_CUE_ACCURACY = 0.75
SPATIAL_CUEING_INVALID_CUE_OMISSION_RATE = 0.05
SPATIAL_CUEING_NO_CUE_ACCURACY = 0.75
SPATIAL_CUEING_NO_CUE_OMISSION_RATE = 0.05
SPATIAL_CUEING_VALID_CUE_ACCURACY = 0.75
SPATIAL_CUEING_VALID_CUE_OMISSION_RATE = 0.05

# Spatial TS thresholds
SPATIAL_TS_STAY_STAY_ACCURACY = 0.75 
SPATIAL_TS_STAY_SWITCH_ACCURACY = 0.75
SPATIAL_TS_SWITCH_SWITCH_ACCURACY = 0.75
SPATIAL_TS_STAY_STAY_OMISSION_RATE = 0.05 
SPATIAL_TS_STAY_SWITCH_OMISSION_RATE = 0.05
SPATIAL_TS_SWITCH_SWITCH_OMISSION_RATE = 0.05
SPATIAL_TS_COLOR_ACCURACY = 0.75
SPATIAL_TS_FORM_ACCURACY = 0.75

# Stroop thresholds
STROOP_CONGRUENT_ACCURACY = 0.75
STROOP_CONGRUENT_OMISSION_RATE = .05        
STROOP_INCONGRUENT_ACCURACY = 0.75
STROOP_INCONGRUENT_OMISSION_RATE = .05  

# Visual search thresholds
VISUAL_SEARCH_CONJUNCTION_24_ACCURACY = 0.75      
VISUAL_SEARCH_CONJUNCTION_24_OMISSION_RATE = 0.05 
VISUAL_SEARCH_CONJUNCTION_8_ACCURACY = 0.75      
VISUAL_SEARCH_CONJUNCTION_8_OMISSION_RATE = 0.05 
VISUAL_SEARCH_FEATURE_24_ACCURACY = 0.75      
VISUAL_SEARCH_FEATURE_24_OMISSION_RATE = 0.05 
VISUAL_SEARCH_FEATURE_8_ACCURACY = 0.75      
VISUAL_SEARCH_FEATURE_8_OMISSION_RATE = 0.05 

# Flanker thresholds
FLANKER_ACCURACY_CONGRUENT_ACCURACY = 0.75
FLANKER_CONGRUENT_OMISSION_RATE = .05        
FLANKER_INCONGRUENT_ACCURACY = 0.75
FLANKER_INCONGRUENT_OMISSION_RATE = .05   