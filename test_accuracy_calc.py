#!/usr/bin/env python3
import polars as pl

# Read the parquet file
df = pl.read_parquet('preprocessed_data/sub-sM/axCPT/sub-sM_task-axCPT_run-1.json.parquet')
test_trials = df.filter(pl.col('trial_id') == 'test_trial')

print("AX-CPT Accuracy Calculation Analysis")
print("=" * 50)

# Show current data (no omissions)
print("Current Data (no omissions):")
for condition in ['AX', 'BX', 'AY', 'BY']:
    cond_trials = test_trials.filter(pl.col('condition') == condition)
    total_trials = cond_trials.height
    trials_with_response = cond_trials.filter(pl.col('rt').is_not_null()).height
    correct_trials = cond_trials.filter(
        (pl.col('rt').is_not_null()) & (pl.col('correct_trial') == True)
    ).height
    
    accuracy_method1 = correct_trials / total_trials if total_trials > 0 else 0
    accuracy_method2 = correct_trials / trials_with_response if trials_with_response > 0 else 0
    
    print(f"{condition}: {correct_trials}/{total_trials} = {accuracy_method1:.3f} (all trials) vs {accuracy_method2:.3f} (with responses)")

print(f"\n" + "=" * 50)
print("HYPOTHETICAL EXAMPLE with omissions:")
print("Let's say we had 10 AX trials with 2 omissions and 6 correct responses:")

# Hypothetical example
total_trials = 10
trials_with_response = 8  # 10 - 2 omissions
correct_trials = 6
omitted_trials = 2

accuracy_all_trials = correct_trials / total_trials
accuracy_with_responses = correct_trials / trials_with_response
omission_rate = omitted_trials / total_trials

print(f"Total trials: {total_trials}")
print(f"Trials with response: {trials_with_response}")
print(f"Omitted trials: {omitted_trials}")
print(f"Correct trials: {correct_trials}")
print(f"Accuracy (out of all trials): {correct_trials}/{total_trials} = {accuracy_all_trials:.3f}")
print(f"Accuracy (out of trials with responses): {correct_trials}/{trials_with_response} = {accuracy_with_responses:.3f}")
print(f"Omission rate: {omitted_trials}/{total_trials} = {omission_rate:.3f}")

print(f"\n" + "=" * 50)
print("The get_metrics function calculates accuracy as:")
print("correct_trial.filter(rt.is_not_null()).mean()")
print("This means accuracy is calculated OUT OF TRIALS WITH RESPONSES ONLY")
print("(not out of all trials)")
print("\nThis is the CORRECT approach because:")
print("- Omitted trials should not count against accuracy")
print("- Accuracy should only reflect performance on trials where participant responded")
print("- Omission rate separately captures the failure to respond") 