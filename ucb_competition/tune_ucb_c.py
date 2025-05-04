# START OF FILE tune_ucb_c.py
#!/usr/bin/env python
from __future__ import print_function

import argparse
import gzip
import numpy as np
import math
import os
import tempfile
from sklearn.model_selection import train_test_split # For splitting data
from tqdm import tqdm # For progress bars

# Import necessary classes and functions from the starter kit
from criteo_dataset import CriteoDataset
from ucb_policy import UCBPolicy
from compute_score import grade_predictions # Import the scoring function
from write_criteo_helper import write_criteo_format # Import the helper

def _format_predictions(predictions):
    """ Formats the predictions """
    return ",".join(["{}:{}".format(idx, val) for idx, val in enumerate(predictions)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune the exploration factor 'c' for UCBPolicy.")
    parser.add_argument("--dataset_path", required=True, help="Path to the full dataset file (e.g., criteo_train_small.txt.gz).")
    parser.add_argument("--n_arms", type=int, default=75, help="Maximum number of candidate positions (arms).")
    parser.add_argument("--c_values", nargs='+', type=float, default=[0.1, 0.5, 1.0, 1.414, 2.0, 4.0, 8.0], help="List of exploration factor 'c' values to test.")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Ratio of data to use for training (e.g., 0.8 for 80/20 split).")
    parser.add_argument("--force_gzip", action='store_true', help="Force reading/writing files as gzip.")
    parser.add_argument("--salt_swap", default=False, help="Salt for hashing if used (typically False).")
    parser.add_argument("--no_inverse_propensity", action='store_true', help="Flag if propensity is NOT stored as inverse in data.")

    args = parser.parse_args()

    inverse_propensity_in_data = not args.no_inverse_propensity

    print(f"Loading and splitting dataset: {args.dataset_path}")
    full_dataset = CriteoDataset(args.dataset_path, isTest=False, isGzip=args.force_gzip, inverse_propensity=inverse_propensity_in_data)
    all_impressions = list(full_dataset) # Read all impressions into memory
    full_dataset.close()
    print(f"Total impressions loaded: {len(all_impressions)}")

    if len(all_impressions) < 2:
         print("Error: Dataset too small to split.")
         exit()

    # Split data into training and validation sets (list of dictionaries)
    train_impressions, validation_impressions = train_test_split(
        all_impressions, train_size=args.split_ratio, shuffle=True, random_state=42 # Added shuffle and random_state
    )
    print(f"Split into {len(train_impressions)} training and {len(validation_impressions)} validation impressions.")

    best_c = None
    best_snips = -float('inf')
    results_log = []

    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Using temporary directory: {tmpdir}")

        temp_gold_path = os.path.join(tmpdir, "validation_gold.txt.gz")
        temp_pred_path = os.path.join(tmpdir, "ucb_predictions.gz")

        # Write validation data to a temporary gold file ONCE
        write_criteo_format(validation_impressions, temp_gold_path, force_gzip=True, inverse_propensity=inverse_propensity_in_data)

        for c_value in args.c_values:
            print(f"\n----- Testing c = {c_value:.3f} -----")

            # 1. Initialize and Train Policy
            policy = UCBPolicy(n_arms=args.n_arms, c=c_value)
            policy.train_on_data(train_impressions, salt_swap=args.salt_swap)

            # 2. Generate Predictions for Validation Set
            print("Generating predictions for validation set...")
            try:
                with gzip.open(temp_pred_path, "wt", encoding='utf-8') as output: # Use text mode 'wt'
                    for impression in tqdm(validation_impressions, desc=f"Predicting (c={c_value:.3f})"):
                        predictions = policy.predict(impression["candidates"])
                        predictionline = "{};{}".format(
                            impression["id"],
                            _format_predictions(predictions)
                        )
                        output.write(predictionline + "\n")
                print(f"Predictions saved temporarily to {temp_pred_path}")

                # 3. Evaluate Predictions
                print("Evaluating predictions...")
                # Important: pass inverse_propensity_in_gold_data according to how the temp gold file was written
                score_results = grade_predictions(
                    predictions_path=temp_pred_path,
                    gold_labels_path=temp_gold_path, # Use the temp gold file
                    force_gzip=True, # Both temp files are gzipped
                    salt_swap=args.salt_swap, # Pass salt_swap if used
                    inverse_propensity_in_gold_data=inverse_propensity_in_data,
                    _debug=False # Disable debug printing within grade_predictions for cleaner output here
                )

                snips_score = score_results.get("snips", -float('inf')) # Get SNIPS score
                print(f"Results for c = {c_value:.3f}: SNIPS = {snips_score:.4f}, IPS = {score_results.get('ips', 0.0):.4f}")
                results_log.append({"c": c_value, "snips": snips_score, "ips": score_results.get('ips', 0.0)})

                # 4. Update Best C
                if snips_score > best_snips:
                    best_snips = snips_score
                    best_c = c_value
                    print(f"!!! New best SNIPS score found: {best_snips:.4f} for c = {best_c:.3f} !!!")

            except Exception as e:
                 print(f"ERROR occurred during processing for c = {c_value:.3f}: {e}")
                 import traceback
                 traceback.print_exc()
                 results_log.append({"c": c_value, "snips": "Error", "ips": "Error"})

            # Clean up prediction file for the next iteration (optional, handled by tempdir exit)
            # if os.path.exists(temp_pred_path):
            #    os.remove(temp_pred_path)

    # End of loop through c_values

    print("\n----- Hyperparameter Tuning Summary -----")
    print("Tested c values and SNIPS scores (*10^4):")
    for res in results_log:
        snips_str = res['snips'] if isinstance(res['snips'], str) else f"{res['snips']:.4f}"
        print(f"  c = {res['c']:.3f} : SNIPS = {snips_str}")

    if best_c is not None:
        print(f"\nBest exploration factor c = {best_c:.3f} with SNIPS score = {best_snips:.4f} (*10^4)")
    else:
        print("\nNo successful evaluation completed.")

# END OF FILE tune_ucb_c.py