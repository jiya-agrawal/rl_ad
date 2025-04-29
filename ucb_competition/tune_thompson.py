#!/usr/bin/env python
from __future__ import print_function

import argparse
import gzip
import numpy as np
import os
import tempfile
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from criteo_dataset import CriteoDataset
from thompson_policy import ThompsonSamplingPolicy
from compute_score import grade_predictions
from write_criteo_helper import write_criteo_format

def _format_predictions(predictions):
    """ Formats the predictions """
    return ",".join(["{}:{}".format(idx, val) for idx, val in enumerate(predictions)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Thompson Sampling for the dataset.")
    parser.add_argument("--dataset_path", required=True, help="Path to the full dataset file (e.g., criteo_train_small.txt.gz).")
    parser.add_argument("--n_arms", type=int, default=75, help="Maximum number of candidate positions (arms).")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Ratio of data to use for training (e.g., 0.8 for 80/20 split).")
    parser.add_argument("--force_gzip", action='store_true', help="Force reading/writing files as gzip.")
    parser.add_argument("--no_inverse_propensity", action='store_true', help="Flag if propensity is NOT stored as inverse in data.")

    args = parser.parse_args()

    inverse_propensity_in_data = not args.no_inverse_propensity

    print(f"Loading and splitting dataset: {args.dataset_path}")
    full_dataset = CriteoDataset(args.dataset_path, isTest=False, isGzip=args.force_gzip, inverse_propensity=inverse_propensity_in_data)
    all_impressions = list(full_dataset)
    full_dataset.close()
    print(f"Total impressions loaded: {len(all_impressions)}")

    if len(all_impressions) < 2:
        print("Error: Dataset too small to split.")
        exit()

    train_impressions, validation_impressions = train_test_split(
        all_impressions, train_size=args.split_ratio, shuffle=True, random_state=42
    )
    print(f"Split into {len(train_impressions)} training and {len(validation_impressions)} validation impressions.")

    results_log = []

    # Define a directory for storing intermediate files
    intermediate_dir = os.path.join(os.getcwd(), "intermediate_files")
    os.makedirs(intermediate_dir, exist_ok=True)

    # Replace temporary file paths with fixed paths in the intermediate directory
    temp_gold_path = os.path.join(intermediate_dir, "validation_gold.txt.gz")
    temp_pred_path = os.path.join(intermediate_dir, "thompson_predictions.gz")

    # Update the print statements to reflect the new file paths
    print(f"Validation gold labels will be saved to: {temp_gold_path}")
    print(f"Thompson predictions will be saved to: {temp_pred_path}")

    write_criteo_format(validation_impressions, temp_gold_path, force_gzip=True, inverse_propensity=inverse_propensity_in_data)

    print("\n----- Training Thompson Sampling Policy -----")
    policy = ThompsonSamplingPolicy(n_arms=args.n_arms)
    policy.train_on_data(train_impressions)

    print("Generating predictions for validation set...")
    try:
        with gzip.open(temp_pred_path, "wt", encoding='utf-8') as output:
            for impression in tqdm(validation_impressions, desc="Predicting"):
                predictions = policy.predict(impression["candidates"])
                predictionline = "{};{}".format(
                    impression["id"],
                    _format_predictions(predictions)
                )
                output.write(predictionline + "\n")
        print(f"Predictions saved temporarily to {temp_pred_path}")

        print("Evaluating predictions...")
        score_results = grade_predictions(
            predictions_path=temp_pred_path,
            gold_labels_path=temp_gold_path,
            force_gzip=True,
            inverse_propensity_in_gold_data=inverse_propensity_in_data,
            _debug=False
        )

        snips_score = score_results.get("snips", -float('inf'))
        print(f"Results: SNIPS = {snips_score:.4f}, IPS = {score_results.get('ips', 0.0):.4f}")
        results_log.append({"snips": snips_score, "ips": score_results.get('ips', 0.0)})

    except Exception as e:
        print(f"ERROR occurred during processing: {e}")
        import traceback
        traceback.print_exc()

    print("\n----- Summary -----")
    for res in results_log:
        print(f"SNIPS = {res['snips']:.4f}, IPS = {res['ips']:.4f}")