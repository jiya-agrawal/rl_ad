#!/usr/bin/env python
from __future__ import print_function

import argparse
import gzip
import numpy as np
import sys
from tqdm import tqdm

from criteo_prediction import CriteoPrediction # Needed for parsing prediction format if comparing
from criteo_dataset import CriteoDataset
from softmax_policy import SoftMaxPolicy # Import our SoftMax policy class

def _format_predictions(predictions):
    """ Formats the predictions """
    return ",".join(["{}:{}".format(idx, val) for idx, val in enumerate(predictions)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_set", required=True, help="Path to the dataset file used for training (e.g., criteo_train_small.txt.gz)")
    parser.add_argument("--test_set", required=True, help="Path to the dataset file to generate predictions for (can be the same as training_set)")
    parser.add_argument("--output_path", required=True, help="Path where the predictions file will be saved (e.g., data/softmax_predictions_small.gz)")
    parser.add_argument("--n_arms", type=int, default=75, help="Maximum number of candidate positions (arms) to track. Should be >= max candidates per impression.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature parameter for SoftMax. Lower values favor exploitation.")
    parser.add_argument("--force_gzip", action='store_true', help="Force reading files as gzip.")
    parser.add_argument("--salt_swap", default=False, help="Salt for hashing if used to determine logged action during training (typically False for this setup).")
    parser.add_argument("--inverse_propensity", type=bool, default=True, help="Whether propensity is stored as inverse in training data.")

    args = parser.parse_args()

    print("Initializing SoftMax Policy...")
    softmax_policy = SoftMaxPolicy(n_arms=args.n_arms, temperature=args.temperature)

    print(f"Training SoftMax Policy on logged data from: {args.training_set}")
    softmax_policy.train(
        args.training_set,
        force_gzip=args.force_gzip,
        salt_swap=args.salt_swap,
        inverse_propensity_in_gold_data=args.inverse_propensity
    )

    print(f"Loading data for prediction from: {args.test_set}")
    # Important: Use isTest=True when reading the file for prediction,
    # even if it's the training file, as we only need impression structure.
    test_data = CriteoDataset(args.test_set, isTest=True, isGzip=args.force_gzip)

    print("Generating predictions...")
    output = gzip.open(args.output_path, "wb")

    # Use tqdm for progress bar
    for _idx, _impression in enumerate(tqdm(test_data, desc="Predicting")):
        # Get SoftMax scores for each candidate position
        predictions = softmax_policy.predict(_impression["candidates"])

        # Format the prediction line: impression_id;cand_0:score_0,cand_1:score_1,...
        predictionline = "{};{}".format(
            _impression["id"],
            _format_predictions(predictions)
        ).encode('utf-8') # Encode to bytes for gzip

        output.write(predictionline + b"\n")

    output.close()
    test_data.close()
    print("Predictions saved to {}".format(args.output_path))
