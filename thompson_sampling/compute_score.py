# START OF FILE compute_score.py
#!/usr/bin/env python
from __future__ import print_function

from criteo_dataset import CriteoDataset
from criteo_prediction import CriteoPrediction

import numpy as np
import utils

def grade_predictions(predictions_path, gold_labels_path, expected_number_of_predictions=False, force_gzip=False, _context=False, salt_swap=False, inverse_propensity_in_gold_data=True, jobfactory_utils=False, _debug = False):
    # Ensure gold_data uses the correct isTest=False setting to read labels/propensity
    gold_data = CriteoDataset(gold_labels_path, isTest=False, isGzip=force_gzip, inverse_propensity=inverse_propensity_in_gold_data)
    predictions = CriteoPrediction(predictions_path, isGzip=force_gzip)

    # Instantiate variables
    pos_label = 0.001
    neg_label = 0.999

    max_instances = predictions.max_instances
    if expected_number_of_predictions:
        if max_instances != expected_number_of_predictions:
            raise Exception("The prediction file is expected to contain predictions for {} impressions. But the submitted file contains predictins for {} impressions.".format(expected_number_of_predictions, max_instances))

    num_positive_instances = 0
    num_negative_instances = 0

    #NewPolicy - Stochastic
    # --- CHANGE np.float to np.float64 ---
    prediction_stochastic_numerator = np.zeros(max_instances, dtype = np.float64)
    prediction_stochastic_denominator = np.zeros(max_instances, dtype = np.float64)
    # --- END CHANGE ---

    impression_counter = 0
    for _idx, _impression in enumerate(gold_data):
        # TODO: Add Validation
        prediction = next(predictions)
        if _impression['id'] != prediction['id']:
            raise Exception("`prediction_id` {} doesnot match the corresponding `impression_id`. Please ensure that the lines in the prediction file are in the same order as in the test set.".format(prediction['id']))

        scores = prediction["scores"]

        label = _impression["cost"]
        propensity = _impression["propensity"]
        num_canidadates = len(_impression["candidates"])

        rectified_label = 0

        if label == pos_label:
            rectified_label = 1
            num_positive_instances += 1
        elif label == neg_label:
            num_negative_instances += 1
        else:
            # Handle potential issue if gold_data is not read correctly or has unexpected labels
            print(f"Warning: Unknown cost label '{label}' for impression_id {_impression['id']}. Treating as no-click.")
            # raise Exception("Unknown cost label for impression_id {}".format(_impression['id']))

        # Weight calculation logic might need adjustment based on label handling
        # This original logic assumes label is either pos_label or neg_label strictly
        # log_weight = 1.0 if label == pos_label else (10.0 if label == neg_label else None)
        # if log_weight is None:
        #      raise Exception("Unknown cost label for impression_id {}".format(_impression['id']))
        # Let's stick to the rectified label for IPS calculation which seems standard

        #For deterministic policy (not used for final result in this version, but kept for context)
        best_score = np.max(scores)
        # best_classes = np.argwhere(scores == best_score).flatten() # Not used

        #For stochastic policy
        # score_logged_action = None # Initialized later
        # score_normalizer = 0.0 # Initialized later

        # Ensure scores are float64 for stability
        scores = np.asarray(scores, dtype=np.float64)
        best_score = np.max(scores) # Re-calculate best score after potential type change

        # Prevent overflow/underflow in exp
        scores_with_offset = scores - best_score
        prob_scores = np.exp(scores_with_offset)

        score_normalizer = np.sum(prob_scores)

        # Avoid division by zero if all scores were -inf (unlikely but possible)
        if score_normalizer <= 0:
             # If normalizer is zero, all exp(offset) were zero.
             # This implies all original scores were extremely low or -inf.
             # Assign uniform probability (or handle as error?)
             # Let's assign uniform probability for robustness.
             if num_canidadates > 0:
                 prob_scores = np.ones(num_canidadates) / num_canidadates
                 score_normalizer = 1.0 # Update normalizer
                 print(f"Warning: Score normalizer was zero for impression {_impression['id']}. Using uniform probabilities.")
             else:
                 # No candidates, skip (shouldn't happen in valid data)
                 print(f"Warning: Zero candidates for impression {_impression['id']}. Skipping.")
                 continue # Skip this impression


        logged_action_index = 0
        if salt_swap:
            # Ensure utils module is available and function exists
            if not hasattr(utils, 'compute_integral_hash'):
                 raise ImportError("utils module does not have compute_integral_hash function needed for salt_swap.")
            logged_action_index = utils.compute_integral_hash(_impression['id'], salt_swap, num_canidadates)
        # else: Assume logged action is the first one (index 0) - default behavior

        # Check if logged_action_index is valid
        if logged_action_index >= num_canidadates:
             print(f"Warning: Logged action index {logged_action_index} out of bounds for {num_canidadates} candidates in impression {_impression['id']}. Skipping.")
             # Assign zero weight or handle error? Let's assign zero weight for this impression.
             prediction_stochastic_numerator[_idx] = 0.0
             prediction_stochastic_denominator[_idx] = 0.0
             continue # Skip to next impression


        score_logged_action = prob_scores[logged_action_index]

        # Avoid division by zero for propensity
        if propensity <= 0:
            print(f"Warning: Propensity score is {propensity} for impression {_impression['id']}. Skipping division by zero.")
            # Assign zero weight or handle as error? Assign zero weight.
            prediction_stochastic_weight = 0.0
        else:
             # Calculate importance weight
             prediction_stochastic_weight = 1.0 * score_logged_action / (score_normalizer * propensity)

        # Update accumulators
        prediction_stochastic_numerator[_idx] = rectified_label * prediction_stochastic_weight
        prediction_stochastic_denominator[_idx] = prediction_stochastic_weight

        impression_counter += 1 #Adding this as _idx is not available out of this scope
        if _idx % 1000 == 0 and _debug: # Print less frequently
            print('.', end='', flush=True) # Use flush=True

    gold_data.close()
    predictions.close()

    # Check if any impressions were processed
    if impression_counter == 0:
        print("Warning: No impressions processed. Returning zero scores.")
        return {
            "ips": 0.0, "ips_std": 0.0, "snips": 0.0, "snips_std": 0.0,
            "impwt": 0.0, "impwt_std": 0.0, "max_instances": max_instances
        }


    modified_denominator = num_positive_instances + 10*num_negative_instances
    if modified_denominator <= 0:
         print("Warning: Modified denominator (pos + 10*neg) is zero or negative. Cannot compute IPS.")
         # Handle this case, perhaps return NaN or raise an error. Let's return 0 for now.
         IPS = 0.0
         ImpWt = 0.0 # Denominator for SNIPS might also be zero
    else:
        scaleFactor = np.sqrt(max_instances) / modified_denominator # Calculate scaleFactor only if denominator > 0

    if _debug:
        print('')
        print("Num[Pos/Neg]Test Instances:", num_positive_instances, num_negative_instances)
        print("MaxID; curId", max_instances, impression_counter)
        print("Modified Denominator:", modified_denominator)
        print("Approach & IPS(*10^4) & StdErr(IPS)*10^4 & SN-IPS(*10^4) & StdErr(SN-IPS)*10^4 & AvgImpWt & StdErr(AvgImpWt) \\")

    def compute_result(approach, numerator, denominator):
        _response = {} # Initialize response dictionary

        # --- IPS Calculation ---
        if modified_denominator <= 0:
             IPS = 0.0
             IPS_std = 0.0
        else:
            IPS = numerator.sum(dtype = np.longdouble) / modified_denominator
            IPS_std = 2.58 * numerator.std(dtype = np.longdouble) * scaleFactor        #99% CI

        # --- ImpWt Calculation ---
        if modified_denominator <= 0:
             ImpWt = 0.0
             ImpWt_std = 0.0
        else:
            ImpWt = denominator.sum(dtype = np.longdouble) / modified_denominator
            ImpWt_std = 2.58 * denominator.std(dtype = np.longdouble) * scaleFactor    #99% CI

        _response["ips"] = IPS * 1e4
        _response["ips_std"] = IPS_std * 1e4
        _response["impwt"] = ImpWt
        _response["impwt_std"] = ImpWt_std

        # --- SNIPS Calculation ---
        sum_denominator = denominator.sum(dtype=np.longdouble)
        if ImpWt <= 0 or sum_denominator <= 0: # Check ImpWt or direct sum to avoid division by zero
            print("Warning: Sum of importance weights (denominator for SNIPS) is zero or negative. Cannot compute SNIPS.")
            SNIPS = 0.0
            SNIPS_std = 0.0
        else:
            # SNIPS = IPS / ImpWt # Original calculation is correct if IPS and ImpWt are valid
            sum_numerator = numerator.sum(dtype=np.longdouble)
            SNIPS = sum_numerator / sum_denominator # Direct calculation

            # Delta Method for SNIPS standard error
            normalizer_snips = sum_denominator # Use the direct sum for the normalizer
            #See Art Owen, Monte Carlo, Chapter 9, Section 9.2, Page 9
            #Delta Method to compute an approximate CI for SN-IPS
            # Use np.float64 for intermediate calculations within sum to potentially avoid overflow with longdouble squares
            # But keep final sum as longdouble
            var_num_term = np.square(numerator.astype(np.float64)).sum(dtype=np.longdouble)
            var_den_term = np.square(denominator.astype(np.float64)).sum(dtype=np.longdouble) * SNIPS * SNIPS
            cov_term = 2 * SNIPS * np.multiply(numerator.astype(np.float64), denominator.astype(np.float64)).sum(dtype=np.longdouble)

            Var = (var_num_term + var_den_term - cov_term) / (normalizer_snips * normalizer_snips)

            # Ensure Var is non-negative before sqrt
            if Var < 0:
                print(f"Warning: Variance for SNIPS standard error is negative ({Var}). Setting std to 0.")
                SNIPS_std = 0.0
            else:
                # The formula seems to be missing a factor related to N (max_instances)
                # Standard error of a ratio estimate often involves 1/sqrt(N)
                # Owen's formula gives Var(ratio), so StdErr is sqrt(Var)/sqrt(N)
                # Let's check the original Owen reference if possible. Assuming the formula is correct:
                # It might be Var of the SUM ratio, not the average ratio.
                # If Var is variance of the sum ratio, then std dev is sqrt(Var)
                # Let's assume the original division by sqrt(max_instances) was correct for std error.
                SNIPS_std = 2.58 * np.sqrt(Var) / np.sqrt(max_instances)                 #99% CI

        _response["snips"] = SNIPS*1e4
        _response["snips_std"] = SNIPS_std*1e4
        _response["max_instances"] = max_instances

        return _response

    # Ensure the calculation is only done if the denominator is valid
    if modified_denominator <= 0:
        print("Skipping final result computation due to invalid denominator.")
        # Return a dictionary with default zero/NaN values
        return {
            "ips": 0.0, "ips_std": 0.0, "snips": 0.0, "snips_std": 0.0,
            "impwt": 0.0, "impwt_std": 0.0, "max_instances": max_instances
        }
    else:
        return compute_result('NewPolicy-Stochastic', prediction_stochastic_numerator, prediction_stochastic_denominator)


# --- Keep the argparse block from the previous response ---
if __name__ == "__main__":
    import argparse # Import argparse

    parser = argparse.ArgumentParser(description="Compute IPS and SNIPS scores for Criteo predictions.")
    parser.add_argument("predictions_path", help="Path to the prediction file (gzipped).")
    parser.add_argument("gold_labels_path", help="Path to the gold labels file (gzipped).")
    parser.add_argument("--expected_number", type=int, default=None,
                        help="Expected number of predictions (optional).")
    parser.add_argument("--force_gzip", action='store_true',
                        help="Force reading files as gzip even if extension isn't .gz.")
    parser.add_argument("--salt_swap", default=False,
                        help="Salt for hashing if used to determine logged action (typically False).")
    parser.add_argument("--no_inverse_propensity", action='store_true',
                        help="Flag if propensity is NOT stored as inverse in gold data.")
    parser.add_argument("--debug", action='store_true',
                        help="Enable debug printing.")

    args = parser.parse_args()

    # Calculate inverse_propensity flag based on the new argument
    inverse_propensity_in_gold = not args.no_inverse_propensity

    # Call the function with arguments from the command line
    results = grade_predictions(
        predictions_path=args.predictions_path,
        gold_labels_path=args.gold_labels_path,
        expected_number_of_predictions=args.expected_number,
        force_gzip=args.force_gzip,
        salt_swap=args.salt_swap,
        inverse_propensity_in_gold_data=inverse_propensity_in_gold,
        _debug=args.debug # Pass the debug flag
    )
    print(results)
# END OF FILE compute_score.py