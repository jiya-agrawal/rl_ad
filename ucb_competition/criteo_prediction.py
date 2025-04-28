# START OF FILE criteo_prediction.py

from __future__ import print_function
import utils
from itertools import (takewhile,repeat)
import numpy as np
import gzip

class CriteoPrediction:
    def __init__(self, filepath, isGzip=False, debug=False):
        self.debug = debug
        if filepath.endswith(".gz") or isGzip:
            # Open in text mode ('rt') for automatic decoding if gzipped
            # Use utf-8 encoding, which is standard for text
            self.fp = gzip.open(filepath, "rt", encoding='utf-8')
        else:
            # Open in text mode ('r') for automatic decoding if not gzipped
            self.fp = open(filepath, "r", encoding='utf-8')
        self.count_max_instances()

    def count_max_instances(self):
        # --- Reading line by line is safer for counting with text mode ---
        self.fp.seek(0) # Go to start
        self.max_instances = sum(1 for _ in self.fp)
        self.fp.seek(0) # Go back to start for iteration

    def __iter__(self):
        return self

    def parse_valid_line(self, line):
        # --- No decoding needed here anymore, line is already a string ---
        line = line.strip()
        if not line: # Handle potential empty lines
             return None # Or raise an error if empty lines are invalid

        try:
            impression_id_marker = line.index(";")
            impression_id = line[:impression_id_marker]
            assert impression_id != ""

            prediction_string = line[impression_id_marker+1:]
            predictions = prediction_string.strip().split(",")
            # Handle case where there might be no predictions after semicolon
            if not prediction_string.strip():
                 parsed_predictions = np.array([]) # Empty array
            else:
                predictions = [x.split(":") for x in predictions if ':' in x] # Ensure ':' exists

                # Determine size dynamically or find max index
                max_action_index = -1
                parsed_preds_dict = {}
                for _pred in predictions:
                    if len(_pred) == 2:
                        try:
                            action = int(_pred[0])
                            score = np.float64(_pred[1])
                            parsed_preds_dict[action] = score
                            if action > max_action_index:
                                max_action_index = action
                        except ValueError:
                             print(f"Warning: Skipping invalid prediction part '{':'.join(_pred)}' in line: {line[:50]}...")
                             continue # Skip malformed prediction parts
                    else:
                         print(f"Warning: Skipping invalid prediction part '{':'.join(_pred)}' (malformed split) in line: {line[:50]}...")
                         continue


                parsed_predictions = np.zeros(max_action_index + 1) # Size based on max index found
                for action, score in parsed_preds_dict.items():
                    parsed_predictions[action] = score


            return {
                    "id" : impression_id,
                    "scores" : parsed_predictions
                    }
        except ValueError as e:
            print(f"Error parsing line: '{line[:100]}...'")
            print(f"ValueError: {e}")
            # Option: return None, raise error, or return partial data
            # Returning None to allow compute_score to potentially handle it gracefully
            return None
        except AssertionError as e:
             print(f"AssertionError parsing line: '{line[:100]}...' - {e}")
             return None


    def next(self):
        # The iterator directly yields strings now due to 'rt'/'r' mode
        line = next(self.fp) # Can raise StopIteration
        parsed_data = self.parse_valid_line(line)
        # Handle potential None return from parse_valid_line
        while parsed_data is None:
            print("Warning: Skipping a line due to parsing error.")
            line = next(self.fp) # Get the next line
            parsed_data = self.parse_valid_line(line)
            # If next() raises StopIteration here, it propagates correctly.
        return parsed_data


    def __next__(self):
        return self.next()

    def close(self):
        self.__del__()

    def __del__(self):
        # Check if fp exists before closing, in case __init__ failed
        if hasattr(self, 'fp') and self.fp:
             self.fp.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Submit the result to crowdAI')
    parser.add_argument('--predictions', dest='predictions', action='store', required=False, default="data/ucb_predictions_on_small_train.gz")
    args = parser.parse_args()

    print("Reading predictions from : ",args.predictions)
    predictions = CriteoPrediction(args.predictions, isGzip=True)

    for _idx, prediction in enumerate(predictions):
        print(prediction)
        if _idx % 500 == 0:
            print("Processed {} impressions...".format(_idx))

    predictions.close()

# END OF FILE criteo_prediction.py