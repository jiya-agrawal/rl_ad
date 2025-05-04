# START OF FILE write_criteo_helper.py
import gzip

def format_features(features_dict):
    """Formats a feature dictionary into 'id:val id:val ...' string."""
    # Sort by feature ID for consistency, although not strictly required by format
    return " ".join([f"{k}:{v}" for k, v in sorted(features_dict.items())])

def write_criteo_format(impressions, output_path, force_gzip=True, inverse_propensity=True):
    """
    Writes a list of impression dictionaries to a file in Criteo format.

    Args:
        impressions (list): List of impression dictionaries.
        output_path (str): Path to the output file.
        force_gzip (bool): Whether to gzip the output.
        inverse_propensity (bool): Whether propensity was stored as inverse (affects writing).
    """
    open_func = gzip.open if force_gzip else open
    mode = "wt" # Text mode for writing strings
    encoding = 'utf-8'

    print(f"Writing {len(impressions)} impressions to {output_path}...")
    with open_func(output_path, mode, encoding=encoding) as f:
        for impression in impressions:
            imp_id = impression["id"]
            cost = impression["cost"]
            prop = impression["propensity"]
            candidates = impression["candidates"]

            # Adjust propensity for writing if it was originally inverse
            prop_to_write = 1.0 / prop if inverse_propensity and prop != 0 else prop

            # Write the first (logged) line - ASSUMING first candidate is the logged one here
            # This part is crucial and needs to match how the original data was structured
            # If the logged action wasn't always the first, this needs modification.
            # For simplicity in this example, we write the cost/propensity with the first candidate's features.
            if not candidates: continue # Skip if no candidates

            first_cand_features_str = format_features(candidates[0])
            first_line = f"{imp_id} |l {cost} |p {prop_to_write} |f {first_cand_features_str}\n"
            f.write(first_line)

            # Write subsequent candidate lines
            for i in range(1, len(candidates)):
                cand_features_str = format_features(candidates[i])
                cand_line = f"{imp_id} |f {cand_features_str}\n"
                f.write(cand_line)
    print("Finished writing gold data.")

# END OF FILE write_criteo_helper.py