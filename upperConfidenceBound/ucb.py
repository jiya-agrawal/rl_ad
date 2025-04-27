
import random
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import BayesianRidge 
import time
import math
from scipy.stats import norm 

# Data Parsing and Batch Generation Code


def parse_line(line):
    """Split a line by the '|' separator and strip whitespace."""
    parts = line.strip().split('|')
    return [part.strip() for part in parts if part.strip()]

def parse_header(parts):
    """
    Parse the header line.
    Expected tokens:
      - The first token is the impression ID.
      - A token starting with 'l' gives the reward/label.
      - A token starting with 'p' gives the propensity.
      - A token starting with 'f' gives banner (display) features.
    """
    impression_id = parts[0].split()[0]
    label = None
    propensity = None
    banner_features = {}

    for part in parts[1:]:
        if part.startswith('l'):
            tokens = part.split()
            # Convert label: 0.001 -> 1 (click), 0.999 -> 0 (no click)
            raw_label = float(tokens[1])
            label = 1.0 if raw_label < 0.5 else 0.0 # Assuming 0.001 is click, 0.999 no click
        elif part.startswith('p'):
            tokens = part.split()
            propensity = float(tokens[1])
        elif part.startswith('f'):
            tokens = part.split()[1:]  # skip the "f"
            for token in tokens:
                try:
                    key, val = token.split(':')
                    banner_features[int(key)] = float(val)
                except ValueError:
                    print(f"[{time.strftime('%H:%M:%S')}] Warning: Skipping malformed banner feature token: {token}")
                    continue # Skip malformed feature
    # Basic validation
    if label is None or propensity is None:
         print(f"[{time.strftime('%H:%M:%S')}] Warning: Missing label or propensity in header: {parts}")
         # Return None or raise error if critical info is missing
         return None, None, None, None

    return impression_id, label, propensity, banner_features

def parse_candidate(parts):
    """
    Parse a candidate line.
    The first token is the impression ID.
    Then, a token starting with 'f' holds candidate product features.
    """
    impression_id = parts[0].split()[0]
    candidate_features = {}
    for part in parts[1:]:
        if part.startswith('f'):
            tokens = part.split()[1:]  # skip "f"
            for token in tokens:
                try:
                    key, val = token.split(':')
                    candidate_features[int(key)] = float(val)
                except ValueError:
                    print(f"[{time.strftime('%H:%M:%S')}] Warning: Skipping malformed candidate feature token: {token}")
                    continue # Skip malformed feature

    return impression_id, candidate_features

def parse_file_generator(file_path):
    """
    Generator that reads the file line by line and yields one complete impression.
    An impression starts with a header line (containing 'l' and 'p') and is followed by candidate lines.
    Handles potential errors during parsing.
    """
    current_impression = None
    line_num = 0
    with open(file_path, 'r') as f:
        for line in f:
            line_num += 1
            parts = parse_line(line)
            if not parts:
                continue

            # Check if it looks like a header line (must contain label and propensity markers)
            is_header = any(part.startswith('l') for part in parts) and \
                        any(part.startswith('p') for part in parts)

            if is_header:
                # Yield the previous impression if complete
                if current_impression is not None:
                    # Basic validation: Ensure candidates were added
                    if not current_impression.get("candidates"):
                         print(f"[{time.strftime('%H:%M:%S')}] Warning: Impression {current_impression.get('impression_id')} at line ~{line_num} has no candidates. Skipping.")
                    else:
                        yield current_impression

                # Start a new impression
                impression_id, label, propensity, banner_features = parse_header(parts)

                # Handle cases where header parsing failed
                if impression_id is None:
                    print(f"[{time.strftime('%H:%M:%S')}] Error: Failed to parse header at line {line_num}. Skipping line: {line.strip()}")
                    current_impression = None # Reset current impression
                    continue

                current_impression = {
                    "impression_id": impression_id,
                    "label": label,
                    "propensity": propensity,
                    "banner_features": banner_features,
                    "candidates": [],
                    "line_num": line_num # For debugging
                }
            elif current_impression is not None: # If it's not a header, assume it's a candidate for the current impression
                imp_id_candidate, candidate_features = parse_candidate(parts)
                # Check if the candidate belongs to the current impression
                if current_impression["impression_id"] == imp_id_candidate:
                    current_impression["candidates"].append(candidate_features)
                else:
                    # This case might happen if parsing got misaligned or data is corrupt
                    print(f"[{time.strftime('%H:%M:%S')}] Warning: Candidate ImpID {imp_id_candidate} at line {line_num} doesn't match current ImpID {current_impression['impression_id']}. Trying to recover or discarding candidate.")
                    # Option 1: Discard candidate (safer)
                    # Option 2: Could try to yield current_impression and start anew, but risky.
                    # Let's discard the candidate for now.
            else:
                 # Line is not a header and there's no current impression - likely noise or format issue
                 print(f"[{time.strftime('%H:%M:%S')}] Warning: Orphan candidate or noise line {line_num}: {line.strip()}. Skipping.")


        # Yield the last impression if it exists and is valid
        if current_impression is not None:
             if not current_impression.get("candidates"):
                 print(f"[{time.strftime('%H:%M:%S')}] Warning: Last impression {current_impression.get('impression_id')} has no candidates. Skipping.")
             else:
                yield current_impression

def batch_generator(file_path, batch_size):
    """
    Generate batches of impressions from the file.
    This function collects 'batch_size' impressions from the streaming generator
    and yields them as a list.
    """
    batch = []
    processed_count = 0
    gen_start_time = time.time()
    for imp in parse_file_generator(file_path):
        batch.append(imp)
        processed_count += 1
        if len(batch) >= batch_size:
            yield batch
            batch_end_time = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Yielding batch of {len(batch)} impressions (Total processed: {processed_count}). Time: {batch_end_time - gen_start_time:.2f}s")
            batch = []
            gen_start_time = time.time() # Reset timer for next batch

    if batch: # Yield any remaining impressions
        batch_end_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Yielding final batch of {len(batch)} impressions (Total processed: {processed_count}). Time: {batch_end_time - gen_start_time:.2f}s")
        yield batch

def fit_vectorizer_sample(file_path, sample_size=10000):
    """
    Take a sample of impressions to fit the DictVectorizer.
    This pass helps determine the full feature space.
    Uses the generator to handle potentially large files.
    """
    print(f"[{time.strftime('%H:%M:%S')}] Sampling up to {sample_size} impressions to fit the vectorizer...")
    sample_features = []
    processed_count = 0
    start_time = time.time()
    try:
        gen = parse_file_generator(file_path)
        for i, imp in enumerate(gen):
            if not imp or not imp.get("candidates"):
                print(f"[{time.strftime('%H:%M:%S')}] Warning: Skipping empty or invalid impression during vectorizer sampling.")
                continue

            # Use the first candidate for feature structure sampling
            candidate = imp["candidates"][0]
            combined = {}
            # Combine banner and candidate features safely
            if imp.get("banner_features"):
                 combined.update(imp["banner_features"])
            if candidate:
                 combined.update(candidate)

            if combined: # Only add if we have features
                sample_features.append(combined)
            processed_count += 1

            if processed_count >= sample_size:
                break
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Error during vectorizer sampling: {e}")
        # Decide if you want to proceed with potentially incomplete sampling or stop
        if not sample_features:
             raise RuntimeError("Could not collect any samples for vectorizer fitting.") from e

    if not sample_features:
        raise RuntimeError("No valid feature samples found for vectorizer fitting. Check data file and parsing logic.")

    vectorizer = DictVectorizer(sparse=True) # Use sparse matrix
    vectorizer.fit(sample_features)
    end_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Vectorizer fitted with {len(vectorizer.feature_names_)} features from {processed_count} impressions in {end_time - start_time:.2f}s.")
    return vectorizer

