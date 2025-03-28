import random
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDRegressor
import time

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
            label = float(tokens[1])
        elif part.startswith('p'):
            tokens = part.split()
            propensity = float(tokens[1])
        elif part.startswith('f'):
            tokens = part.split()[1:]  # skip the "f"
            for token in tokens:
                key, val = token.split(':')
                banner_features[int(key)] = float(val)
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
                key, val = token.split(':')
                candidate_features[int(key)] = float(val)
    return impression_id, candidate_features

def parse_file_generator(file_path):
    """
    Generator that reads the file line by line and yields one complete impression.
    An impression starts with a header line (containing 'l' and 'p') and is followed by candidate lines.
    """
    current_impression = None
    with open(file_path, 'r') as f:
        for line in f:
            parts = parse_line(line)
            if not parts:
                continue
            # If the line contains both 'l' and 'p', it is a header.
            if any(part.startswith('l') for part in parts) and any(part.startswith('p') for part in parts):
                if current_impression is not None:
                    yield current_impression
                impression_id, label, propensity, banner_features = parse_header(parts)
                current_impression = {
                    "impression_id": impression_id,
                    "label": label,
                    "propensity": propensity,
                    "banner_features": banner_features,
                    "candidates": []
                }
            else:
                imp_id_candidate, candidate_features = parse_candidate(parts)
                if current_impression and current_impression["impression_id"] == imp_id_candidate:
                    current_impression["candidates"].append(candidate_features)
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Warning: Candidate impression ID {imp_id_candidate} doesn't match current impression.")
        if current_impression is not None:
            yield current_impression

def fit_vectorizer_sample(file_path, sample_size=10000):
    """
    Take a sample of impressions to fit the DictVectorizer.
    This pass helps determine the feature space.
    """
    print(f"[{time.strftime('%H:%M:%S')}] Sampling {sample_size} impressions to fit the vectorizer...")
    sample_features = []
    gen = parse_file_generator(file_path)
    for i, imp in enumerate(gen):
        if not imp["candidates"]:
            continue
        candidate = imp["candidates"][0]
        combined = {}
        combined.update(imp["banner_features"])
        combined.update(candidate)
        sample_features.append(combined)
        if (i + 1) >= sample_size:
            break
    vectorizer = DictVectorizer(sparse=True)
    vectorizer.fit(sample_features)
    print(f"[{time.strftime('%H:%M:%S')}] Vectorizer fitted with {len(vectorizer.feature_names_)} features.")
    return vectorizer

def train_model_incremental(file_path, vectorizer, batch_size=10000, epochs=3):
    """
    Train the model incrementally using partial_fit over multiple epochs.
    The data is processed in batches to keep memory usage low.
    """
    # Initialize a model with warm_start to retain parameters across updates.
    model = SGDRegressor(max_iter=1, tol=1e-3, warm_start=True)
    
    for epoch in range(epochs):
        print(f"\n[{time.strftime('%H:%M:%S')}] Starting epoch {epoch+1}/{epochs}...")
        X_batch = []
        y_batch = []
        total_processed = 0
        
        # Re-open the file for each epoch to process the full dataset.
        for imp in parse_file_generator(file_path):
            if not imp["candidates"]:
                continue
            candidate = imp["candidates"][0]  # assume the displayed candidate is at index 0
            combined = {}
            combined.update(imp["banner_features"])
            combined.update(candidate)
            X_batch.append(combined)
            y_batch.append(imp["label"])
            total_processed += 1
            
            if total_processed % batch_size == 0:
                X_vectorized = vectorizer.transform(X_batch)
                model.partial_fit(X_vectorized, y_batch)
                print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch+1}: Processed and updated model with {total_processed} impressions.")
                X_batch = []
                y_batch = []
        
        # Process any remaining impressions in this epoch.
        if X_batch:
            X_vectorized = vectorizer.transform(X_batch)
            model.partial_fit(X_vectorized, y_batch)
            print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch+1}: Final batch processed. Total impressions this epoch: {total_processed}.")
        print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch+1} complete.")
    
    print(f"[{time.strftime('%H:%M:%S')}] Incremental training complete over {epochs} epochs.")
    return model

def epsilon_greedy(model, vectorizer, banner_features, candidate_features_list, epsilon=0.1):
    """
    For an impression, predict rewards for all candidates and decide using epsilon-greedy:
    With probability epsilon, pick a random candidate; otherwise, choose the one with the highest predicted reward.
    """
    combined_list = []
    for candidate in candidate_features_list:
        combined = {}
        combined.update(banner_features)
        combined.update(candidate)
        combined_list.append(combined)
    
    X_candidates = vectorizer.transform(combined_list)
    predicted_rewards = model.predict(X_candidates)
    
    if random.random() < epsilon:
        chosen_index = random.randint(0, len(candidate_features_list) - 1)
        decision = "exploration (random)"
    else:
        chosen_index = int(np.argmax(predicted_rewards))
        decision = "exploitation (best reward)"
    
    print(f"[{time.strftime('%H:%M:%S')}] Epsilon-greedy: {decision} => chosen index {chosen_index}, rewards {predicted_rewards}")
    return chosen_index, predicted_rewards

def compute_offline_metrics(impressions, model, vectorizer, epsilon=0.1):
    """
    Compute offline evaluation metrics (IPS and SNIPS) over impressions.
    For each impression, compute the new policy probability for the displayed candidate (assumed index 0)
    based on the epsilon-greedy rule, and weight the observed reward by the ratio of the new policy probability
    to the logged propensity.
    This function expects `impressions` to be an iterable (it can be a generator).
    """
    total_weighted_reward = 0.0
    total_weight = 0.0
    N = 0
    for imp in impressions:
        if not imp["candidates"]:
            continue
        combined_list = []
        for candidate in imp["candidates"]:
            combined = {}
            combined.update(imp["banner_features"])
            combined.update(candidate)
            combined_list.append(combined)
        X_candidates = vectorizer.transform(combined_list)
        predicted_rewards = model.predict(X_candidates)
        best_index = int(np.argmax(predicted_rewards))
        num_candidates = len(imp["candidates"])
        
        # New policy probability for displayed candidate (index 0):
        if best_index == 0:
            new_policy_prob = (1 - epsilon) + (epsilon / num_candidates)
        else:
            new_policy_prob = epsilon / num_candidates
        
        logged_propensity = imp["propensity"]
        reward = imp["label"]
        weight = new_policy_prob / logged_propensity
        
        total_weighted_reward += reward * weight
        total_weight += weight
        N += 1
    
    IPS = total_weighted_reward / N if N > 0 else 0
    SNIPS = total_weighted_reward / total_weight if total_weight > 0 else 0
    print(f"\n[{time.strftime('%H:%M:%S')}] Computed Metrics over {N} impressions:")
    print(f"IPS estimate: {IPS}")
    print(f"SNIPS estimate: {SNIPS}")

def main():
    file_path = 'criteo_train_small.txt/criteo_train_small.txt'
    epochs = 3  # Number of complete passes over the dataset
    batch_size = 10000
    
    print(f"[{time.strftime('%H:%M:%S')}] Starting main process...")
    
    # First pass: sample a small subset to fit the vectorizer.
    vectorizer = fit_vectorizer_sample(file_path, sample_size=10000)
    
    # Train the model incrementally over multiple epochs.
    model = train_model_incremental(file_path, vectorizer, batch_size=batch_size, epochs=epochs)
    
    # Compute offline evaluation metrics using ALL impressions (streaming the file).
    print(f"\n[{time.strftime('%H:%M:%S')}] Computing offline evaluation metrics on ALL impressions...")
    # Here we pass the generator directly so that all impressions are processed.
    compute_offline_metrics(parse_file_generator(file_path), model, vectorizer, epsilon=0.1)
    
    # Test epsilon-greedy on a few impressions.
    print(f"\n[{time.strftime('%H:%M:%S')}] Testing epsilon-greedy decisions on sample impressions:")
    gen = parse_file_generator(file_path)
    for i in range(5):
        try:
            imp = next(gen)
        except StopIteration:
            break
        if not imp["candidates"]:
            continue
        print(f"\n[{time.strftime('%H:%M:%S')}] Impression {imp['impression_id']} with {len(imp['candidates'])} candidates:")
        epsilon_greedy(model, vectorizer, imp["banner_features"], imp["candidates"], epsilon=0.1)
    
    print(f"[{time.strftime('%H:%M:%S')}] Main process completed.")

if __name__ == "__main__":
    main()
