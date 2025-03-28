import random
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDRegressor
import time

def parse_line(line):
    """
    Split a line by the '|' separator and strip whitespace.
    """
    parts = line.strip().split('|')
    return [part.strip() for part in parts if part.strip()]

def parse_header(parts):
    """
    Parse the header line.
    Expected tokens:
      - The first token contains the impression ID.
      - A token starting with 'l' gives the reward or label (e.g., "l 0.999").
      - A token starting with 'p' gives the propensity (e.g., "p 336.294857951").
      - A token starting with 'f' lists banner (display) features.
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
            tokens = part.split()[1:]  # skip the "f"
            for token in tokens:
                key, val = token.split(':')
                candidate_features[int(key)] = float(val)
    return impression_id, candidate_features

def parse_file(file_path):
    """
    Reads the file and groups lines by impression.
    Each impression starts with a header line (containing 'l' and 'p') 
    and is followed by candidate lines.
    """
    impressions = []
    current_impression = None
    line_count = 0
    impression_count = 0

    print(f"[{time.strftime('%H:%M:%S')}] Starting to parse file: {file_path}")
    with open(file_path, 'r') as f:
        for line in f:
            line_count += 1
            parts = parse_line(line)
            if not parts:
                continue
            # Identify header lines by the presence of both 'l' and 'p'
            if any(part.startswith('l') for part in parts) and any(part.startswith('p') for part in parts):
                impression_id, label, propensity, banner_features = parse_header(parts)
                current_impression = {
                    "impression_id": impression_id,
                    "label": label,
                    "propensity": propensity,
                    "banner_features": banner_features,
                    "candidates": []
                }
                impressions.append(current_impression)
                impression_count += 1
                if impression_count % 10000 == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] Parsed {impression_count} impressions so far...")
            else:
                imp_id_candidate, candidate_features = parse_candidate(parts)
                if current_impression and current_impression["impression_id"] == imp_id_candidate:
                    current_impression["candidates"].append(candidate_features)
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Warning: Candidate impression ID {imp_id_candidate} does not match current impression.")
    print(f"[{time.strftime('%H:%M:%S')}] Completed parsing file. Total lines processed: {line_count}, Total impressions: {impression_count}")
    return impressions

def train_model(impressions, vectorizer=None):
    """
    For training, we assume that the displayed candidate is the first candidate.
    We combine banner features with candidate features to form a feature vector and use the header label as the target.
    """
    X = []
    y = []
    total_impressions = len(impressions)
    print(f"[{time.strftime('%H:%M:%S')}] Starting training on {total_impressions} impressions...")
    
    for i, imp in enumerate(impressions):
        if not imp["candidates"]:
            continue
        candidate = imp["candidates"][0]
        combined_features = {}
        combined_features.update(imp["banner_features"])
        combined_features.update(candidate)
        X.append(combined_features)
        y.append(imp["label"])
        
        if (i + 1) % 10000 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] Processed {i + 1}/{total_impressions} impressions for training.")
    
    if vectorizer is None:
        vectorizer = DictVectorizer(sparse=False)
        print(f"[{time.strftime('%H:%M:%S')}] Fitting vectorizer on training data...")
        X_vectorized = vectorizer.fit_transform(X)
    else:
        X_vectorized = vectorizer.transform(X)
    
    print(f"[{time.strftime('%H:%M:%S')}] Training linear model (SGDRegressor)...")
    model = SGDRegressor(max_iter=1000, tol=1e-3)
    model.fit(X_vectorized, y)
    print(f"[{time.strftime('%H:%M:%S')}] Training complete.")
    
    return model, vectorizer

def epsilon_greedy(model, vectorizer, banner_features, candidate_features_list, epsilon=0.1):
    """
    For a given impression, compute predicted rewards for all candidates and decide using epsilon-greedy:
    With probability epsilon, choose uniformly at random; otherwise, pick the candidate with the highest predicted reward.
    """
    combined_features_list = []
    for candidate in candidate_features_list:
        combined = {}
        combined.update(banner_features)
        combined.update(candidate)
        combined_features_list.append(combined)
    
    X_candidate = vectorizer.transform(combined_features_list)
    predicted_rewards = model.predict(X_candidate)
    
    if random.random() < epsilon:
        chosen_index = random.randint(0, len(candidate_features_list) - 1)
        decision = "exploration (random choice)"
    else:
        chosen_index = int(np.argmax(predicted_rewards))
        decision = "exploitation (best predicted reward)"
        
    print(f"[{time.strftime('%H:%M:%S')}] Epsilon-greedy decision: {decision} chosen index {chosen_index} with predicted rewards {predicted_rewards}")
    return chosen_index, predicted_rewards

def compute_offline_metrics(impressions, model, vectorizer, epsilon=0.1):
    """
    Compute offline evaluation metrics (IPS and SNIPS) over all impressions.
    We assume that the displayed candidate is at index 0.
    For each impression, we compute the new policy probability for the displayed candidate:
      - If the displayed candidate is the best (highest predicted reward) among all candidates,
        its probability is (1 - epsilon) + (epsilon / num_candidates);
      - Otherwise, it is epsilon / num_candidates.
    Then, for each impression, we weight the observed reward by the ratio of the new policy probability
    to the logged propensity.
    """
    total_weighted_reward = 0.0
    total_weight = 0.0
    N = 0
    for imp in impressions:
        if not imp["candidates"]:
            continue
        
        # Build combined features for all candidates in this impression.
        combined_features_list = []
        for candidate in imp["candidates"]:
            combined = {}
            combined.update(imp["banner_features"])
            combined.update(candidate)
            combined_features_list.append(combined)
        X_candidates = vectorizer.transform(combined_features_list)
        predicted_rewards = model.predict(X_candidates)
        
        best_index = int(np.argmax(predicted_rewards))
        num_candidates = len(imp["candidates"])
        
        # Compute new policy probability for the displayed candidate (assumed index 0).
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
    
    IPS = total_weighted_reward / N
    SNIPS = total_weighted_reward / total_weight if total_weight != 0 else 0
    print(f"\n[{time.strftime('%H:%M:%S')}] Computed Metrics over {N} impressions:")
    print(f"IPS estimate: {IPS}")
    print(f"SNIPS estimate: {SNIPS}")

def main():
    # file_path = 'criteo_train.txt/criteo_train.txt'
    file_path = 'criteo_train_small.txt/criteo_train_small.txt'
    
    print(f"[{time.strftime('%H:%M:%S')}] Starting main process.")
    
    impressions = parse_file(file_path)
    print(f"[{time.strftime('%H:%M:%S')}] Parsed {len(impressions)} impressions from the file.")
    
    model, vectorizer = train_model(impressions)
    print(f"[{time.strftime('%H:%M:%S')}] Model training completed. Model is ready.")
    
    print(f"\n[{time.strftime('%H:%M:%S')}] Starting epsilon-greedy simulation on sample impressions...")
    for imp in impressions[:5]:
        if not imp["candidates"]:
            continue
        print(f"\n[{time.strftime('%H:%M:%S')}] Processing impression {imp['impression_id']} with {len(imp['candidates'])} candidates.")
        chosen_index, predicted_rewards = epsilon_greedy(
            model, vectorizer, imp["banner_features"], imp["candidates"], epsilon=0.1)
        print(f"[{time.strftime('%H:%M:%S')}] Impression {imp['impression_id']}: Chosen candidate index {chosen_index}")
    
    # Compute offline evaluation metrics (IPS and SNIPS)
    compute_offline_metrics(impressions, model, vectorizer, epsilon=0.1)
    print(f"[{time.strftime('%H:%M:%S')}] Offline metric computation completed.")

if __name__ == "__main__":
    main()
