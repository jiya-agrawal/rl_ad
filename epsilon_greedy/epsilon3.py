import random
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDRegressor
import time
import math

########################################
# Data Parsing and Batch Generation Code
########################################

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
            # If the line contains both 'l' and 'p', it's a header.
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

def batch_generator(file_path, batch_size):
    """
    Generate batches of impressions from the file.
    This function collects 'batch_size' impressions from the streaming generator
    and yields them as a list.
    """
    batch = []
    for imp in parse_file_generator(file_path):
        batch.append(imp)
        if len(batch) >= batch_size:
            yield batch
            print(f"[{time.strftime('%H:%M:%S')}] Yielding a batch of {len(batch)} impressions.")
            batch = []
    if batch:
        print(f"[{time.strftime('%H:%M:%S')}] Yielding final batch of {len(batch)} impressions.")
        yield batch

def fit_vectorizer_sample(file_path, sample_size=10000):
    """
    Take a sample of impressions to fit the DictVectorizer.
    This pass helps determine the full feature space.
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

########################################
# Incremental Training
########################################

def train_model_incremental(file_path, vectorizer, batch_size=10000, epochs=3):
    """
    Train the model incrementally over multiple epochs.
    Data is processed in batches (via batch_generator) so memory usage remains low.
    """
    model = SGDRegressor(max_iter=1, tol=1e-3, warm_start=True)
    
    for epoch in range(epochs):
        print(f"\n[{time.strftime('%H:%M:%S')}] Starting epoch {epoch+1}/{epochs}...")
        batch_num = 0
        impressions_processed = 0
        
        for batch in batch_generator(file_path, batch_size):
            X_batch = []
            y_batch = []
            for imp in batch:
                if not imp["candidates"]:
                    continue
                candidate = imp["candidates"][0]  # assume displayed candidate is at index 0
                combined = {}
                combined.update(imp["banner_features"])
                combined.update(candidate)
                X_batch.append(combined)
                y_batch.append(imp["label"])
                impressions_processed += 1
            
            if X_batch:
                X_vectorized = vectorizer.transform(X_batch)
                model.partial_fit(X_vectorized, y_batch)
                batch_num += 1
                print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch+1}: Processed batch {batch_num} with {len(X_batch)} impressions. Total processed in epoch: {impressions_processed}.")
        
        print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch+1} complete. Total impressions processed: {impressions_processed}.")
    
    print(f"[{time.strftime('%H:%M:%S')}] Incremental training complete over {epochs} epochs.")
    return model

########################################
# Improved Epsilon-Greedy Decision Function
########################################

# Global variable to track the number of decisions made
decision_count = 0

def get_epsilon(decision_count, initial_epsilon=0.5, epsilon_min=0.05, decay_rate=0.00001):
    """Compute the decaying epsilon value based on the number of decisions made."""
    # A simple decay: epsilon = max(epsilon_min, initial_epsilon / (1 + decay_rate * decision_count))
    return max(epsilon_min, initial_epsilon / (1 + decay_rate * decision_count))

def improved_epsilon_greedy(model, vectorizer, banner_features, candidate_features_list, 
                            initial_epsilon=0.5, epsilon_min=0.05, decay_rate=0.00001, beta=0.1):
    """
    An improved epsilon-greedy function that uses:
      - A decaying epsilon schedule (epsilon decreases as more decisions are made).
      - An exploration bonus added to the predicted reward when exploiting.
    
    The bonus is computed as:
       bonus = beta * (predicted_reward - mean(predicted_rewards)) / (std(predicted_rewards) + 1e-5)
    """
    global decision_count
    decision_count += 1
    current_epsilon = get_epsilon(decision_count, initial_epsilon, epsilon_min, decay_rate)
    
    # Build feature vectors for all candidates.
    combined_list = []
    for candidate in candidate_features_list:
        combined = {}
        combined.update(banner_features)
        combined.update(candidate)
        combined_list.append(combined)
        
    X_candidates = vectorizer.transform(combined_list)
    predicted_rewards = model.predict(X_candidates)
    
    # Compute bonus for each candidate based on prediction uncertainty.
    std_rewards = np.std(predicted_rewards)
    if std_rewards > 0:
        bonus = beta * (predicted_rewards - np.mean(predicted_rewards)) / (std_rewards + 1e-5)
    else:
        bonus = np.zeros_like(predicted_rewards)
    
    # Adjusted scores combine the predicted rewards and the bonus.
    adjusted_scores = predicted_rewards + bonus
    
    if random.random() < current_epsilon:
        chosen_index = random.randint(0, len(candidate_features_list) - 1)
        decision = f"exploration (random) with epsilon={current_epsilon:.4f}"
    else:
        chosen_index = int(np.argmax(adjusted_scores))
        decision = f"exploitation (best adjusted reward) with epsilon={current_epsilon:.4f}"
    
    print(f"[{time.strftime('%H:%M:%S')}] Improved epsilon-greedy decision: {decision} => chosen index {chosen_index}")
    print(f"   Predicted rewards: {predicted_rewards}")
    print(f"   Bonus: {bonus}")
    print(f"   Adjusted scores: {adjusted_scores}")
    
    return chosen_index, predicted_rewards, bonus, adjusted_scores

########################################
# Offline Metric Computation
########################################

def compute_offline_metrics(impressions, model, vectorizer, epsilon=0.1):
    """
    Compute offline evaluation metrics (IPS and SNIPS) over a set of impressions.
    For each impression, the new policy probability for the displayed candidate (assumed index 0)
    is computed based on the epsilon-greedy rule.
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
        
        # New policy probability for the displayed candidate (index 0)
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
    print(f"\n[{time.strftime('%H:%M:%S')}] Computed metrics over {N} impressions:")
    print(f"IPS estimate: {IPS}")
    print(f"SNIPS estimate: {SNIPS}")

########################################
# Main Process
########################################

def main():
    file_path = 'criteo_train.txt/criteo_train.txt'
    # file_path = 'criteo_train_small.txt/criteo_train_small.txt'
    epochs = 3         # Number of full passes over the dataset
    batch_size = 10000 # Adjust based on your memory constraints
    
    print(f"[{time.strftime('%H:%M:%S')}] Starting main process...")
    
    # Step 1: Fit vectorizer on a sample of the data.
    vectorizer = fit_vectorizer_sample(file_path, sample_size=10000)
    
    # Step 2: Train the model incrementally over multiple epochs.
    model = train_model_incremental(file_path, vectorizer, batch_size=batch_size, epochs=epochs)
    
    # Step 3: Optionally compute offline evaluation metrics on a sample of impressions.
    print(f"\n[{time.strftime('%H:%M:%S')}] Computing offline evaluation metrics on a sample of impressions...")
    sample_impressions = []
    gen = parse_file_generator(file_path)
    for i in range(1000):
        try:
            sample_impressions.append(next(gen))
        except StopIteration:
            break
    compute_offline_metrics(sample_impressions, model, vectorizer, epsilon=0.1)
    
    # Step 4: Test the improved epsilon-greedy decisions on a few sample impressions.
    print(f"\n[{time.strftime('%H:%M:%S')}] Testing improved epsilon-greedy decisions on sample impressions:")
    gen = parse_file_generator(file_path)
    for i in range(5):
        try:
            imp = next(gen)
        except StopIteration:
            break
        if not imp["candidates"]:
            continue
        print(f"\n[{time.strftime('%H:%M:%S')}] Impression {imp['impression_id']} with {len(imp['candidates'])} candidates:")
        improved_epsilon_greedy(model, vectorizer, imp["banner_features"], imp["candidates"],
                                initial_epsilon=0.5, epsilon_min=0.05, decay_rate=0.00001, beta=0.1)
    
    print(f"[{time.strftime('%H:%M:%S')}] Main process completed.")

if __name__ == "__main__":
    main()
