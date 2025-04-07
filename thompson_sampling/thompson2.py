#with incremntal batch training
import random
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import BayesianRidge
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
# Incremental Training with Bayesian Regression
########################################

def train_bayesian_model_incremental(file_path, vectorizer, batch_size=10000, epochs=3):
    """
    Train the Bayesian Ridge Regression model incrementally over multiple epochs.
    Unlike SGDRegressor, BayesianRidge doesn't have partial_fit, so we need to maintain
    the training data and retrain on each batch.
    """
    # BayesianRidge doesn't support partial_fit, but we can simulate it
    model = BayesianRidge(compute_score=True, n_iter=50, alpha_1=1e-6, alpha_2=1e-6, 
                         lambda_1=1e-6, lambda_2=1e-6)
    
    all_X_batch = []
    all_y_batch = []
    
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
                
                # For the first epoch, we collect all data
                if epoch == 0:
                    all_X_batch.append(X_vectorized)
                    all_y_batch.extend(y_batch)
                    
                    # To avoid memory issues, we limit how much data we keep
                    if len(all_y_batch) > 100000:
                        print(f"[{time.strftime('%H:%M:%S')}] Limiting training data to last 100,000 examples.")
                        all_X_batch = [all_X_batch[-1]]
                        all_y_batch = all_y_batch[-100000:]
                    
                    # Concatenate and fit
                    X_train = all_X_batch[0] if len(all_X_batch) == 1 else np.vstack(all_X_batch)
                    model.fit(X_train, all_y_batch)
                
                batch_num += 1
                print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch+1}: Processed batch {batch_num} with {len(X_batch)} impressions. Total processed in epoch: {impressions_processed}.")
        
        print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch+1} complete. Total impressions processed: {impressions_processed}.")
    
    print(f"[{time.strftime('%H:%M:%S')}] Incremental training complete over {epochs} epochs.")
    return model

########################################
# Thompson Sampling Decision Function
########################################

def thompson_sampling(model, vectorizer, banner_features, candidate_features_list, n_samples=10):
    """
    Implements Thompson Sampling for candidate selection.
    
    For each candidate:
    1. Get predicted mean and standard deviation from the Bayesian model
    2. Sample from the posterior distribution (Normal distribution)
    3. Choose the candidate with the highest sampled value
    
    Args:
        model: Trained BayesianRidge model
        vectorizer: Fitted DictVectorizer
        banner_features: Features of the banner/impression
        candidate_features_list: List of candidate features
        n_samples: Number of samples to draw from posterior (default: 10)
    
    Returns:
        chosen_index: Index of the chosen candidate
        predicted_means: Mean predictions for each candidate
        predicted_stds: Standard deviation of predictions for each candidate
        sampled_values: The sampled values from posterior distributions
    """
    # Build feature vectors for all candidates
    combined_list = []
    for candidate in candidate_features_list:
        combined = {}
        combined.update(banner_features)
        combined.update(candidate)
        combined_list.append(combined)
        
    X_candidates = vectorizer.transform(combined_list)
    
    # Get predicted means
    predicted_means = model.predict(X_candidates)
    
    # Estimate standard deviations (uncertainty)
    # BayesianRidge doesn't directly give uncertainties, so we approximate
    # using the model's alpha_ and lambda_ parameters
    
    # Get the diagonal of the covariance matrix for each prediction
    # Note: This is computationally intensive for high-dimensional features
    # We'll use a simplified approach
    
    # For each feature vector, estimate prediction variance
    # sigma^2 = sigma_noise^2 + x^T Sigma x
    # where sigma_noise^2 is model.sigma_ and Sigma is approx 1/alpha_ * I
    
    sigma_noise = np.sqrt(1.0 / model.alpha_)
    n_candidates = len(candidate_features_list)
    predicted_stds = np.ones(n_candidates) * sigma_noise
    
    # Draw samples from the posterior for each candidate
    sampled_values = np.zeros(n_candidates)
    for i in range(n_candidates):
        # Sample from Normal(mean, std)
        sampled_values[i] = np.random.normal(predicted_means[i], predicted_stds[i])
    
    # Choose the candidate with highest sampled value
    chosen_index = int(np.argmax(sampled_values))
    
    print(f"[{time.strftime('%H:%M:%S')}] Thompson Sampling decision:")
    print(f"   Predicted means: {predicted_means}")
    print(f"   Predicted stds: {predicted_stds}")
    print(f"   Sampled values: {sampled_values}")
    print(f"   Chosen index: {chosen_index}")
    
    return chosen_index, predicted_means, predicted_stds, sampled_values

########################################
# Offline Metric Computation
########################################

def compute_offline_metrics_ts(impressions, model, vectorizer):
    """
    Compute offline evaluation metrics (IPS and SNIPS) for Thompson Sampling over a set of impressions.
    For each impression, the new policy probability for the displayed candidate (assumed index 0)
    is estimated by running multiple simulations of Thompson Sampling.
    """
    total_weighted_reward = 0.0
    total_weight = 0.0
    N = 0
    n_simulations = 100  # Number of simulations to estimate probability
    
    for imp in impressions:
        if not imp["candidates"]:
            continue
            
        # Run simulations to estimate probability of selecting each candidate
        selection_counts = np.zeros(len(imp["candidates"]))
        for _ in range(n_simulations):
            chosen_index, _, _, _ = thompson_sampling(model, vectorizer, imp["banner_features"], imp["candidates"])
            selection_counts[chosen_index] += 1
            
        # Convert counts to probabilities
        new_policy_probs = selection_counts / n_simulations
        
        # Get probability for displayed candidate (index 0)
        new_policy_prob = new_policy_probs[0]
        
        logged_propensity = imp["propensity"]
        reward = imp["label"]
        
        # Skip if new policy probability is too small to avoid extreme weights
        if new_policy_prob < 0.01:
            continue
            
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
    
    print(f"[{time.strftime('%H:%M:%S')}] Starting main process with Thompson Sampling...")
    
    # Step 1: Fit vectorizer on a sample of the data
    vectorizer = fit_vectorizer_sample(file_path, sample_size=10000)
    
    # Step 2: Train the Bayesian model incrementally
    model = train_bayesian_model_incremental(file_path, vectorizer, batch_size=batch_size, epochs=epochs)
    
    # Step 3: Optionally compute offline evaluation metrics on a sample of impressions
    print(f"\n[{time.strftime('%H:%M:%S')}] Computing offline evaluation metrics with Thompson Sampling on a sample of impressions...")
    sample_impressions = []
    gen = parse_file_generator(file_path)
    for i in range(1000):
        try:
            sample_impressions.append(next(gen))
        except StopIteration:
            break
    compute_offline_metrics_ts(sample_impressions, model, vectorizer)
    
    # Step 4: Test Thompson Sampling decisions on a few sample impressions
    print(f"\n[{time.strftime('%H:%M:%S')}] Testing Thompson Sampling decisions on sample impressions:")
    gen = parse_file_generator(file_path)
    for i in range(5):
        try:
            imp = next(gen)
        except StopIteration:
            break
        if not imp["candidates"]:
            continue
        print(f"\n[{time.strftime('%H:%M:%S')}] Impression {imp['impression_id']} with {len(imp['candidates'])} candidates:")
        thompson_sampling(model, vectorizer, imp["banner_features"], imp["candidates"])
    
    print(f"[{time.strftime('%H:%M:%S')}] Main process completed.")

if __name__ == "__main__":
    main()