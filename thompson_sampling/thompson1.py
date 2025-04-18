
import random
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDRegressor, BayesianRidge
import time
import math
from scipy.stats import norm

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
# Thompson Sampling Model Training
########################################

def train_bayesian_model_incremental(file_path, vectorizer, batch_size=10000, epochs=3):
    """
    Train a Bayesian linear regression model incrementally over multiple epochs.
    This will be used for Thompson Sampling as it provides uncertainty estimates.
    """
    # BayesianRidge doesn't support partial_fit, so we'll have to use a different approach
    model = BayesianRidge(fit_intercept=True, compute_score=True)
    
    all_X = []
    all_y = []
    
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
                # For BayesianRidge, we'll collect all data to train in one go
                all_X.append(X_vectorized)
                all_y.extend(y_batch)
                batch_num += 1
                print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch+1}: Processed batch {batch_num} with {len(X_batch)} impressions. Total processed in epoch: {impressions_processed}.")
        
        print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch+1} complete. Total impressions processed: {impressions_processed}.")
    
    # Combine all data and train the model once
    if all_X:
        X_combined = np.vstack([X.toarray() for X in all_X])
        print(f"[{time.strftime('%H:%M:%S')}] Training BayesianRidge model on {X_combined.shape[0]} samples...")
        model.fit(X_combined, all_y)
        print(f"[{time.strftime('%H:%M:%S')}] BayesianRidge model training complete.")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] No data collected for training!")
    
    return model

class ThompsonSamplingModel:
    """
    A wrapper around SGDRegressor that maintains a variance estimate for each feature
    to support Thompson Sampling.
    """
    def __init__(self, alpha=0.1, l2_reg=0.01):
        self.model = SGDRegressor(learning_rate='constant', eta0=alpha, alpha=l2_reg, max_iter=1, tol=1e-3, warm_start=True)
        self.feature_var = None  # Will be initialized later
        self.noise_var = 1.0     # Initial estimate of noise variance
        self.n_samples = 0       # Number of samples seen so far
        
    def partial_fit(self, X, y):
        """Update the model with a batch of data."""
        n_features = X.shape[1]
        
        # Initialize feature variance if this is the first batch
        if self.feature_var is None:
            self.feature_var = np.ones(n_features) * 10.0  # Start with high uncertainty
        
        # Update the mean model
        self.model.partial_fit(X, y)
        
        # Update the variance estimates
        # Simple approach: decay the variance with more observations
        self.n_samples += X.shape[0]
        
        # As we see more data, reduce variance (simple approach)
        decay_factor = max(0.9, 1.0 / (1.0 + self.n_samples * 0.001))
        self.feature_var *= decay_factor
        
        # Update noise variance based on prediction errors
        y_pred = self.model.predict(X)
        errors = (y - y_pred) ** 2
        self.noise_var = 0.9 * self.noise_var + 0.1 * np.mean(errors)
        
    def predict(self, X):
        """Predict the mean reward for each instance."""
        return self.model.predict(X)
    
    def sample_parameters(self):
        """Sample a set of parameters from the approximate posterior distribution."""
        coef = self.model.coef_.copy()
        intercept = self.model.intercept_
        
        # Sample perturbed coefficients
        perturbed_coef = np.random.normal(coef, np.sqrt(self.feature_var))
        perturbed_intercept = np.random.normal(intercept, np.sqrt(self.noise_var))
        
        return perturbed_coef, perturbed_intercept
    
    def predict_with_sample(self, X):
        """Predict using a sampled set of parameters (Thompson Sampling)."""
        coef, intercept = self.sample_parameters()
        return X.dot(coef) + intercept

def train_thompson_model_incremental(file_path, vectorizer, batch_size=10000, epochs=3):
    """
    Train the Thompson Sampling model incrementally over multiple epochs.
    Data is processed in batches to keep memory usage low.
    """
    model = ThompsonSamplingModel()
    
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
                X_vectorized = vectorizer.transform(X_batch).toarray()  # Convert to dense for simplicity
                model.partial_fit(X_vectorized, y_batch)
                batch_num += 1
                print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch+1}: Processed batch {batch_num} with {len(X_batch)} impressions. Total processed in epoch: {impressions_processed}.")
        
        print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch+1} complete. Total impressions processed: {impressions_processed}.")
    
    print(f"[{time.strftime('%H:%M:%S')}] Incremental Thompson model training complete over {epochs} epochs.")
    return model

########################################
# Thompson Sampling Decision Function
########################################

# Global variable to track the number of decisions made
decision_count = 0

def thompson_sampling_decision(model, vectorizer, banner_features, candidate_features_list):
    """
    Implement Thompson Sampling for decision making.
    
    For each candidate arm:
    1. Sample parameter values from the posterior distribution
    2. Calculate expected reward using these sampled parameters
    3. Choose the arm with the highest sampled expected reward
    """
    global decision_count
    decision_count += 1
    
    # Build feature vectors for all candidates
    combined_list = []
    for candidate in candidate_features_list:
        combined = {}
        combined.update(banner_features)
        combined.update(candidate)
        combined_list.append(combined)
    
    X_candidates = vectorizer.transform(combined_list).toarray()  # Convert to dense for simplicity
    
    # Get mean predictions for logging purposes
    mean_rewards = model.predict(X_candidates)
    
    # Sample from the posterior and make predictions with the sampled parameters
    sampled_rewards = model.predict_with_sample(X_candidates)
    
    # Select the candidate with the highest sampled reward
    chosen_index = int(np.argmax(sampled_rewards))
    
    print(f"[{time.strftime('%H:%M:%S')}] Thompson sampling decision: chosen index {chosen_index}")
    print(f"   Mean predicted rewards: {mean_rewards}")
    print(f"   Sampled rewards: {sampled_rewards}")
    
    return chosen_index, mean_rewards, sampled_rewards

def gaussian_thompson_sampling(model, vectorizer, banner_features, candidate_features_list):
    """
    Thompson sampling using BayesianRidge which provides mean and std deviation for predictions.
    
    For each candidate:
    1. Get the mean and std deviation of the reward prediction
    2. Sample from this Gaussian distribution
    3. Choose the arm with the highest sampled reward
    """
    global decision_count
    decision_count += 1
    
    # Build feature vectors for all candidates
    combined_list = []
    for candidate in candidate_features_list:
        combined = {}
        combined.update(banner_features)
        combined.update(candidate)
        combined_list.append(combined)
    
    X_candidates = vectorizer.transform(combined_list)
    
    # Get mean and std dev of predictions
    mean_rewards, std_rewards = model.predict(X_candidates, return_std=True)
    
    # Sample from Gaussian distributions for each arm
    sampled_rewards = np.random.normal(mean_rewards, std_rewards)
    
    # Select the candidate with the highest sampled reward
    chosen_index = int(np.argmax(sampled_rewards))
    
    print(f"[{time.strftime('%H:%M:%S')}] Gaussian Thompson sampling decision: chosen index {chosen_index}")
    print(f"   Mean predicted rewards: {mean_rewards}")
    print(f"   Reward standard deviations: {std_rewards}")
    print(f"   Sampled rewards: {sampled_rewards}")
    
    return chosen_index, mean_rewards, std_rewards, sampled_rewards

########################################
# Offline Metric Computation
########################################

def compute_thompson_offline_metrics(impressions, model, vectorizer, n_samples=50):
    """
    Compute offline evaluation metrics (IPS and SNIPS) over a set of impressions.
    For each impression, the new policy probability for the displayed candidate is computed
    based on Thompson Sampling - we approximate this by sampling multiple times.
    """
    total_weighted_reward = 0.0
    total_weight = 0.0
    N = 0
    
    for imp in impressions:
        if not imp["candidates"]:
            continue
        
        # Build feature vectors for all candidates
        combined_list = []
        for candidate in imp["candidates"]:
            combined = {}
            combined.update(imp["banner_features"])
            combined.update(candidate)
            combined_list.append(combined)
        
        X_candidates = vectorizer.transform(combined_list).toarray()
        
        # Approximate the policy probability by sampling multiple times
        arm_counts = np.zeros(len(imp["candidates"]))
        for _ in range(n_samples):
            # Sample and predict
            sampled_rewards = model.predict_with_sample(X_candidates)
            chosen_idx = np.argmax(sampled_rewards)
            arm_counts[chosen_idx] += 1
        
        # Compute probability of choosing the displayed candidate (index 0)
        new_policy_prob = arm_counts[0] / n_samples
        
        # Ensure we don't have zero probability (add small epsilon)
        new_policy_prob = max(new_policy_prob, 1e-6)
        
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

def compute_gaussian_thompson_offline_metrics(impressions, model, vectorizer, n_samples=50):
    """
    Compute offline evaluation metrics for Gaussian Thompson Sampling over a set of impressions.
    """
    total_weighted_reward = 0.0
    total_weight = 0.0
    N = 0
    
    for imp in impressions:
        if not imp["candidates"]:
            continue
        
        # Build feature vectors for all candidates
        combined_list = []
        for candidate in imp["candidates"]:
            combined = {}
            combined.update(imp["banner_features"])
            combined.update(candidate)
            combined_list.append(combined)
        
        X_candidates = vectorizer.transform(combined_list)
        
        # Get mean and std dev of predictions
        mean_rewards, std_rewards = model.predict(X_candidates, return_std=True)
        
        # Approximate the policy probability by sampling multiple times
        arm_counts = np.zeros(len(imp["candidates"]))
        for _ in range(n_samples):
            # Sample from Gaussian distributions for each arm
            sampled_rewards = np.random.normal(mean_rewards, std_rewards)
            chosen_idx = np.argmax(sampled_rewards)
            arm_counts[chosen_idx] += 1
        
        # Compute probability of choosing the displayed candidate (index 0)
        new_policy_prob = arm_counts[0] / n_samples
        
        # Ensure we don't have zero probability (add small epsilon)
        new_policy_prob = max(new_policy_prob, 1e-6)
        
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
    
    print(f"[{time.strftime('%H:%M:%S')}] Starting Thompson Sampling implementation...")
    
    # Step 1: Fit vectorizer on a sample of the data.
    vectorizer = fit_vectorizer_sample(file_path, sample_size=10000)
    
    # Step 2a: Train the custom Thompson Sampling model incrementally
    print(f"\n[{time.strftime('%H:%M:%S')}] Training custom Thompson Sampling model...")
    thompson_model = train_thompson_model_incremental(file_path, vectorizer, batch_size=batch_size, epochs=epochs)
    
    # Step 2b: Train the Bayesian Ridge model for Gaussian Thompson Sampling
    print(f"\n[{time.strftime('%H:%M:%S')}] Training Bayesian Ridge model for Gaussian Thompson Sampling...")
    bayesian_model = train_bayesian_model_incremental(file_path, vectorizer, batch_size=batch_size, epochs=epochs)
    
    # Step 3: Compute offline evaluation metrics on a sample of impressions
    print(f"\n[{time.strftime('%H:%M:%S')}] Computing offline evaluation metrics on a sample of impressions...")
    sample_impressions = []
    gen = parse_file_generator(file_path)
    for i in range(1000):
        try:
            sample_impressions.append(next(gen))
        except StopIteration:
            break
    
    compute_thompson_offline_metrics(sample_impressions, thompson_model, vectorizer, n_samples=50)
    
    # Step 4: Test the Thompson Sampling decisions on a few sample impressions
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
        thompson_sampling_decision(thompson_model, vectorizer, imp["banner_features"], imp["candidates"])
    
    # Step 5: Test the Gaussian Thompson Sampling decisions
    print(f"\n[{time.strftime('%H:%M:%S')}] Testing Gaussian Thompson Sampling decisions on sample impressions:")
    gen = parse_file_generator(file_path)
    for i in range(5):
        try:
            imp = next(gen)
        except StopIteration:
            break
        if not imp["candidates"]:
            continue
        print(f"\n[{time.strftime('%H:%M:%S')}] Impression {imp['impression_id']} with {len(imp['candidates'])} candidates:")
        gaussian_thompson_sampling(bayesian_model, vectorizer, imp["banner_features"], imp["candidates"])
    
    print(f"[{time.strftime('%H:%M:%S')}] Thompson Sampling implementation completed.")

if __name__ == "__main__":
    main()
