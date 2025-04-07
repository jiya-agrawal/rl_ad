import random
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from scipy.stats import beta
import time

class ThompsonSamplingBandit:
    def __init__(self, num_arms, feature_dim):
        """
        Initialize Thompson Sampling Bandit for multi-armed contextual bandit problem
        
        Args:
        - num_arms: Number of candidate ads/arms
        - feature_dim: Dimensionality of feature space
        """
        self.num_arms = num_arms
        self.feature_dim = feature_dim
        
        # Prior parameters for Beta distribution for each arm
        # We use Beta(1,1) as a non-informative prior
        self.alpha = np.ones(num_arms)
        self.beta = np.ones(num_arms)
        
        # Linear model parameters for contextual bandits
        self.theta = np.zeros((num_arms, feature_dim))
        self.cov = np.zeros((num_arms, feature_dim, feature_dim))
        for i in range(num_arms):
            self.cov[i] = np.eye(feature_dim)

    def sample_action(self, context):
        """
        Sample an action using Thompson Sampling
        
        Args:
        - context: Feature vector of the current impression
        
        Returns:
        - Chosen arm index
        """
        # Sample from posterior distributions
        samples = []
        for i in range(self.num_arms):
            # Sample reward mean from Beta distribution
            mean = self.alpha[i] / (self.alpha[i] + self.beta[i])
            
            # Sample from linear model
            mu = np.dot(self.theta[i], context)
            sigma_squared = np.dot(context, np.linalg.inv(self.cov[i])).dot(context)
            
            # Thompson Sampling sample
            sample = mean + mu + np.random.normal(0, np.sqrt(sigma_squared))
            samples.append(sample)
        
        return np.argmax(samples)

    def update(self, chosen_arm, context, reward):
        """
        Update the model parameters based on observed reward
        
        Args:
        - chosen_arm: Index of the selected arm
        - context: Feature vector
        - reward: Observed reward (0 or 1)
        """
        # Update Beta distribution parameters
        if reward > 0:
            self.alpha[chosen_arm] += reward
        else:
            self.beta[chosen_arm] += 1 - reward
        
        # Update linear model parameters
        # Use simplified Bayesian linear regression update
        self.cov[chosen_arm] += np.outer(context, context)
        error = reward - np.dot(self.theta[chosen_arm], context)
        self.theta[chosen_arm] += np.linalg.solve(self.cov[chosen_arm], context) * error

def thompson_sampling_train(file_path, vectorizer, batch_size=10000, epochs=3):
    """
    Train Thompson Sampling model incrementally
    
    Args:
    - file_path: Path to the training data
    - vectorizer: Fitted DictVectorizer
    - batch_size: Number of impressions to process in each batch
    - epochs: Number of passes through the dataset
    """
    # Use the same parse_file_generator and batch_generator from the original code
    from epsilon_greedy.epsilon3 import parse_file_generator, batch_generator
    
    # Determine feature dimension
    sample_impressions = []
    gen = parse_file_generator(file_path)
    for i in range(1000):
        try:
            imp = next(gen)
            if imp["candidates"]:
                sample_features = []
                for candidate in imp["candidates"]:
                    combined = {}
                    combined.update(imp["banner_features"])
                    combined.update(candidate)
                    sample_features.append(combined)
                sample_impressions.append((imp, sample_features))
        except StopIteration:
            break
    
    # Vectorize a sample to get feature dimension
    sample_X = vectorizer.transform(
        [feature for _, features in sample_impressions for feature in features]
    )
    feature_dim = sample_X.shape[1]
    
    # Initialize Thompson Sampling Bandit
    bandit = ThompsonSamplingBandit(
        num_arms=len(sample_impressions[0][1]), 
        feature_dim=feature_dim
    )
    
    # Incremental training loop
    for epoch in range(epochs):
        print(f"\n[{time.strftime('%H:%M:%S')}] Starting Thompson Sampling epoch {epoch+1}/{epochs}...")
        
        for batch in batch_generator(file_path, batch_size):
            for imp in batch:
                if not imp["candidates"]:
                    continue
                
                # Prepare feature vectors for all candidates
                combined_list = []
                for candidate in imp["candidates"]:
                    combined = {}
                    combined.update(imp["banner_features"])
                    combined.update(candidate)
                    combined_list.append(combined)
                
                # Vectorize features
                X_candidates = vectorizer.transform(combined_list)
                
                # Choose action using Thompson Sampling
                chosen_index = bandit.sample_action(X_candidates.toarray()[0])
                
                # Update with observed reward (label)
                bandit.update(chosen_index, X_candidates.toarray()[0], imp["label"])
    
    return bandit

def main():
    file_path = 'criteo_train.txt/criteo_train.txt'
    epochs = 3
    batch_size = 10000
    
    # Reuse vectorizer fitting from original code
    from epsilon_greedy.epsilon3 import fit_vectorizer_sample
    vectorizer = fit_vectorizer_sample(file_path, sample_size=10000)
    
    # Train Thompson Sampling model
    thompson_model = thompson_sampling_train(
        file_path, vectorizer, 
        batch_size=batch_size, 
        epochs=epochs
    )

if __name__ == "__main__":
    main()