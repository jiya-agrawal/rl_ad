import numpy as np
from scipy.sparse import csr_matrix
import math
from sklearn.preprocessing import normalize
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ThompsonSamplingPolicy:
    """
    Implements Thompson Sampling algorithm with Bayesian Logistic Regression
    for the Criteo ad placement contextual bandit problem.
    """
    def __init__(self, feature_dim=74000, lambda_prior=1.0, sparse_features=True):
        """
        Initializes the Thompson Sampling policy with Bayesian Logistic Regression.

        Args:
            feature_dim (int): Dimensionality of the feature space.
            lambda_prior (float): Regularization parameter for the prior.
            sparse_features (bool): Whether features are stored as sparse vectors.
        """
        self.feature_dim = feature_dim
        self.lambda_prior = lambda_prior
        self.sparse_features = sparse_features
        
        # Initialize prior distribution (mean and precision matrix diagonal)
        self.mean = np.zeros(feature_dim)  # Prior mean
        self.precision = lambda_prior * np.ones(feature_dim)  # Prior precision (diagonal)
        
        # Current sampled weights (will be resampled for each prediction)
        self.coef_ = None
        
        # Counters for tracking training progress
        self.n_observations = 0
        self.n_clicks = 0
    
    def _extract_features(self, candidate):
        """
        Extract features from a candidate dictionary.
        Handles the specific data format where features are represented as a dictionary of index:value pairs.
        """
        # Check if the candidate is a dictionary of index:value pairs
        if isinstance(candidate, dict):
            sparse_vec = np.zeros(self.feature_dim)

            for idx, value in candidate.items():
                if isinstance(idx, int) and 0 <= idx < self.feature_dim:
                    sparse_vec[idx] = value

            # logging.debug(f"Extracted sparse vector: {sparse_vec}")
            return sparse_vec

        # Default: return zero vector if feature extraction fails
        logging.debug(f"Feature extraction failed for candidate: {candidate}. Returning zero vector.")
        return np.zeros(self.feature_dim)
    
    def _sigmoid(self, z):
        """Sigmoid function with numerical stability."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    
    def _sample_weights(self):
        """Sample weights from the posterior distribution."""
        # With a diagonal approximation, we can sample each weight independently
        self.coef_ = np.random.normal(
            self.mean,
            1.0 / np.sqrt(self.precision)
        )
    
    def _predict_single(self, features):
        """Predict click probability for a single candidate using current weights."""
        if self.coef_ is None:
            return 0.5  # Default prediction if no weights sampled yet
        
        # Compute logistic regression prediction
        z = np.dot(features, self.coef_)
        return self._sigmoid(z)
    
    def _update_posterior(self, features, reward, learning_rate=0.01):
        """
        Update the posterior distribution with a new observation.
        Using a simplified diagonal covariance approximation for computational efficiency.
        
        Args:
            features: Feature vector of the observed action
            reward: Binary reward (1 for click, 0 for no-click)
            learning_rate: Controls the update step size
        """
        self.n_observations += 1
        if reward > 0.5:  # Click observed
            self.n_clicks += 1
        
        # Current prediction using current weights
        pred = self._predict_single(features)
        
        # Error gradient
        error = reward - pred
        
        # Update mean parameters (simplified gradient step)
        # For non-zero features only to be computationally efficient
        if isinstance(features, np.ndarray) and features.size == self.feature_dim:
            # For dense features
            non_zero_indices = np.nonzero(features)[0]
            self.mean[non_zero_indices] += learning_rate * error * features[non_zero_indices]
            
            # Update precision (increase confidence for observed features)
            # This is a simplified approximation of the Bayesian update
            self.precision[non_zero_indices] += features[non_zero_indices]**2 * 0.01
        else:
            # Fallback for unexpected feature format
            for i in range(self.feature_dim):
                if features[i] != 0:
                    self.mean[i] += learning_rate * error * features[i]
                    self.precision[i] += features[i]**2 * 0.01
    
    def train_on_data(self, training_impressions):
        """
        Trains the policy on a list of impression dictionaries.

        Args:
            training_impressions (list): A list of impression dicts from the dataset.
        """
        print(f"Training Thompson Sampling policy with Bayesian Logistic Regression...")
        print(f"Feature dimensionality: {self.feature_dim}")
        
        # Process training data with progress bar
        for impression in tqdm(training_impressions, desc="Training"):
            # Get the logged action and its features
            logged_action_index = impression.get("logged_action", 0)
            candidates = impression.get("candidates", [])
            
            if not candidates or logged_action_index >= len(candidates):
                continue  # Skip invalid impressions
            
            # Extract features for the logged action
            features = self._extract_features(candidates[logged_action_index])
            
            # Get the reward (1 for click, 0 for no-click)
            # Criteo convention: cost=0.001 for click, cost=0.999 for no-click
            cost = impression.get("cost", 0.999)
            reward = 1.0 if cost < 0.5 else 0.0
            
            # Update our posterior with this observation
            self._update_posterior(features, reward)
        
        # Print training summary
        click_rate = (self.n_clicks / self.n_observations) if self.n_observations > 0 else 0
        print(f"Training complete. Processed {self.n_observations} impressions.")
        print(f"Observed click rate: {click_rate:.4f} ({self.n_clicks} clicks)")
    
    def predict(self, candidates):
        """
        Calculates Thompson Sampling scores for each candidate in an impression.

        Args:
            candidates (list): List of candidate dictionaries for the impression.

        Returns:
            list: Scores for each candidate based on sampled weights.
        """
        # Sample weights from posterior distribution
        self._sample_weights()
        
        # Score each candidate
        scores = []
        for candidate in candidates:
            features = self._extract_features(candidate)
            score = self._predict_single(features)
            scores.append(score)
        
        return scores