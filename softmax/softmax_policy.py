# softmax_policy.py
import numpy as np
import math
from criteo_dataset import CriteoDataset

class SoftMaxPolicy:
    """
    Implements the SoftMax algorithm treating each candidate position as an arm.
    Learns from logged data by updating counts for the logged action's position.
    
    Algorithm:
    - For each arm i, compute Q_t(a_i) as the average reward
    - Convert to probability distribution using softmax: 
      π_t(a_i) = exp(Q_t(a_i)/τ) / ∑_j exp(Q_t(a_j)/τ)
    - Where τ is the temperature parameter controlling exploration
    """
    def __init__(self, n_arms, temperature=0.1):
        """
        Initializes the SoftMax policy.

        Args:
            n_arms (int): The maximum number of arms (candidate positions) expected.
            temperature (float): Temperature parameter controlling exploration.
                                Lower values make the policy more greedy.
        """
        self.n_arms = n_arms
        self.temperature = temperature  # Temperature parameter τ
        self.n_pulls = np.ones(n_arms)  # Number of times each arm was pulled
        self.sum_rewards = np.zeros(n_arms)  # Sum of rewards for each arm
        self.total_steps = 1  # Total number of training impressions processed
        self.default_avg_reward = 0.01  # Default reward for unknown arms

    def _update_stats(self, impression, salt_swap=False):
        """Internal helper to update stats for a single impression."""
        self.total_steps += 1
        num_candidates = len(impression.get("candidates", []))
        if num_candidates == 0:
            return  # Skip impressions with no candidates

        pos_label = 0.001  # Label for positive click
        neg_label = 0.999  # Label for no click

        # Determine the index of the action that was logged
        logged_action_index = 0  # Default assumption: first candidate was shown
        # Note: salt_swap functionality is preserved but typically unused in this setup

        if logged_action_index >= self.n_arms or logged_action_index >= num_candidates:
            return  # Skip update if index is invalid

        # Extract reward from impression (1 for data, 0 for no click)
        label = impression.get("cost")
        reward = 0
        if label == pos_label:
            reward = 1
        elif label != neg_label:
            pass  # Treat unknown as no click

        # Update statistics for the arm that was pulled
        self.n_pulls[logged_action_index] += 1
        self.sum_rewards[logged_action_index] += reward
        return reward  # Return reward for potential CTR calculation

    def train_on_data(self, training_impressions, salt_swap=False):
        """
        Trains the policy on a list of impression dictionaries.

        Args:
            training_impressions (list): A list of impression dicts from CriteoDataset.
            salt_swap (str/False): Salt for hashing if used to determine logged action.
        """
        print(f"Training SoftMax policy on {len(training_impressions)} impressions...")
        total_clicks = 0
        processed_count = 0
        
        for _idx, impression in enumerate(training_impressions):
            reward = self._update_stats(impression, salt_swap)
            if reward == 1:
                total_clicks += 1
            processed_count += 1
            if _idx % 1000 == 0 and _idx > 0:
                print(f"  Processed {_idx} training impressions...")

        # Calculate and store the average reward across all training data
        if processed_count > 0:
            self.default_avg_reward = total_clicks / processed_count
            print(f"Calculated default average reward (CTR on training split): {self.default_avg_reward:.4f}")
        else:
            print(f"Warning: No impressions processed for CTR calculation. Using default: {self.default_avg_reward}")
        
        # Update total_steps to reflect actual data processed
        self.total_steps = processed_count + 1  # +1 because it starts at 1
        print(f"Training complete. Final total_steps: {self.total_steps}")

    def train(self, training_data_path, force_gzip=False, salt_swap=False, inverse_propensity_in_gold_data=True):
        """
        Simulates the learning process on the training data from a file.
        """
        print(f"Training SoftMax policy from file: {training_data_path}...")
        gold_data = CriteoDataset(training_data_path, isTest=False, isGzip=force_gzip, 
                                inverse_propensity=inverse_propensity_in_gold_data)

        total_clicks = 0
        processed_count = 0
        
        for _idx, _impression in enumerate(gold_data):
            reward = self._update_stats(_impression, salt_swap)
            if reward == 1:
                total_clicks += 1
            processed_count += 1
            if _idx % 10000 == 0 and _idx > 0:
                print(f"  Processed {_idx} training impressions...")

        gold_data.close()
        
        # Calculate and store the average reward across all training data
        if processed_count > 0:
            self.default_avg_reward = total_clicks / processed_count
            print(f"Calculated default average reward (overall CTR): {self.default_avg_reward:.4f}")
        else:
            print(f"Warning: No impressions processed for CTR calculation. Using default: {self.default_avg_reward}")
        
        self.total_steps = processed_count + 1  # +1 because it starts at 1
        print(f"Training complete. Final total_steps: {self.total_steps}")

    def predict(self, candidates):
        """
        Calculates SoftMax scores for each candidate position in an impression.
        This implements the core SoftMax algorithm by:
        1. Computing average reward for each arm (Q-values)
        2. Applying softmax transformation with temperature parameter
        3. Returning raw scores (not probabilities) for ranking
        """
        num_candidates = len(candidates)
        q_values = np.zeros(num_candidates)
        
        # Calculate Q-values (average reward) for each arm
        for j in range(num_candidates):
            if j >= self.n_arms:
                # For arms beyond our tracking capacity, use default reward
                q_values[j] = self.default_avg_reward
            else:
                pulls = self.n_pulls[j]
                if pulls <= 1:
                    # If arm has not been pulled or only once, use default
                    q_values[j] = self.default_avg_reward
                else:
                    # Otherwise use empirical average reward
                    q_values[j] = self.sum_rewards[j] / pulls
        
        # Apply softmax with temperature parameter
        # Note: We use temperature to control exploration/exploitation balance
        # Lower temperature = more exploitation (higher scores for best arms)
        # Higher temperature = more exploration (more uniform scores)
        scores = np.zeros(num_candidates)
        
        # Handle numerical stability by normalizing q_values
        # This prevents overflow/underflow when using very small temperature
        max_q = np.max(q_values)
        scores = np.exp((q_values - max_q) / self.temperature)
        
        # For prediction, we return the raw scores (not normalized probabilities)
        # since we only care about the relative ordering for selection
        
        # Replace any NaN or Inf with sensible values
        if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
            print("Warning: NaN/Inf detected in scores. Replacing with fallback values.")
            scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
        
        return scores
