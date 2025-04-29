import numpy as np
import math

class ThompsonSamplingPolicy:
    """
    Implements the Thompson Sampling algorithm treating each candidate position as an arm.
    Learns from logged data by updating counts for the logged action's position.
    """
    def __init__(self, n_arms):
        """
        Initializes the Thompson Sampling policy.

        Args:
            n_arms (int): The maximum number of arms (candidate positions) expected.
        """
        self.n_arms = n_arms
        self.successes = np.zeros(n_arms)  # Number of successes (clicks) for each arm
        self.failures = np.zeros(n_arms)  # Number of failures (no clicks) for each arm

    def _update_stats(self, impression):
        """Internal helper to update stats for a single impression."""
        num_candidates = len(impression.get("candidates", []))
        if num_candidates == 0:
            return  # Skip impressions with no candidates

        # Determine the index of the action that was logged
        logged_action_index = impression.get("logged_action", 0)
        if logged_action_index >= self.n_arms or logged_action_index >= num_candidates:
            return  # Skip update if index is invalid

        label = impression.get("cost")
        if label == 0:  # No click
            self.failures[logged_action_index] += 1
        elif label == 1:  # Click
            self.successes[logged_action_index] += 1

    def train_on_data(self, training_impressions):
        """
        Trains the policy on a list of impression dictionaries.

        Args:
            training_impressions (list): A list of impression dicts from the dataset.
        """
        print(f"Training Thompson Sampling policy on {len(training_impressions)} impressions...")
        for impression in training_impressions:
            self._update_stats(impression)
        print("Training complete.")

    def predict(self, candidates):
        """
        Calculates Thompson Sampling scores for each candidate position in an impression.

        Args:
            candidates (list): List of candidate positions for the impression.

        Returns:
            list: Scores for each candidate position.
        """
        num_candidates = len(candidates)
        scores = np.zeros(num_candidates)

        for j in range(num_candidates):
            if j >= self.n_arms:
                scores[j] = np.random.beta(1, 1)  # Default prior for unseen arms
            else:
                scores[j] = np.random.beta(self.successes[j] + 1, self.failures[j] + 1)

        return scores