# START OF FILE ucb_policy.py
import numpy as np
import math
from criteo_dataset import CriteoDataset
import utils # Assuming utils.py contains compute_integral_hash if needed

class UCBPolicy:
    """
    Implements the UCB1 algorithm treating each candidate position as an arm.
    Learns from logged data by updating counts for the logged action's position.
    """
    def __init__(self, n_arms, c=2.0):
        """
        Initializes the UCB policy.

        Args:
            n_arms (int): The maximum number of arms (candidate positions) expected.
            c (float): The exploration parameter (often sqrt(2) or 2).
        """
        self.n_arms = n_arms
        self.c = c # Exploration parameter
        self.n_pulls = np.ones(n_arms)
        self.sum_rewards = np.zeros(n_arms)
        self.total_steps = 1 # Total number of training impressions processed
        self.default_avg_reward = 0.01

    def _update_stats(self, impression, salt_swap=False):
        """Internal helper to update stats for a single impression."""
        self.total_steps += 1
        num_candidates = len(impression.get("candidates", []))
        if num_candidates == 0:
            return # Skip impressions with no candidates

        pos_label = 0.001
        neg_label = 0.999

        # Determine the index of the action that was logged
        logged_action_index = 0
        if salt_swap:
            if not utils or not hasattr(utils, 'compute_integral_hash'):
                raise ImportError("utils.compute_integral_hash needed for salt_swap but not found/imported.")
            logged_action_index = utils.compute_integral_hash(impression['id'], salt_swap, num_candidates)
        else:
            logged_action_index = 0 # Default assumption

        if logged_action_index >= self.n_arms or logged_action_index >= num_candidates:
            # print(f"Warning: Logged action index {logged_action_index} out of bounds ({self.n_arms}, {num_candidates}). Skipping.")
            return # Skip update if index is invalid

        label = impression.get("cost")
        reward = 0
        if label == pos_label:
            reward = 1
        elif label != neg_label:
            # print(f"Warning: Unknown cost label {label} for impression {impression['id']}. Assuming no click.")
            pass # Treat unknown as no click

        self.n_pulls[logged_action_index] += 1
        self.sum_rewards[logged_action_index] += reward
        return reward # Return reward for potential CTR calculation

    def train_on_data(self, training_impressions, salt_swap=False):
        """
        Trains the policy on a list of impression dictionaries.

        Args:
            training_impressions (list): A list of impression dicts from CriteoDataset.
            salt_swap (str/False): Salt for hashing if used to determine logged action.
        """
        print(f"Training UCB policy on {len(training_impressions)} impressions...")
        total_clicks = 0
        processed_count = 0
        for _idx, impression in enumerate(training_impressions):
            reward = self._update_stats(impression, salt_swap)
            if reward == 1:
                total_clicks += 1
            processed_count += 1
            if _idx % 1000 == 0 and _idx > 0:
                 print(f"  Processed {_idx} training impressions...")

        if processed_count > 0:
             self.default_avg_reward = total_clicks / processed_count
             print(f"Calculated default average reward (CTR on training split): {self.default_avg_reward:.4f}")
        else:
             print(f"Warning: No impressions processed for CTR calculation. Using default: {self.default_avg_reward}")
        # Adjust total_steps as it's incremented in _update_stats
        self.total_steps = processed_count + 1 # +1 because it starts at 1
        print(f"Training complete. Final total_steps for UCB calc: {self.total_steps}")


    def train(self, training_data_path, force_gzip=False, salt_swap=False, inverse_propensity_in_gold_data=True):
        """
        Simulates the learning process on the training data from a file.
        DEPRECATED in favor of train_on_data for tuning script, but kept for standalone use.
        """
        print(f"Training UCB policy from file: {training_data_path}...")
        gold_data = CriteoDataset(training_data_path, isTest=False, isGzip=force_gzip, inverse_propensity=inverse_propensity_in_gold_data)

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
        if processed_count > 0:
             self.default_avg_reward = total_clicks / processed_count
             print(f"Calculated default average reward (overall CTR): {self.default_avg_reward:.4f}")
        else:
             print(f"Warning: No impressions processed for CTR calculation. Using default: {self.default_avg_reward}")
        self.total_steps = processed_count + 1 # +1 because it starts at 1
        print(f"Training complete. Final total_steps for UCB calc: {self.total_steps}")

    def predict(self, candidates):
        """
        Calculates UCB scores for each candidate position in an impression.
        (Keep the predict method exactly as it was in the previous correct version)
        """
        num_candidates = len(candidates)
        scores = np.zeros(num_candidates)

        current_total_steps = max(self.total_steps, 2) # Ensure >= 2 for log
        log_total_steps = math.log(current_total_steps)

        for j in range(num_candidates):
            if j >= self.n_arms:
                 avg_reward = self.default_avg_reward
                 bonus = self.c * math.sqrt(log_total_steps / 1.0) # Treat as pulled once
                 scores[j] = avg_reward + bonus
            else:
                pulls = self.n_pulls[j]
                if pulls <= 1:
                    avg_reward = self.default_avg_reward
                    bonus = self.c * math.sqrt(log_total_steps / 1.0) # Max bonus
                else:
                    avg_reward = self.sum_rewards[j] / pulls
                    bonus = self.c * math.sqrt(log_total_steps / pulls)
                scores[j] = avg_reward + bonus

        if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
            print("Warning: NaN/Inf detected in scores. Replacing with 0.")
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        return scores

# END OF FILE ucb_policy.py