#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import argparse
import pickle
import os
from criteo_dataset import CriteoDataset
from sklearn.linear_model import SGDClassifier

class MonteCarloPolicy:
    """
    Monte Carlo Policy Learning using Inverse Propensity Scoring
    """
    def __init__(self, n_features=74000, learning_rate=0.01, reg_strength=0.0001):
        self.n_features = n_features
        self.model = SGDClassifier(
            loss='log_loss',  # Using logistic regression as the base model
            alpha=reg_strength,
            learning_rate='constant',
            eta0=learning_rate,
            warm_start=True
        )
        # Initialize the model with zeros
        self.model.fit(np.zeros((1, self.n_features)), np.array([0]))
        self.coef_backup = self.model.coef_.copy()
    
    def extract_feature_matrix(self, candidates):
        """Convert candidate feature dictionaries to sparse feature matrix"""
        X = np.zeros((len(candidates), self.n_features))
        for i, candidate in enumerate(candidates):
            for feature_idx, feature_val in candidate.items():
                X[i, int(feature_idx)] = float(feature_val)
        return X
    
    def predict(self, candidates):
        """Return scores for each candidate"""
        X = self.extract_feature_matrix(candidates)
        # Get probability estimates (scores between 0 and 1)
        scores = self.model.predict_proba(X)[:, 1]
        return scores
    
    def update(self, impression, chosen_action, importance_weight):
        """Update policy using the IPS-weighted gradient"""
        candidates = impression["candidates"]
        X = self.extract_feature_matrix(candidates)
        
        # Create a zero vector for labels
        y = np.zeros(len(candidates))
        
        # Set the chosen action as positive example (clicked = 1)
        # We'll weight it by the importance weight
        y[chosen_action] = 1
        
        # For the SGDClassifier, we need sample weights
        # We'll use the importance weight for the chosen action
        sample_weight = np.zeros(len(candidates))
        sample_weight[chosen_action] = importance_weight
        
        # Update the model with this weighted example
        self.model.partial_fit(X, y, classes=[0, 1], sample_weight=sample_weight)
    
    def save(self, filepath):
        """Save the model to a file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filepath):
        """Load the model from a file"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)


def compute_importance_weight(selected_prob, propensity, clip_value=10.0):
    """
    Compute clipped importance weight to reduce variance
    """
    if selected_prob <= 0:
        return 0.0
    
    # Importance weight: probability under new policy / probability under logging policy
    importance_weight = selected_prob / propensity
    
    # Clip importance weight to reduce variance
    importance_weight = min(importance_weight, clip_value)
    
    return importance_weight


def train_policy(data_path, model_path, epochs=5, learning_rate=0.01, reg_strength=0.0001, clip_value=10.0):
    """
    Train a policy using Monte Carlo with IPS weighting
    """
    # Initialize the policy
    policy = MonteCarloPolicy(learning_rate=learning_rate, reg_strength=reg_strength)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Create a new dataset iterator for each epoch
        data = CriteoDataset(data_path, isTest=False, inverse_propensity=True)
        
        # Track metrics for this epoch
        total_weight = 0
        total_reward = 0
        total_impressions = 0
        
        for idx, impression in enumerate(data):
            candidates = impression["candidates"]
            cost = impression["cost"]
            propensity = impression["propensity"]
            
            # Get the actual reward (cost is inverted in this dataset: 0.001 for click, 0.999 for no-click)
            # Convert to binary reward: 1 for click, 0 for no-click
            reward = 1.0 if cost < 0.5 else 0.0
            
            # Get scores from current policy for all candidates
            scores = policy.predict(candidates)
            
            # The logged data only shows us the outcome for the selected action
            # In the training data, this is always the first candidate (index 0)
            chosen_action = 0
            
            # Softmax to convert scores to probabilities
            exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
            probs = exp_scores / np.sum(exp_scores)
            
            # Probability of selected action under our current policy
            selected_prob = probs[chosen_action]
            
            # Compute importance weight
            importance_weight = compute_importance_weight(selected_prob, propensity, clip_value)
            
            # Only update policy if there's a non-zero importance weight
            if importance_weight > 0:
                # If the action resulted in a click (reward=1), update policy
                policy.update(impression, chosen_action, importance_weight * reward)
                
                # Track metrics
                total_weight += importance_weight
                total_reward += importance_weight * reward
            
            total_impressions += 1
            
            if idx % 5000 == 0:
                # Print progress
                print(f"Processed {idx} impressions. Current IPS estimate: {total_reward/max(1, total_weight):.6f}")
        
        # End of epoch metrics
        if total_weight > 0:
            ips_estimate = total_reward / total_weight
            print(f"Epoch {epoch+1} finished. IPS estimate: {ips_estimate:.6f}, Processed {total_impressions} impressions")
        else:
            print(f"Epoch {epoch+1} finished. No valid updates. Processed {total_impressions} impressions")
        
        # Save model after each epoch
        epoch_model_path = f"{os.path.splitext(model_path)[0]}_epoch{epoch+1}.pkl"
        policy.save(epoch_model_path)
        print(f"Saved model to {epoch_model_path}")
    
    # Save final model
    policy.save(model_path)
    print(f"Training complete. Final model saved to {model_path}")
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a policy using Monte Carlo methods')
    parser.add_argument('--train_path', dest='train_path', action='store', default='competition/data/criteo_train_small.txt', required=True, help='Path to the training data file')
    parser.add_argument('--model_path', dest='model_path', action='store', default='model',required=True,
                        help='Path to save the trained model')
    parser.add_argument('--epochs', dest='epochs', action='store', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--lr', dest='learning_rate', action='store', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--reg', dest='reg_strength', action='store', type=float, default=0.0001,
                        help='Regularization strength')
    parser.add_argument('--clip', dest='clip_value', action='store', type=float, default=10.0,
                        help='Importance weight clipping value')
    
    args = parser.parse_args()
    
    train_policy(
        args.train_path,
        args.model_path,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        reg_strength=args.reg_strength,
        clip_value=args.clip_value
    )