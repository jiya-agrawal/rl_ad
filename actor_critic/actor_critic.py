#!/usr/bin/env python
"""
Actor-Critic implementation for the Criteo dataset contextual bandit problem.

This implementation uses a neural network with two heads:
1. Actor: Policy network that selects actions (ads to display)
2. Critic: Value network that estimates the expected reward

The Actor-Critic algorithm combines policy-based and value-based reinforcement learning.
"""

import argparse
import gzip
import logging
import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import defaultdict, deque

# Add the project root directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from criteo_dataset import CriteoDataset
from utils import dump_impression

logger = logging.getLogger(__name__)

class ActorCriticNetwork(nn.Module):
    """
    Neural network with two heads: Actor (policy) and Critic (value)
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        Initialize the Actor-Critic network.
        
        Args:
            input_dim (int): Dimension of the input features
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Dimension of the output (number of actions)
        """
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extraction layers
        self.feature_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # Actor head (policy network)
        self.actor_head = nn.Linear(prev_dim, output_dim)
        
        # Critic head (value network)
        self.critic_head = nn.Linear(prev_dim, 1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            action_probs: Action probabilities from actor
            value: Value estimate from critic
        """
        # Pass through shared layers
        for layer in self.feature_layers:
            x = F.relu(layer(x))
        
        # Actor: Action probabilities using softmax
        action_logits = self.actor_head(x)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic: Value estimate
        value = self.critic_head(x)
        
        return action_probs, value
    
    def get_action(self, state, explore=True):
        """
        Sample action from the policy distribution.
        
        Args:
            state: Input state tensor
            explore (bool): If True, sample from distribution; otherwise, pick most likely action
            
        Returns:
            action: Selected action
            log_prob: Log probability of the selected action
            value: Value estimate
        """
        action_probs, value = self.forward(state)
        
        if explore:
            # Sample action from the probability distribution
            distribution = Categorical(action_probs)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
        else:
            # Pick the most likely action (exploitation)
            action = torch.argmax(action_probs, dim=-1)
            log_prob = torch.log(action_probs.gather(-1, action.unsqueeze(-1))).squeeze(-1)
            
        return action.item(), log_prob, value


class ActorCriticAgent:
    """
    Agent implementing actor-critic algorithm for the Criteo dataset.
    """
    def __init__(self, feature_dim, hidden_dims, learning_rate=0.001, gamma=0.99):
        """
        Initialize the Actor-Critic agent.
        
        Args:
            feature_dim (int): Dimension of the input features
            hidden_dims (list): List of hidden layer dimensions
            learning_rate (float): Learning rate for optimizer
            gamma (float): Discount factor
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCriticNetwork(feature_dim, hidden_dims, 1).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        
        # Buffer for experience storage
        self.states = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
    def select_action(self, state, num_candidates, explore=True):
        """
        Select an action given the current state.
        
        Args:
            state: Current state
            num_candidates: Number of candidates to pick from
            explore (bool): Whether to explore or exploit
            
        Returns:
            action: Selected action
            action_probs: Probabilities for all actions (numpy array)
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.model(state_tensor.unsqueeze(0))
        
        # Create a probability array for num_candidates
        # Since our model outputs a fixed size, we need to ensure we have enough elements
        # for the actual number of candidates
        output_probs = np.zeros(max(num_candidates, action_probs.shape[1]))
        
        # Copy the probabilities from the model output
        output_probs[:action_probs.shape[1]] = action_probs.squeeze().cpu().numpy()
        
        # We are only interested in the action probabilities for the available candidates
        valid_probs = output_probs[:num_candidates]
        
        # Normalize to ensure they sum to 1
        if np.sum(valid_probs) > 0:
            valid_probs = valid_probs / np.sum(valid_probs)
        else:
            # If all probabilities are 0, set uniform distribution
            valid_probs = np.ones(num_candidates) / num_candidates
        
        if explore:
            # Sample from the distribution
            action = np.random.choice(num_candidates, p=valid_probs)
        else:
            # Pick the most likely action
            action = np.argmax(valid_probs)
            
        # Return the selected action and all probabilities
        return action, valid_probs
    
    def update_policy(self, optimizer):
        """
        Update the policy and value function using collected experiences.
        
        Args:
            optimizer: Optimizer to use for the update
        """
        if len(self.states) == 0:
            return 0.0, 0.0  # Nothing to update
            
        # Convert lists to numpy arrays first, then to tensors
        states_np = np.array(self.states)
        states = torch.FloatTensor(states_np).to(self.device)
        log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        
        # We need to recompute values to maintain the computation graph
        action_probs, values = self.model(states)
        values = values.squeeze(-1)
        
        # Calculate returns (discounted sum of future rewards)
        returns = []
        R = 0
        for r in reversed(rewards.cpu().numpy()):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns for stability
        if len(returns) > 1:  # Only normalize if we have more than one sample
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate advantage (how much better was the action than expected)
        advantages = returns.detach() - values
        
        # Calculate losses
        actor_loss = -(torch.FloatTensor(log_probs).to(self.device) * advantages).mean()
        critic_loss = F.smooth_l1_loss(values, returns)
        
        # Combined loss
        loss = actor_loss + critic_loss
        
        # Update the network
        optimizer.zero_grad()
        loss.backward()
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        optimizer.step()
        
        # Clear buffers
        self.states = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
        return actor_loss.item(), critic_loss.item()


def convert_features_to_tensor(features, max_feature_id):
    """
    Convert sparse feature dictionaries to one-hot encoded tensors.
    For efficiency, consider using sparse tensors for large feature spaces.
    
    Args:
        features (list): List of feature dictionaries
        max_feature_id (int): Maximum feature ID
        
    Returns:
        tensor: Feature tensor
    """
    # For this implementation, we'll use a simple bag-of-words representation
    tensor = np.zeros(max_feature_id + 1)
    for feat_id in features:
        tensor[feat_id] = 1
    return tensor


def main():
    parser = argparse.ArgumentParser(
        description="Train an Actor-Critic model on Criteo dataset for contextual bandit problem"
    )
    parser.add_argument(
        "--train-path",
        default="data/criteo_train.txt",
        help="Path to the training data file.",
    )
    parser.add_argument(
        "--output-path",
        default="data/actor_critic_predictions_on_test.txt.gz",
        help="Path to save the predictions.",
    )
    parser.add_argument(
        "--test-data-path",
        default="data/criteo_test_split.txt.gz",
        help="Path to save the 20% test data as ground truth.",
    )
    parser.add_argument(
        "--split-ratio", type=float, default=0.8, help="Train-test split ratio (default: 0.8)."
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=[128, 64],
        help="Hidden layer dimensions for the neural network.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor for future rewards."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--train-epochs", type=int, default=3, help="Number of training epochs."
    )
    parser.add_argument(
        "--log-path", default="actor_critic.log", help="Path to log file."
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (DEBUG, INFO, ...)."
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility."
    )
    
    args = parser.parse_args()
    
    # Set up logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)
    
    logging.basicConfig(
        filename=args.log_path,
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=numeric_level,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(numeric_level)
    logger.addHandler(console)
    
    # Set random seed for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    
    # First pass: collect all impression IDs to create the split
    impression_ids = []
    logger.info(f"Reading impression IDs from {args.train_path}")
    dataset_iterator = CriteoDataset(args.train_path, isTest=False)
    for impression_block in dataset_iterator:
        impression_ids.append(impression_block['id'])
    dataset_iterator.close()
    
    # Create the train-test split
    num_impressions = len(impression_ids)
    logger.info(f"Total impressions found: {num_impressions}")
    
    # Shuffle and split impression IDs
    random.shuffle(impression_ids)
    train_size = int(num_impressions * args.split_ratio)
    train_impression_ids = set(impression_ids[:train_size])
    test_impression_ids = set(impression_ids[train_size:])
    
    logger.info(f"Split data: {len(train_impression_ids)} training impressions, {len(test_impression_ids)} test impressions")
    
    # Second pass: collect and process features
    # First, determine the maximum feature ID to properly size our feature vectors
    max_feature_id = 0
    dataset_iterator = CriteoDataset(args.train_path, isTest=False)
    for impression_block in dataset_iterator:
        for candidate in impression_block['candidates']:
            if candidate:
                max_feature_id = max(max_feature_id, max(candidate.keys(), default=0))
    dataset_iterator.close()
    
    logger.info(f"Maximum feature ID found: {max_feature_id}")
    
    # Third pass: Save test data and prepare training data
    train_impressions = []
    os.makedirs(os.path.dirname(args.test_data_path), exist_ok=True)
    with gzip.open(args.test_data_path, 'wb') as test_file:
        dataset_iterator = CriteoDataset(args.train_path, isTest=False)
        for impression_block in dataset_iterator:
            impression_id = impression_block['id']
            candidates = impression_block['candidates']
            cost = impression_block['cost']
            propensity = impression_block['propensity']
            
            if impression_id in test_impression_ids:
                # Save test data for ground truth
                impression_data = {
                    'id': impression_id,
                    'candidates': candidates,
                    'cost': cost,
                    'propensity': propensity
                }
                test_file.write((dump_impression(impression_data) + '\n').encode())
            else:
                # Store training impressions
                train_impressions.append((impression_id, candidates, cost, propensity))
        
        dataset_iterator.close()
    
    logger.info(f"Test data saved to {args.test_data_path}")
    logger.info(f"Training impressions collected: {len(train_impressions)}")
    
    # Initialize actor-critic agent
    feature_dim = max_feature_id + 1  # +1 because feature IDs are 0-indexed
    hidden_dims = args.hidden_dims
    agent = ActorCriticAgent(
        feature_dim=feature_dim,
        hidden_dims=hidden_dims,
        learning_rate=args.learning_rate,
        gamma=args.gamma
    )
    
    logger.info(f"Initialized Actor-Critic model with feature_dim={feature_dim}, hidden_dims={hidden_dims}")
    logger.info(f"Training for {args.train_epochs} epochs with batch size {args.batch_size}")
    
    # Training loop
    for epoch in range(args.train_epochs):
        random.shuffle(train_impressions)
        total_reward = 0
        correct_actions = 0
        total_actions = 0
        
        for i, (impression_id, candidates, cost, propensity) in enumerate(train_impressions):
            # Features for all candidates
            candidate_features = [
                convert_features_to_tensor(candidate, max_feature_id) 
                for candidate in candidates
            ]
            
            # Select first candidate's features as state
            state = candidate_features[0]
            
            # Get action from the agent
            action, action_probs = agent.select_action(state, len(candidates), explore=True)
            
            # Get state value estimate
            state_tensor = torch.FloatTensor(state).to(agent.device)
            with torch.no_grad():
                _, value = agent.model(state_tensor.unsqueeze(0))
            state_value = value.item()
            
            # Calculate reward (binary: 1 if clicked, 0 if not)
            # In the Criteo dataset, cost=0.001 means the ad was clicked
            reward = 1.0 if cost == 0.001 and action == 0 else 0.0
            
            # Off-policy correction using propensity score
            # The logged action was always the first candidate (index 0)
            if action == 0:  # If our action matches the logged action
                corrected_reward = reward / propensity
            else:
                corrected_reward = 0  # We can only learn from logged actions in this setting
            
            # Update agent's experience buffer
            agent.states.append(state)
            agent.rewards.append(corrected_reward)
            agent.log_probs.append(np.log(action_probs[action] + 1e-9))  # Add small constant for numerical stability
            agent.values.append(state_value)  # Use the actual value estimate from the critic
            agent.dones.append(False)
            
            total_reward += corrected_reward
            if action == 0 and cost == 0.001:
                correct_actions += 1
            total_actions += 1
            
            # Update policy every batch_size steps
            if (i + 1) % args.batch_size == 0:
                actor_loss, critic_loss = agent.update_policy(agent.optimizer)
                if (i + 1) % (args.batch_size * 10) == 0:
                    logger.info(f"Epoch {epoch+1}, Impression {i+1}/{len(train_impressions)}, " +
                               f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, " +
                               f"Reward: {total_reward/args.batch_size:.4f}, " +
                               f"Accuracy: {correct_actions/(total_actions+1e-9):.4f}")
                    total_reward = 0
                    correct_actions = 0
                    total_actions = 0
        
        logger.info(f"Completed epoch {epoch+1}/{args.train_epochs}")
    
    logger.info("Training completed. Generating predictions on test data...")
    
    # Generate predictions on test data
    test_dataset = CriteoDataset(args.test_data_path, isGzip=True, isTest=False)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    with gzip.open(args.output_path, 'wb') as output_file:
        for i, impression_block in enumerate(test_dataset):
            impression_id = impression_block['id']
            candidates = impression_block['candidates']
            
            # Features for all candidates
            candidate_features = [
                convert_features_to_tensor(candidate, max_feature_id) 
                for candidate in candidates
            ]
            
            # Select first candidate's features as state
            state = candidate_features[0]
            
            # Get action probabilities from the agent (no exploration during evaluation)
            _, action_probs = agent.select_action(
                state, len(candidates), explore=False
            )
            
            # For batch prediction, we need scores for all candidates
            # Since our network is designed for single-candidate scoring,
            # we'll use the first candidate's features for context and predict scores
            # for each candidate by position
            scores = action_probs[:len(candidates)]
            
            # Format and write the prediction line
            predictions_formatted = [f"{idx}:{score}" for idx, score in enumerate(scores)]
            prediction_line = f"{impression_id};{','.join(predictions_formatted)}\n"
            output_file.write(prediction_line.encode())
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i+1} test impressions")
    
    test_dataset.close()
    logger.info(f"Successfully saved predictions to {args.output_path}")
    logger.info(f"To evaluate the model, run: python compute_score.py --predictions-path {args.output_path} --gold-labels-path {args.test_data_path}")


if __name__ == "__main__":
    main()