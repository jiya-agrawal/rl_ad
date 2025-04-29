#!/usr/bin/env python
import argparse
import gzip
import logging
import numpy as np
import scipy.sparse
import random
import os
import sys
from sklearn.linear_model import LogisticRegression

# Add the project root directory to the Python path so we can import from competition
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from criteo_dataset import CriteoDataset
from utils import dump_impression

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Train a model with Epsilon-Greedy approach on 80% of data and predict on 20% test data"
    )
    parser.add_argument(
        "--train-path",
        default="data/criteo_train_small.txt",
        help="Path to the training data file.",
    )
    parser.add_argument(
        "--output-path",
        default="data/epsilon_greedy_predictions_on_test.txt.gz",
        help="Path to save the predictions.",
    )
    parser.add_argument(
        "--test-data-path",
        default="data/criteo_test_split.txt.gz",
        help="Path to save the 20% test data as ground truth.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Epsilon value for exploration (between 0 and 1).",
    )
    parser.add_argument(
        "--solver",
        default="liblinear",
        help="Solver for Logistic Regression (e.g., 'liblinear', 'saga').",
    )
    parser.add_argument(
        "--penalty", default="l2", help="Penalty for Logistic Regression ('l1', 'l2')."
    )
    parser.add_argument(
        "--C", type=float, default=1.0, help="Inverse regularization strength."
    )
    parser.add_argument(
        "--max-iter", type=int, default=1000, help="Max iterations for solver."
    )
    parser.add_argument(
        "--log-path", default="epsilon_greedy.log", help="Path to log file."
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (DEBUG, INFO, ...)."
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--split-ratio", type=float, default=0.8, help="Train-test split ratio (default: 0.8)."
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
    
    # Validate epsilon
    if args.epsilon < 0 or args.epsilon > 1:
        logger.error("Epsilon must be between 0 and 1")
        return

    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    logger.info(f"Using epsilon = {args.epsilon} for the epsilon-greedy approach")
    
    # First pass: collect all impression IDs to create the split
    impression_ids = []
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
    
    # Second pass: collect data while separating into train and test sets
    all_train_features = []
    all_train_costs = []
    train_impression_data = []  # (impression_id, start_idx, num_candidates)
    current_train_idx = 0
    
    all_test_features = []
    all_test_costs = []
    test_impression_data = []  # (impression_id, start_idx, num_candidates)
    current_test_idx = 0
    
    # Save test data to a separate file for ground truth
    os.makedirs(os.path.dirname(args.test_data_path), exist_ok=True)
    with gzip.open(args.test_data_path, 'wb') as test_file:
        dataset_iterator = CriteoDataset(args.train_path, isTest=False)
        for impression_block in dataset_iterator:
            impression_id = impression_block['id']
            candidates = impression_block['candidates']
            cost = impression_block['cost']
            propensity = impression_block['propensity']
            num_candidates = len(candidates)
            
            # Format and store data differently based on train/test split
            if impression_id in test_impression_ids:
                # Save test data for ground truth
                impression_data = {
                    'id': impression_id,
                    'candidates': candidates,
                    'cost': cost,
                    'propensity': propensity
                }
                test_file.write((dump_impression(impression_data) + '\n').encode())
                
                # Store features and metadata for test set
                test_impression_data.append((impression_id, current_test_idx, num_candidates))
                all_test_features.extend(candidates)
                all_test_costs.append(cost)
                all_test_costs.extend([np.nan] * (num_candidates - 1))
                current_test_idx += num_candidates
            else:
                # Store features and metadata for training set
                train_impression_data.append((impression_id, current_train_idx, num_candidates))
                all_train_features.extend(candidates)
                all_train_costs.append(cost)
                all_train_costs.extend([np.nan] * (num_candidates - 1))
                current_train_idx += num_candidates
                
        dataset_iterator.close()
    
    logger.info(f"Test data saved to {args.test_data_path}")
    logger.info(f"Training features: {len(all_train_features)}, Test features: {len(all_test_features)}")
    
    # Convert features to sparse matrices - TRAINING DATA
    logger.info("Converting training features to sparse matrix...")
    train_rows, train_cols, train_data = [], [], []
    for i, feat_dict in enumerate(all_train_features):
        for feat_id in feat_dict:
            train_rows.append(i)
            train_cols.append(feat_id)
            train_data.append(1)  # Binary features
    
    # Determine feature dimensionality from all features
    all_cols = train_cols.copy()
    for i, feat_dict in enumerate(all_test_features):
        for feat_id in feat_dict:
            all_cols.append(feat_id)
    
    num_features = max(all_cols) + 1 if all_cols else 0
    X_train_all = scipy.sparse.csr_matrix(
        (train_data, (train_rows, train_cols)), 
        shape=(len(all_train_features), num_features)
    )
    
    # Convert features to sparse matrices - TEST DATA
    logger.info("Converting test features to sparse matrix...")
    test_rows, test_cols, test_data = [], [], []
    for i, feat_dict in enumerate(all_test_features):
        for feat_id in feat_dict:
            test_rows.append(i)
            test_cols.append(feat_id)
            test_data.append(1)
    
    X_test_all = scipy.sparse.csr_matrix(
        (test_data, (test_rows, test_cols)), 
        shape=(len(all_test_features), num_features)
    )
    
    # Prepare training data (features of first candidate + its outcome)
    logger.info("Preparing training features and labels...")
    train_indices = [imp_data[1] for imp_data in train_impression_data]  # Index of the first candidate
    X_train = X_train_all[train_indices]
    y_train_costs = [all_train_costs[i] for i in train_indices]
    y_train = np.array([1.0 if c == 0.001 else 0.0 for c in y_train_costs])
    
    logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Number of positive samples in training: {int(np.sum(y_train))}")
    
    # Train the model
    logger.info("Training Logistic Regression model...")
    model = train_model(X_train, y_train, args)
    
    # Generate predictions on the test data using epsilon-greedy approach
    logger.info(f"Generating predictions with epsilon-greedy approach on TEST DATA...")
    
    # Open output file
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with gzip.open(args.output_path, "wb") as output_file:
        processed_count = 0
        
        for impression_id, start_idx, num_candidates in test_impression_data:
            if (processed_count + 1) % 1000 == 0:
                logger.info("Processed %d test impressions", processed_count + 1)
            
            # Get features for all candidates in this impression
            candidate_indices = list(range(start_idx, start_idx + num_candidates))
            X_candidates = X_test_all[candidate_indices]
            
            # Predict probabilities for all candidates
            base_scores = model.predict_proba(X_candidates)[:, 1]  # Probability of class 1 (click)
            
            # Apply epsilon-greedy policy to modify the scores
            scores = epsilon_greedy_policy(base_scores, args.epsilon)
            
            # Format and write the prediction line
            predictions_formatted = [f"{idx}:{score}" for idx, score in enumerate(scores)]
            prediction_line = f"{impression_id};{','.join(predictions_formatted)}\n"
            output_file.write(prediction_line.encode())
            
            processed_count += 1
    
    logger.info(f"Successfully saved predictions to {args.output_path}")
    logger.info(f"Successfully saved test ground truth to {args.test_data_path}")
    logger.info(f"Total test impressions processed: {processed_count}")

def train_model(X_train, y_train, args):
    """Train a logistic regression model."""
    model = LogisticRegression(
        solver=args.solver,
        penalty=args.penalty,
        C=args.C,
        random_state=args.random_seed,
        max_iter=args.max_iter,
        class_weight='balanced'  # Handle class imbalance
    )
    model.fit(X_train, y_train)
    return model

def epsilon_greedy_policy(scores, epsilon):
    """
    Apply epsilon-greedy policy to the scores.
    
    Args:
        scores: Array of predicted click probabilities for candidates
        epsilon: Probability of choosing a random action
    
    Returns:
        Modified scores reflecting the epsilon-greedy policy
    """
    num_candidates = len(scores)
    
    # With probability 1-epsilon: Keep the original scores (exploitation)
    # With probability epsilon: Set a random candidate to have the highest score (exploration)
    if np.random.random() < epsilon:
        # Create exploration scores by giving a high value to one random candidate
        exploration_scores = np.zeros_like(scores)
        random_idx = np.random.randint(0, num_candidates)
        exploration_scores[random_idx] = 1.0
        
        # Blend the original scores with exploration
        return exploration_scores
    else:
        # Exploitation: use model's predicted probabilities
        return scores

if __name__ == "__main__":
    main()