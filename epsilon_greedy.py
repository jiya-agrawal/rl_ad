#!/usr/bin/env python
import argparse
import gzip
import logging
import numpy as np
import scipy.sparse
from sklearn.linear_model import LogisticRegression
from criteo_dataset import CriteoDataset

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Train a model with Epsilon-Greedy approach and predict on the same training data"
    )
    parser.add_argument(
        "--train-path",
        default="data/criteo_train_small.txt",
        help="Path to the training data file.",
    )
    parser.add_argument(
        "--output-path",
        default="data/epsilon_greedy_predictions_on_train.txt.gz",
        help="Path to save the predictions.",
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

    args = parser.parse_args()


    # Validate epsilon
    if args.epsilon < 0 or args.epsilon > 1:
        logger.error("Epsilon must be between 0 and 1")
        return

    logger.info(f"Using epsilon = {args.epsilon} for the epsilon-greedy approach")
    
    # Load and process training data
    logger.info("Loading training data from %s", args.train_path)
    train_features, train_labels, train_impression_data, all_train_features = load_and_process_data(args.train_path)
    
    # Train the base model (logistic regression)
    logger.info("Training Logistic Regression model...")
    model = train_model(train_features, train_labels, args)
    
    # Generate predictions using epsilon-greedy approach on training data
    logger.info("Generating predictions with epsilon-greedy approach ON TRAINING DATA...")
    generate_epsilon_greedy_predictions(model, all_train_features, train_impression_data, args.epsilon, args.output_path)
    
    logger.info("Successfully completed epsilon-greedy predictions on training data")

def load_and_process_data(data_path):
    """Load and process data from the given path."""
    all_features = []
    all_labels = []
    impression_data = []  # (impression_id, start_idx, num_candidates)
    current_idx = 0
    
    dataset_iterator = CriteoDataset(data_path, isTest=False)
    
    for impression_block in dataset_iterator:
        impression_id = impression_block['id']
        candidates = impression_block['candidates']  # List of feature dicts
        cost = impression_block['cost']
        num_candidates = len(candidates)
        
        impression_data.append((impression_id, current_idx, num_candidates))
        
        # Store all candidate features
        all_features.extend(candidates)
        # Store cost only for the first candidate (logged action)
        all_labels.append(1.0 if cost == 0.001 else 0.0)
        # Pad labels for non-logged candidates (won't be used for training)
        all_labels.extend([np.nan] * (num_candidates - 1))
        
        current_idx += num_candidates
    
    dataset_iterator.close()
    
    # Convert features to sparse matrix
    rows, cols, data = [], [], []
    for i, feat_dict in enumerate(all_features):
        for feat_id in feat_dict:
            rows.append(i)
            cols.append(feat_id)
            data.append(1)  # Assuming binary features
    
    # Determine the feature dimension
    num_features = max(cols) + 1 if cols else 0
    X_all = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(len(all_features), num_features))
    
    # For training data, only use the examples with known outcomes
    train_indices = [imp_data[1] for imp_data in impression_data]  # Index of the first candidate for each impression
    X_train = X_all[train_indices]
    y_train = np.array([all_labels[i] for i in train_indices])
    
    return X_train, y_train, impression_data, X_all

def train_model(X_train, y_train, args):
    """Train a logistic regression model."""
    logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Number of positive samples in training: {int(np.sum(y_train))}")
    
    model = LogisticRegression(
        solver=args.solver,
        penalty=args.penalty,
        C=args.C,
        random_state=42,
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
        # For simplicity, we're using exploration_scores directly when exploring
        # In a real system, you might use a weighted blend instead
        return exploration_scores
    else:
        # Exploitation: use model's predicted probabilities
        return scores

def generate_epsilon_greedy_predictions(model, all_features, impression_data, epsilon, output_path):
    """Generate and save predictions using the epsilon-greedy approach."""
    processed_count = 0
    
    with gzip.open(output_path, "wb") as output_file:
        for impression_id, start_idx, num_candidates in impression_data:
            if (processed_count + 1) % 10000 == 0:
                logger.info("Processed %d impressions", processed_count + 1)
            
            # Get features for all candidates in this impression
            candidate_indices = list(range(start_idx, start_idx + num_candidates))
            X_candidates = all_features[candidate_indices]
            
            # Predict probabilities for all candidates
            base_scores = model.predict_proba(X_candidates)[:, 1]  # Probability of class 1 (click)
            
            # Apply epsilon-greedy policy to modify the scores
            scores = epsilon_greedy_policy(base_scores, epsilon)
            
            # Format and write the prediction line
            predictions_formatted = [f"{idx}:{score}" for idx, score in enumerate(scores)]
            prediction_line = f"{impression_id};{','.join(predictions_formatted)}\n"
            output_file.write(prediction_line.encode())
            
            processed_count += 1
    
    logger.info(f"Total impressions processed: {processed_count}")

if __name__ == "__main__":
    main()