import argparse
import gzip
import logging
import numpy as np
import scipy.sparse
from sklearn.linear_model import LogisticRegression
from criteo_dataset import CriteoDataset # Assuming this class can be iterated upon

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression on training data and predict on the same data."
    )
    parser.add_argument(
        "--data-path",
        default="data/criteo_train_small.txt",
        help="Path to the training data file (uncompressed).",
    )
    parser.add_argument(
        "--output-path",
        default="data/predictions_on_train.txt.gz",
        help="Path to save the predictions.",
    )
    parser.add_argument(
        "--log-path", default="train_predict_on_train.log", help="Path to log file."
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (DEBUG, INFO, ...)."
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


    args = parser.parse_args()

    logger.info("Loading training data from %s", args.data_path)
    # Instantiate CriteoDataset - assuming it processes the file and makes data accessible
    # We need features (X) and outcomes (cost) for training,
    # and features (X) grouped by impression for prediction.
    # CriteoDataset structure based on parser_example.py and previous attempts:
    # It seems designed to be iterated over, yielding blocks. Let's collect data first.

    all_features = []
    all_costs = []
    impression_data = [] # List to store tuples: (impression_id, start_idx, num_candidates)
    current_idx = 0

    logger.info("First pass: Reading data and collecting features/metadata...")
    dataset_iterator = CriteoDataset(args.data_path, isTest=False)
    for impression_block in dataset_iterator:
        impression_id = impression_block['id']
        candidates = impression_block['candidates'] # List of feature dicts
        cost = impression_block['cost']
        num_candidates = len(candidates)

        impression_data.append((impression_id, current_idx, num_candidates))

        # Store features for all candidates
        all_features.extend(candidates)
        # Store cost only for the first candidate (logged action)
        all_costs.append(cost)
        # Pad costs for non-logged candidates (won't be used for training y)
        all_costs.extend([np.nan] * (num_candidates - 1))

        current_idx += num_candidates

    dataset_iterator.close()
    logger.info(f"Finished reading. Total examples: {len(all_features)}, Impressions: {len(impression_data)}")

    # Convert features to sparse matrix (using logic similar to parser_example.py)
    logger.info("Converting features to sparse matrix...")
    rows = []
    cols = []
    data = []
    for i, feat_dict in enumerate(all_features):
        for feat_id in feat_dict:
            rows.append(i)
            cols.append(feat_id)
            data.append(1) # Assuming binary features

    # Infer shape - find the max feature index + 1
    num_features = max(cols) + 1 if cols else 0
    X_all = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(len(all_features), num_features))
    logger.info(f"Full feature matrix shape: {X_all.shape}")

    # Prepare training data (features of first candidate + its outcome)
    logger.info("Preparing training features and labels...")
    train_indices = [imp_data[1] for imp_data in impression_data] # Index of the first candidate for each impression
    X_train = X_all[train_indices]
    # Get costs corresponding to the first candidates and convert to binary label
    y_train_costs = [all_costs[i] for i in train_indices]
    y_train = np.array([1.0 if c == 0.001 else 0.0 for c in y_train_costs])

    logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Number of positive samples in training: {int(np.sum(y_train))}")

    # Train the model
    logger.info("Training Logistic Regression model...")
    logger.info(f"Solver: {args.solver}, Penalty: {args.penalty}, C: {args.C}, MaxIter: {args.max_iter}")
    model = LogisticRegression(
        solver=args.solver,
        penalty=args.penalty,
        C=args.C,
        random_state=42,
        max_iter=args.max_iter,
        class_weight='balanced' # Often helpful for imbalanced click data
    )
    model.fit(X_train, y_train)
    logger.info("Training complete.")

    # Generate predictions
    logger.info("Generating predictions on the training data...")
    predictions = {}
    processed_count = 0
    
    # Open the output file
    logger.info("Saving predictions to %s", args.output_path)
    with gzip.open(args.output_path, "wb") as output_file:
        for impression_id, start_idx, num_candidates in impression_data:
            if (processed_count + 1) % 10000 == 0:
                logger.info("Processed %d impressions for prediction", processed_count + 1)

            # Get features for all candidates in this impression
            candidate_indices = list(range(start_idx, start_idx + num_candidates))
            X_candidates = X_all[candidate_indices]

            # Predict probability of click (class 1) for all candidates
            scores = model.predict_proba(X_candidates)[:, 1]
            
            # Format the prediction line
            predictions_formatted = ["{}:{}".format(idx, score) for idx, score in enumerate(scores)]
            prediction_line = "{};{}\n".format(impression_id, ",".join(predictions_formatted))
            
            # Write to output file
            output_file.write(prediction_line.encode())  # Encode for Python 3 compatibility in binary mode
            processed_count += 1

    logger.info("Successfully saved predictions to %s", args.output_path)

if __name__ == "__main__":
    main()