#!/usr/bin/env python
"""
Tune the epsilon parameter for epsilon-greedy approach by comparing IPS and SNIPS scores.
This script runs the epsilon-greedy algorithm with different epsilon values,
evaluates each one, and identifies the optimal value based on the highest IPS/SNIPS scores.
"""

import argparse
import logging
import numpy as np
import os
import subprocess
import json
import matplotlib.pyplot as plt
import sys
from multiprocessing import Pool, cpu_count

# Add the project root directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from compute_score import grade_predictions

logger = logging.getLogger(__name__)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def setup_logging(log_path, log_level):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    
    logging.basicConfig(
        filename=log_path,
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=numeric_level,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(numeric_level)
    logger.addHandler(console)

def run_epsilon_greedy(epsilon, args):
    """
    Run epsilon-greedy algorithm with specified epsilon value.
    
    Args:
        epsilon: Epsilon value to use
        args: Command-line arguments
        
    Returns:
        Dictionary with epsilon value and output file path
    """
    output_path = f"data/epsilon_greedy_predictions_{epsilon:.3f}.txt.gz"
    
    command = [
        "python3", "epsilon_greedy.py",  # Changed 'python' to 'python3'
        "--epsilon", str(epsilon),
        "--train-path", args.train_path,
        "--output-path", output_path,
        "--test-data-path", args.test_data_path,
        "--solver", args.solver,
        "--penalty", args.penalty,
        "--C", str(args.C),
        "--max-iter", str(args.max_iter),
        "--random-seed", str(args.random_seed),
        "--split-ratio", str(args.split_ratio),
        "--log-path", f"epsilon_greedy_{epsilon:.3f}.log",
        "--log-level", args.log_level
    ]
    
    logger.info(f"Running epsilon-greedy with epsilon={epsilon:.3f}")
    subprocess.run(command, check=True)
    
    return {
        "epsilon": epsilon,
        "output_path": output_path
    }

def evaluate_epsilon(result, args):
    """
    Evaluate predictions generated with a specific epsilon value.
    
    Args:
        result: Dictionary with epsilon and output_path
        args: Command-line arguments
        
    Returns:
        Dictionary with evaluation results
    """
    epsilon = result["epsilon"]
    output_path = result["output_path"]
    
    logger.info(f"Evaluating predictions for epsilon={epsilon:.3f}")
    
    evaluation = grade_predictions(
        output_path,
        args.test_data_path,
        force_gzip=True
    )
    
    evaluation["epsilon"] = epsilon
    logger.info(f"Epsilon={epsilon:.3f}, IPS={evaluation['ips']:.4f}, SNIPS={evaluation['snips']:.4f}")
    
    return evaluation

def run_epsilon_tuning(args):
    """
    Run epsilon tuning process with multiple epsilon values.
    
    Args:
        args: Command-line arguments
    """
    # Generate epsilon values to try
    epsilon_values = np.linspace(args.epsilon_min, args.epsilon_max, args.num_epsilons)
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Check if we need to run in parallel
    results = []
    
    if args.parallel and args.num_epsilons > 1:
        # Run epsilon-greedy in parallel
        with Pool(processes=min(args.num_epsilons, cpu_count())) as pool:
            run_results = []
            for epsilon in epsilon_values:
                run_results.append(pool.apply_async(run_epsilon_greedy, (epsilon, args)))
            
            # Collect results
            for res in run_results:
                results.append(res.get())
    else:
        # Run sequentially
        for epsilon in epsilon_values:
            results.append(run_epsilon_greedy(epsilon, args))
    
    # Evaluate all results
    evaluations = []
    for result in results:
        evaluations.append(evaluate_epsilon(result, args))
    
    # Find best epsilon based on IPS and SNIPS
    best_ips = max(evaluations, key=lambda x: x["ips"])
    best_snips = max(evaluations, key=lambda x: x["snips"])
    
    logger.info("\n=== Epsilon Tuning Results ===")
    logger.info(f"Best epsilon for IPS: {best_ips['epsilon']:.3f} (IPS={best_ips['ips']:.4f})")
    logger.info(f"Best epsilon for SNIPS: {best_snips['epsilon']:.3f} (SNIPS={best_snips['snips']:.4f})")
    
    # Save results to JSON - using the custom encoder
    with open("results/epsilon_tuning_results.json", "w") as f:
        json.dump(evaluations, f, indent=4, cls=NumpyEncoder)
    
    # Plot results
    plot_results(evaluations)
    
    return evaluations

def plot_results(evaluations):
    """
    Plot IPS and SNIPS scores for different epsilon values.
    
    Args:
        evaluations: List of evaluation results
    """
    epsilons = [eval_result["epsilon"] for eval_result in evaluations]
    ips_scores = [eval_result["ips"] for eval_result in evaluations]
    snips_scores = [eval_result["snips"] for eval_result in evaluations]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(epsilons, ips_scores, 'o-', label='IPS Score')
    plt.plot(epsilons, snips_scores, 's-', label='SNIPS Score')
    
    plt.xlabel('Epsilon')
    plt.ylabel('Score (x10^4)')
    plt.title('IPS and SNIPS Scores vs. Epsilon')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Find best epsilons and highlight them
    best_ips_idx = np.argmax(ips_scores)
    best_snips_idx = np.argmax(snips_scores)
    
    plt.axvline(x=epsilons[best_ips_idx], color='blue', linestyle='--', alpha=0.5)
    plt.axvline(x=epsilons[best_snips_idx], color='orange', linestyle='--', alpha=0.5)
    
    plt.annotate(f'Best IPS ε={epsilons[best_ips_idx]:.3f}', 
                 xy=(epsilons[best_ips_idx], ips_scores[best_ips_idx]),
                 xytext=(epsilons[best_ips_idx]+0.05, ips_scores[best_ips_idx]),
                 arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10)
                 
    plt.annotate(f'Best SNIPS ε={epsilons[best_snips_idx]:.3f}', 
                 xy=(epsilons[best_snips_idx], snips_scores[best_snips_idx]),
                 xytext=(epsilons[best_snips_idx]+0.05, snips_scores[best_snips_idx]),
                 arrowprops=dict(facecolor='orange', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10)
    
    plt.savefig('results/epsilon_tuning_plot.png')
    logger.info("Saved results plot to results/epsilon_tuning_plot.png")

def main():
    parser = argparse.ArgumentParser(
        description="Tune epsilon parameter for epsilon-greedy algorithm by comparing IPS and SNIPS scores."
    )
    parser.add_argument(
        "--train-path",
        default="data/criteo_train_small.txt",
        help="Path to the training data file.",
    )
    parser.add_argument(
        "--test-data-path",
        default="data/criteo_test_split.txt.gz",
        help="Path to the test data ground truth file.",
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.00,
        help="Minimum epsilon value to test.",
    )
    parser.add_argument(
        "--epsilon-max",
        type=float,
        default=0.99,
        help="Maximum epsilon value to test.",
    )
    parser.add_argument(
        "--num-epsilons",
        type=int,
        default=100,
        help="Number of epsilon values to test between min and max.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run epsilon-greedy in parallel for different epsilon values.",
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
        "--log-path", default="epsilon_tuning.log", help="Path to log file."
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
    setup_logging(args.log_path, args.log_level)
    
    logger.info("Starting epsilon tuning")
    logger.info(f"Testing epsilon values from {args.epsilon_min} to {args.epsilon_max} "
                f"({args.num_epsilons} values)")
    
    # Run the tuning process
    run_epsilon_tuning(args)
    
    logger.info("Epsilon tuning completed")

if __name__ == "__main__":
    main()