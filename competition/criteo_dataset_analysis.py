#!/usr/bin/env python
"""
Exploratory Data Analysis of Criteo dataset
This script analyzes the structure of the Criteo dataset, focusing on:
- Number of candidates per impression
- Distribution of costs (clicks)
- Distribution of propensity scores
- Feature statistics
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import time
from tqdm import tqdm

from criteo_dataset import CriteoDataset

def analyze_criteo_dataset(filepath, max_impressions=None):
    """
    Analyze the Criteo dataset structure and candidate distribution
    
    Args:
        filepath: Path to the dataset file
        max_impressions: Maximum number of impressions to process (None for all)
    """
    print(f"Analyzing Criteo dataset: {filepath}")
    start_time = time.time()
    
    # Initialize statistics
    candidates_per_impression = []
    click_rates = []
    propensity_scores = []
    feature_counts = defaultdict(int)
    feature_values = defaultdict(list)
    total_impressions = 0
    clicked_impressions = 0
    
    # Open the dataset
    dataset = CriteoDataset(filepath, isTest=False)
    
    # Process impressions
    try:
        for i, impression in enumerate(tqdm(dataset, desc="Processing impressions")):
            if max_impressions and i >= max_impressions:
                break
                
            total_impressions += 1
            candidates_per_impression.append(len(impression["candidates"]))
            
            # Track click information
            cost = impression["cost"]
            if cost == 0.001:  # Clicked (as per README)
                clicked_impressions += 1
                
            click_rates.append(1 if cost == 0.001 else 0)
            propensity_scores.append(impression["propensity"])
            
            # Analyze features of the first candidate
            if impression["candidates"]:
                first_candidate = impression["candidates"][0]
                for feature_id, value in first_candidate.items():
                    feature_counts[feature_id] += 1
                    feature_values[feature_id].append(value)
    
    except KeyboardInterrupt:
        print("Analysis interrupted. Showing partial results...")
    
    finally:
        dataset.close()
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed_time:.2f} seconds")
    
    results = {
        "Total impressions": total_impressions,
        "Click rate": clicked_impressions / total_impressions if total_impressions > 0 else 0,
        "Candidates per impression": {
            "Min": min(candidates_per_impression) if candidates_per_impression else 0,
            "Max": max(candidates_per_impression) if candidates_per_impression else 0,
            "Mean": np.mean(candidates_per_impression) if candidates_per_impression else 0,
            "Median": np.median(candidates_per_impression) if candidates_per_impression else 0
        },
        "Propensity scores": {
            "Min": min(propensity_scores) if propensity_scores else 0,
            "Max": max(propensity_scores) if propensity_scores else 0,
            "Mean": np.mean(propensity_scores) if propensity_scores else 0,
            "Median": np.median(propensity_scores) if propensity_scores else 0
        },
        "Most common feature IDs": sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        "Num unique features": len(feature_counts)
    }
    
    # Print results
    print("\n=== Criteo Dataset Analysis ===")
    print(f"Total impressions analyzed: {results['Total impressions']}")
    print(f"Click rate: {results['Click rate']:.4f}")
    
    print("\nCandidates per impression:")
    for stat, value in results["Candidates per impression"].items():
        print(f"  {stat}: {value}")
    
    print("\nDistribution of candidates per impression:")
    counter = Counter(candidates_per_impression)
    for count, freq in sorted(counter.items()):
        print(f"  {count} candidates: {freq} impressions ({freq/total_impressions*100:.2f}%)")
    
    print("\nPropensity scores:")
    for stat, value in results["Propensity scores"].items():
        print(f"  {stat}: {value}")
    
    print(f"\nNumber of unique features: {results['Num unique features']}")
    print("\nTop 10 most common feature IDs:")
    for feature_id, count in results["Most common feature IDs"]:
        print(f"  Feature ID {feature_id}: {count} occurrences")
    
    # Generate plots
    plt.figure(figsize=(10, 6))
    plt.hist(candidates_per_impression, bins=max(results["Candidates per impression"]["Max"], 10), alpha=0.7)
    plt.title('Distribution of Candidates per Impression')
    plt.xlabel('Number of Candidates')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('eda_graphs/candidates_distribution.png')
    print("\nSaved candidates distribution plot to eda_graphs/candidates_distribution.png")
    
    # Log-scale plot for candidates distribution
    plt.figure(figsize=(10, 6))
    plt.hist(candidates_per_impression, bins=max(results["Candidates per impression"]["Max"], 10), alpha=0.7)
    plt.title('Distribution of Candidates per Impression (Log Scale)')
    plt.xlabel('Number of Candidates')
    plt.ylabel('Frequency (Log Scale)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('eda_graphs/candidates_distribution_log.png')
    print("Saved log-scale candidates distribution plot to eda_graphs/candidates_distribution_log.png")
    
    plt.figure(figsize=(10, 6))
    plt.hist(propensity_scores, bins=30, alpha=0.7)
    plt.title('Distribution of Propensity Scores')
    plt.xlabel('Propensity Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('eda_graphs/propensity_distribution.png')
    print("Saved propensity distribution plot to eda_graphs/propensity_distribution.png")
    
    # Log-scale plot for propensity scores
    plt.figure(figsize=(10, 6))
    plt.hist(propensity_scores, bins=30, alpha=0.7)
    plt.title('Distribution of Propensity Scores (Log Scale)')
    plt.xlabel('Propensity Score')
    plt.ylabel('Frequency (Log Scale)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('eda_graphs/propensity_distribution_log.png')
    print("Saved log-scale propensity distribution plot to eda_graphs/propensity_distribution_log.png")
    
    return results

if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs('eda_graphs', exist_ok=True)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "data/criteo_train.txt.gz"
    
    # Max impressions to analyze (use None for all)
    max_impressions = None
    
    # Run analysis
    analyze_criteo_dataset(filepath, max_impressions)