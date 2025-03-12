import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import re

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)

def parse_example_line(line):
    """Parse the header line of an example"""
    parts = line.strip().split()
    example_id = int(parts[1].strip(':'))
    hash_id = parts[2]
    was_ad_clicked = int(parts[3])
    propensity = float(parts[4])
    nb_slots = int(parts[5])
    nb_candidates = int(parts[6])
    
    # Parse display features
    display_features = {}
    for feature in parts[7:]:
        if ':' in feature:
            feat_id, feat_val = feature.split(':')
            display_features[int(feat_id)] = float(feat_val) if '.' in feat_val else int(feat_val)
    
    return {
        'example_id': example_id,
        'hash_id': hash_id,
        'was_ad_clicked': was_ad_clicked,
        'propensity': propensity,
        'nb_slots': nb_slots,
        'nb_candidates': nb_candidates,
        'display_features': display_features
    }

def parse_product_line(line):
    """Parse a product line"""
    parts = line.strip().split()
    was_product_clicked = int(parts[0])
    example_id = int(parts[1].split(':')[1])
    
    # Parse product features
    product_features = {}
    for feature in parts[2:]:
        if ':' in feature:
            feat_id, feat_val = feature.split(':')
            feat_id = int(feat_id)
            
            # Handle multi-valued features by storing as lists
            if feat_id in product_features:
                if isinstance(product_features[feat_id], list):
                    product_features[feat_id].append(int(feat_val) if feat_val.isdigit() else feat_val)
                else:
                    product_features[feat_id] = [product_features[feat_id], 
                                               int(feat_val) if feat_val.isdigit() else feat_val]
            else:
                product_features[feat_id] = int(feat_val) if feat_val.isdigit() else feat_val
    
    return {
        'was_product_clicked': was_product_clicked,
        'example_id': example_id,
        'product_features': product_features
    }

def load_data(filepath, max_examples=1000):
    """Load data from file with limit on number of examples"""
    examples = []
    products = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    example_count = 0
    i = 0
    
    while i < len(lines) and example_count < max_examples:
        line = lines[i]
        if line.startswith('example'):
            example_data = parse_example_line(line)
            examples.append(example_data)
            
            # Get the number of candidates for this example
            nb_candidates = example_data['nb_candidates']
            
            # Parse all product lines for this example
            for j in range(1, nb_candidates + 1):
                if i + j < len(lines):
                    product_data = parse_product_line(lines[i + j])
                    product_data['position'] = j if j <= example_data['nb_slots'] else None
                    products.append(product_data)
            
            i += nb_candidates + 1
            example_count += 1
        else:
            i += 1
    
    print(f"Loaded {len(examples)} examples with {len(products)} products")
    return examples, products

def analyze_data(examples, products):
    """Perform analysis on the data"""
    # Convert to DataFrames
    examples_df = pd.DataFrame(examples)
    
    # Extract display features into separate columns
    display_features_df = pd.DataFrame([ex['display_features'] for ex in examples])
    examples_df = pd.concat([examples_df.drop('display_features', axis=1), display_features_df], axis=1)
    
    # Create a DataFrame for products
    products_df = pd.DataFrame([
        {
            'example_id': p['example_id'],
            'was_product_clicked': p['was_product_clicked'],
            'position': p['position'],
            **{f"feat_{k}": v for k, v in p['product_features'].items()}
        }
        for p in products
    ])
    
    # Join with examples to get additional information
    products_df = products_df.merge(
        examples_df[['example_id', 'nb_slots', 'propensity', 'was_ad_clicked']], 
        on='example_id', 
        how='left'
    )
    
    # Flag displayed products
    products_df['was_displayed'] = products_df['position'].notna()
    
    return examples_df, products_df

def create_visualizations(examples_df, products_df):
    """Create visualizations for the data"""
    figures = []
    
    # 1. Distribution of banner sizes (number of slots)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='nb_slots', data=examples_df)
    plt.title('Distribution of Banner Sizes (Number of Slots)')
    plt.xlabel('Number of Slots')
    plt.ylabel('Count')
    figures.append(plt.gcf())
    
    # 2. Click-through rate by banner size
    plt.figure(figsize=(10, 6))
    click_rate_by_size = examples_df.groupby('nb_slots')['was_ad_clicked'].mean()
    click_rate_by_size.plot(kind='bar')
    plt.title('Click-Through Rate by Banner Size')
    plt.xlabel('Number of Slots')
    plt.ylabel('Click-Through Rate')
    plt.grid(axis='y')
    figures.append(plt.gcf())
    
    # 3. Distribution of propensities (log scale)
    plt.figure(figsize=(10, 6))
    sns.histplot(examples_df['propensity'], log_scale=True)
    plt.title('Distribution of Propensities (Log Scale)')
    plt.xlabel('Propensity (Log Scale)')
    plt.ylabel('Count')
    figures.append(plt.gcf())
    
    # 4. Click rate by position in banner
    displayed_products = products_df[products_df['was_displayed']]
    plt.figure(figsize=(10, 6))
    click_rate_by_position = displayed_products.groupby('position')['was_product_clicked'].mean()
    click_rate_by_position.plot(kind='bar')
    plt.title('Click-Through Rate by Position in Banner')
    plt.xlabel('Position in Banner')
    plt.ylabel('Click-Through Rate')
    plt.grid(axis='y')
    figures.append(plt.gcf())
    
    # 5. Banner type analysis (using features 1, 2, 3, 5)
    if all(col in examples_df.columns for col in [1, 2, 3, 5]):
        examples_df['banner_type'] = examples_df.apply(
            lambda row: f"{int(row[1])}x{int(row[2])}-{int(row[3])}-{int(row[5])}", 
            axis=1
        )
        
        plt.figure(figsize=(12, 6))
        top_types = examples_df['banner_type'].value_counts().head(10)
        top_types.plot(kind='bar')
        plt.title('Top 10 Banner Types')
        plt.xlabel('Banner Type (dimensions-type-subtype)')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        figures.append(plt.gcf())
        
        # Click rate by banner type
        plt.figure(figsize=(12, 6))
        click_rate_by_type = examples_df.groupby('banner_type')['was_ad_clicked'].mean()
        click_rate_by_type.sort_values(ascending=False).head(10).plot(kind='bar')
        plt.title('Click-Through Rate by Top 10 Banner Types')
        plt.xlabel('Banner Type')
        plt.ylabel('Click-Through Rate')
        plt.grid(axis='y')
        plt.xticks(rotation=45)
        figures.append(plt.gcf())
    
    # 6. Relationship between propensity and clicks
    plt.figure(figsize=(10, 6))
    examples_df['propensity_bin'] = pd.qcut(examples_df['propensity'], 10, duplicates='drop')
    click_rate_by_propensity = examples_df.groupby('propensity_bin')['was_ad_clicked'].mean()
    click_rate_by_propensity.plot(kind='bar')
    plt.title('Click-Through Rate by Propensity Bin')
    plt.xlabel('Propensity Bin')
    plt.ylabel('Click-Through Rate')
    plt.grid(axis='y')
    plt.xticks(rotation=90)
    figures.append(plt.gcf())
    
    # 7. Feature importance for clicks (using logistic regression)
    if len(products_df) > 0:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            
            # Select product feature columns
            product_feature_cols = [col for col in products_df.columns if col.startswith('feat_')]
            
            if product_feature_cols:
                displayed_products = products_df[products_df['was_displayed']].copy()
                
                # Handle missing values
                for col in product_feature_cols:
                    if displayed_products[col].dtype != object:
                        displayed_products[col] = displayed_products[col].fillna(0)
                
                # Select only numeric columns for this analysis
                numeric_cols = [col for col in product_feature_cols 
                               if displayed_products[col].dtype != object 
                               and not isinstance(displayed_products[col].iloc[0], list)]
                
                if numeric_cols:
                    X = displayed_products[numeric_cols]
                    y = displayed_products['was_product_clicked']
                    
                    # Standardize features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Fit logistic regression
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_scaled, y)
                    
                    # Plot feature importance
                    plt.figure(figsize=(12, 8))
                    feature_importance = pd.DataFrame({
                        'Feature': numeric_cols,
                        'Importance': np.abs(model.coef_[0])
                    })
                    feature_importance = feature_importance.sort_values('Importance', ascending=False)
                    
                    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
                    plt.title('Feature Importance for Click Prediction')
                    plt.tight_layout()
                    figures.append(plt.gcf())
        except Exception as e:
            print(f"Couldn't create feature importance plot: {e}")
    
    return figures

# Main execution
def main(filepath, max_examples=5000):
    # Load data
    examples, products = load_data(filepath, max_examples)
    
    # Analyze data
    examples_df, products_df = analyze_data(examples, products)
    
    # Create visualizations
    figures = create_visualizations(examples_df, products_df)
    
    # Display summary statistics
    print("\nSummary Statistics for Examples:")
    print(f"Total examples: {len(examples_df)}")
    print(f"Click-through rate: {examples_df['was_ad_clicked'].mean():.4f}")
    print(f"Average propensity: {examples_df['propensity'].mean():.6f}")
    print(f"Median propensity: {examples_df['propensity'].median():.6f}")
    
    banner_size_distribution = examples_df['nb_slots'].value_counts().sort_index()
    print("\nBanner Size Distribution:")
    for size, count in banner_size_distribution.items():
        print(f"{size} slots: {count} examples ({count/len(examples_df)*100:.1f}%)")
    
    displayed_products = products_df[products_df['was_displayed']]
    print("\nProduct Click Statistics:")
    print(f"Total displayed products: {len(displayed_products)}")
    print(f"Product click-through rate: {displayed_products['was_product_clicked'].mean():.4f}")
    
    # Show position click rates
    position_ctr = displayed_products.groupby('position')['was_product_clicked'].mean()
    print("\nClick-through Rate by Position:")
    for position, ctr in position_ctr.items():
        print(f"Position {position}: {ctr:.4f}")
    
    # Save figures
    for i, fig in enumerate(figures):
        fig.savefig(f"eda_figure_{i+1}.png", bbox_inches='tight')
        plt.close(fig)
    
    print(f"\nSaved {len(figures)} visualizations as PNG files")
    
    return examples_df, products_df

# Uncomment and run with your file path
examples_df, products_df = main("D:/SEMESTER6/RL/rl_ad/sample.txt")