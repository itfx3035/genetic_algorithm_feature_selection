"""
Example usage of GA Feature Selection Library with scikit-learn RandomForest.

This script demonstrates how to use the genetic algorithm for feature selection
with a RandomForest classifier from sklearn on real datasets.
"""

import csv
from typing import List, Tuple, Callable
from ga_feature_selection_lib import GeneticFeatureSelector, select_features_with_ga


def load_wine_quality_dataset(filepath: str = "wine-quality-red.csv") -> Tuple[List[List[float]], List[int], List[str]]:
    """
    Load Wine Quality dataset from UCI repository.
    
    Args:
        filepath: Path to CSV file (semicolon-separated)
        
    Returns:
        Tuple of (X as list of lists, y as list of labels, feature_names)
    """
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)
        
        X = []
        y = []
        feature_names = [
            "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide",
            "density", "pH", "sulphates", "alcohol"
        ]
        
        for row in reader:
            X.append([float(val) for val in row[:-1]])
            y.append(int(row[-1]))
            
    return X, y, feature_names


def create_rf_evaluation_function(
    X: List[List[float]], 
    y: List[int],
    feature_names: List[str], 
    n_estimators: int = 100, 
    random_state: int = 42
) -> Callable[[List[str]], float]:
    """Create evaluation function using RandomForest with cross-validation."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("scikit-learn is required.")
    
    feature_indices = {f: i for i, f in enumerate(feature_names)}
    
    def evaluation_func(selected_features: List[str]) -> float:
        if not selected_features:
            return 1.0
        
        feature_indices_selected = [feature_indices[f] for f in selected_features]
        
        X_subset = []
        y_list = []
        for row_idx, row in enumerate(X):
            sel_row = [row[idx] for idx in feature_indices_selected]
            X_subset.append(sel_row)
            y_list.append(y[row_idx])
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_subset, y_list, test_size=0.2, random_state=random_state
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        rf_clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            random_state=random_state,
            n_jobs=-1
        )
        
        scores = cross_val_score(rf_clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
        return 1.0 - scores.mean()
    
    return evaluation_func


def main():
    """Main function demonstrating GA feature selection with RandomForest."""
    
    print("=" * 60)
    print("GA Feature Selection with RandomForest Demo")
    print("=" * 60)
    
    # Step 1: Load Wine Quality dataset
    print("\n[1] Loading Wine Quality dataset...")
    X, y, feature_names = load_wine_quality_dataset("wine-quality-red.csv")
    print(f"   Dataset shape: {len(X)} samples x {len(feature_names)} features")
    
    # Display target distribution
    from collections import Counter
    class_dist = Counter(y)
    for cls in sorted(class_dist.keys()):
        print(f"   Quality {cls}: {class_dist[cls]} samples ({class_dist[cls]/len(y)*100:.1f}%)")
    
    # Step 2: Define evaluation function
    print("\n[2] Setting up RandomForest evaluation...")
    eval_func = create_rf_evaluation_function(
        X=X, y=y, feature_names=feature_names, n_estimators=100, random_state=42
    )
    
    # Test evaluation function
    test_result = eval_func(["alcohol"])
    print(f"   Evaluation function returned error for 1 feature: {test_result:.4f}")
    
    # Step 3: Run GA feature selection
    print("\n[3] Running Genetic Algorithm...")
    print("-" * 50)
    
    selected_features, error = select_features_with_ga(
        features=feature_names,
        evaluation_function=eval_func,
        population_size=50,
        mutation_rate=0.02,
        max_generations=40,
        verbose=True
    )
    
    # Step 4: Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nOriginal feature set size: {len(feature_names)}")
    print(f"Selected feature count: {len(selected_features)}")
    print(f"Error metric (1 - accuracy): {error:.4f}")
    print(f"\nSelected features (in rank order):")
    for i, feat in enumerate(selected_features, 1):
        print(f"   {i}. {feat}")
    
    # Step 5: Baseline comparison (all features)
    print("\n[4] Running baseline with ALL features...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    
    rf_baseline = RandomForestClassifier(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
    )
    X_scaled_full = StandardScaler().fit_transform(X)
    
    baseline_scores = cross_val_score(rf_baseline, X_scaled_full, y, cv=5, scoring='accuracy')
    baseline_error = 1.0 - baseline_scores.mean()
    
    print(f"\nBaseline (all {len(feature_names)} features):")
    print(f"   CV Accuracy: {baseline_scores.mean():.4f} (+/- {baseline_scores.std():.4f})")
    print(f"   Error metric: {baseline_error:.4f}")
    print(f"\nGA Selected (8 features):")
    print(f"   CV Accuracy: {1.0 - error:.4f} (+/- {(eval_func(selected_features) if len(selected_features) > 0 else 1.0):.4f})")
    print(f"   Error metric: {error:.4f}")
    
    improvement = baseline_error - error
    print(f"\nFeature Selection Improvement:")
    print(f"   Accuracy gain: {improvement:.4f} ({improvement*100:.1f} relative)")
    
    # Step 6: Display feature details for GA-selected features
    print("\n[5] Detailed analysis of selected features...")
    import pandas as pd
    
    df = pd.DataFrame(X, columns=feature_names)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    rf_full = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_full.fit(X_scaled_full, y_encoded)
    
    print("\nFeature Importance (from full model):")
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_full.feature_importances_
    }).sort_values('importance', ascending=False)
    for _, row in importance_df.iterrows():
        marker = " <-- SELECTED" if row['feature'] in selected_features else ""
        print(f"   {row['feature']}: {row['importance']:.4f}{marker}")


if __name__ == "__main__":
    main()
