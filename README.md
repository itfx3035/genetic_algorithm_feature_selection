# GA Feature Selection Library

⚠️ **EXPERIMENTAL** — This library is experimental and **not recommended for production use**. The genetic algorithm implementation and feature selection results are subject to change and have not been extensively validated for real-world production environments.

---

A Python library that uses **Genetic Algorithms (GA)** for automatic feature subset selection in machine learning tasks. The library provides a flexible, configurable framework for finding optimal feature combinations that maximize model performance.

---

## 🎯 Overview

Feature selection is crucial for building efficient and accurate machine learning models. This library implements a genetic algorithm approach to:

- **Automatically identify** the most relevant subset of features
- **Balance exploration and exploitation** through evolutionary search
- **Reduce computational cost** by eliminating irrelevant/ redundant features
- **Improve model interpretability** with fewer input variables

The genetic algorithm works by:
1. Representing feature subsets as binary chromosomes (genes = 1 if included, 0 otherwise)
2. Evaluating fitness based on a custom error metric (lower is better)
3. Using selection, crossover, and mutation operators to evolve better solutions over generations
4. Preserving elite individuals across generations to maintain progress

---

## 📦 Installation

### Basic Installation

```bash
pip install .
```

### With Dependencies (for scikit-learn examples)

```bash
pip install -e ".[dev]"
# Or install dependencies manually
pip install scikit-learn pandas numpy
```

---

## 🚀 Quick Start

### Minimal Example

```python
from ga_feature_selection_lib import select_features_with_ga

def dummy_eval(features):
    """Return error metric: lower is better."""
    n = len(features)
    return max(0.1, 0.2 * abs(n - 8)) / 2  # Simulated error

features = [f"feature_{i}" for i in range(20)]

selected, error = select_features_with_ga(
    features=features,
    evaluation_function=dummy_eval,
    population_size=30,
    mutation_rate=0.05,
    max_generations=20,
    verbose=True
)

print(f"Selected {len(selected)} features:")
print(selected)
```

### Real ML Workflow with scikit-learn

```python
from ga_feature_selection_lib import GeneticFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# 1. Prepare your data
X = ...  # Your features (numpy array or list of lists)
y = ...  # Your target labels
feature_names = [...]  # List of feature names

# 2. Create evaluation function using RandomForest with CV
def rf_eval(selected_features):
    if not selected_features:
        return 1.0
    
    X_subset = [X[:, feature_names.index(f)] for f in selected_features]
    X_subset = StandardScaler().fit_transform(X_subset)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    scores = cross_val_score(clf, X_subset, y, cv=5, scoring='accuracy')
    return 1.0 - scores.mean()  # Return error (lower is better)

# 3. Run GA feature selection
selector = GeneticFeatureSelector(
    features=feature_names,
    evaluation_function=rf_eval,
    population_size=50,
    mutation_rate=0.02,
    elite_size=5,
    tournament_size=3,
    max_generations=40,
    verbose=True
)

selected_features, best_error = selector.run()

print(f"Selected features: {selected_features}")
print(f"Error metric: {best_error:.4f}")
```

---

## 📚 Examples

See the provided example script `example_usage.py` for a complete end-to-end demonstration using the Wine Quality dataset with RandomForest classification and 5-fold cross-validation.

**What `example_usage.py` demonstrates:**
- Loading real-world data (UCI Wine Quality dataset)
- Creating evaluation functions with scikit-learn models
- Running GA feature selection with proper hyperparameters
- Baseline comparison (all features vs. selected subset)
- Feature importance analysis with the full model

---

## 🧩 API Reference

### `GeneticFeatureSelector` Class

Primary class for genetic feature selection.

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `features` | `List[str]` | - | List of feature names to consider |
| `evaluation_function` | `Callable[[List[str]], float]` | - | Function that returns error metric (lower is better) |
| `population_size` | `int` | 50 | Number of individuals in each generation |
| `mutation_rate` | `float` | 0.01 | Probability of mutation per gene (0-1) |
| `elite_size` | `int` | 5 | Number of best individuals preserved each generation |
| `tournament_size` | `int` | 3 | Tournament selection pool size |
| `max_generations` | `int` | 100 | Maximum number of generations to run |

#### Methods

| Method | Description |
|--------|-------------|
| `run(verbose: bool = True)` | Execute GA and return best feature subset |

**Returns:** Tuple of `(selected_features: List[str], evaluation_result: float)`

#### Constructor Validation

- `mutation_rate` must be between 0 and 1
- `population_size` must be positive
- `elite_size` must be non-negative
- `tournament_size` must be at least 2

### Helper Functions

#### `select_features_with_ga()`

Convenience wrapper for quick GA execution.

```python
selected, error = select_features_with_ga(
    features=feature_names,
    evaluation_function=my_eval_func,
    population_size=50,
    mutation_rate=0.01,
    max_generations=100,
    verbose=True
)
```

#### `create_binary_classification_error(y_pred, y_true)`

Creates error metric for binary classification: returns `1 - accuracy`.

```python
def custom_eval(selected_features):
    # Get predictions from your model
    y_pred = my_model.predict(X_subset)
    return create_binary_classification_error(y_pred, y_true)
```

#### `create_regression_error(y_pred, y_true)`

Creates normalized MSE metric for regression tasks.

#### `create_custom_error(selected_features, model_predictor)`

Flexible evaluation function creator for custom metrics.

---

## ⚙️ Configuration Guide

### Hyperparameters Explained

| Hyperparameter | Effect | Recommended Range |
|----------------|--------|-------------------|
| `population_size` | Larger → more diverse search, slower | 20-100 (50 is typical) |
| `mutation_rate` | Higher → more exploration, less convergence | 0.001-0.1 (0.01 default) |
| `elite_size` | Preserves top performers; too high reduces diversity | 3-10 (usually 3-5%) |
| `tournament_size` | Larger → stronger selection pressure | 2-5 (3 is typical) |
| `max_generations` | More generations = more computation | 20-200 (depends on problem complexity) |

### Tuning Recommendations

1. **Start with defaults** and monitor progress curves
2. **Increase population size** if search gets stuck in local optima
3. **Adjust mutation rate** based on convergence speed:
   - High error improvement per generation → increase mutation
   - Very smooth/faster convergence → decrease mutation
4. **Elite percentage** should typically be 5-10% of population

---

## 🔄 How It Works: Algorithm Flow

```
┌─────────────────────────────┐
│      Initial Population     │
│  (Random binary vectors     │
│   of feature selection)     │
└───────────┬────────────────┘
            │
            ▼
┌─────────────────────────────┐
│    Fitness Evaluation       │
│  (Evaluate each individual's│
│   selected features via     │
│   custom error metric)      │
│   → Higher fitness = better │
└───────────┬────────────────┘
            │
            ▼
┌─────────────────────────────┐
│  Selection & Reproduction   │
│  ── Tournament Selection    │
│  ── Single-point/Uniform    │
│     Crossover               │
│  ── Mutation (gene flip)    │
└───────────┬────────────────┘
            │
            ▼
┌─────────────────────────────┐
│    Elite Preservation       │
│   Top performers auto-carry │
│   to next generation        │
└───────────┬────────────────┘
            │
            ▼
┌─────────────────────────────┐
│     Check Termination       │
│   (Generation limit?)       │
└───────────┬────────────────┘
            │
      ┌─────┴─────┐
      │ Yes       │ No
      ▼           ▼
 Return Best    Loop Back to
 Solution       Selection
```

---

## 📋 Use Cases

| Scenario | Why GA Feature Selection? |
|----------|---------------------------|
| **High-dimensional data** (100+ features) | Finds optimal subset, reducing overfitting risk |
| **Multi-collinear features** | Selects representative features from correlated groups |
| **Feature interaction discovery** | Can find non-obvious feature combinations |
| **Limited labeled data** | Smaller models train faster and generalize better |
| **Interpretability requirements** | Fewer features = easier to explain decisions |

---

## 🔬 Limitations & Considerations

- **Computationally expensive**: Each evaluation runs your model; use efficient CV strategies
- **Random results**: Genetic algorithms are stochastic; run multiple times for robustness
- **Evaluation function quality matters**: Poor metric leads to poor feature subsets
- **Local optima**: May get stuck; consider running with different seeds or parameters
- **Feature ordering**: Does not guarantee globally optimal solution (NP-hard problem)

---

## 📝 Testing Your Evaluation Function

A good evaluation function should:
1. Accept a list of feature names as input
2. Return a numeric error metric (lower is better)
3. Handle edge cases (empty feature set, insufficient features)
4. Be computationally efficient during the search process

```python
# ✅ Good examples

def eval_with_cv(selected_features):
    # Build model with subset
    # Perform cross-validation
    # Return mean error (0.0 = perfect, 1.0 = worst)
    return error_value

def eval_simple(selected_features):
    n = len(selected_features)
    # Return some meaningful error metric
    return n / 10.0

# ❌ Common mistakes

def eval_bad_1(selected_features):
    # Raises instead of returning large value
    raise ValueError("Not enough features")  

def eval_bad_2(selected_features):
    # Higher is better (violates the convention)
    return accuracy_value  # Should be 1 - accuracy
```

---

## 📚 Related Resources

- Check `ga_feature_selection_lib.py` for source code implementation details
- Compare with [Filter methods](https://scikit-learn.org/stable/modules/feature_selection.html#univariate-selection-methods) or [Embedded methods](https://scikit-learn.org/stable/modules/feature_selection.html#embedded-methods)

---

## 📄 License

MIT License - feel free to use and modify for your projects.
