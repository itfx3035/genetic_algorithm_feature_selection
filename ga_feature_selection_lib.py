"""
Genetic Algorithm for Feature Selection Library

A Python library that uses genetic algorithms to select optimal subsets of features.
The evaluation function should return an error/metric where LOWER is better (e.g., 1 - accuracy).
"""

import random
from typing import List, Callable, Tuple, Set


class GeneticFeatureSelector:
    """
    Genetic Algorithm-based feature selector.
    
    Features are represented as genes in a chromosome (binary vector).
    Uses fitness-based selection with crossover and mutation operators.
    """
    
    def __init__(
        self,
        features: List[str],
        evaluation_function: Callable[[List[str]], float],
        population_size: int = 50,
        mutation_rate: float = 0.01,
        elite_size: int = 5,
        tournament_size: int = 3,
        max_generations: int = 100
    ):
        """
        Initialize the genetic feature selector.
        
        Args:
            features: List of feature names to consider for selection
            evaluation_function: Function that takes a list of selected features
                                and returns a metric (lower is better)
            population_size: Number of individuals in each generation
            mutation_rate: Probability of mutation per gene (0-1)
            elite_size: Number of best individuals preserved each generation
            tournament_size: Tournament selection pool size
            max_generations: Maximum number of generations to run
        """
        self.features = features
        self.evaluation_function = evaluation_function
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.max_generations = max_generations
        
        # Ensure parameters are valid
        if not 0 <= mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        if not population_size > 0:
            raise ValueError("Population size must be positive")
        if not elite_size >= 0:
            raise ValueError("Elite size must be non-negative")
        if not tournament_size >= 2:
            raise ValueError("Tournament size must be at least 2")
        
    def _create_individual(self) -> List[int]:
        """Create a random individual (chromosome)."""
        return [random.randint(0, 1) for _ in self.features]
    
    def _create_population(self) -> List[List[int]]:
        """Create initial population."""
        return [self._create_individual() for _ in range(self.population_size)]
    
    def _evaluate_fitness(self, individual: List[int]) -> float:
        """
        Evaluate fitness of an individual.
        
        Fitness is inverse of evaluation function result (since lower is better).
        This converts the problem to a standard maximization fitness landscape.
        """
        selected_features = [self.features[i] for i in range(len(self.features)) 
                           if individual[i] == 1]
        
        # Handle edge case of empty feature set
        if not selected_features:
            return float('inf')
        
        result = self.evaluation_function(selected_features)
        if result < 0 or result is None:
            return float('inf')
        
        # Fitness is inverse (higher fitness = better)
        # Adding 1 avoids log(0), scaling keeps values reasonable
        return 1.0 / (1.0 + result)
    
    def _fitness_to_probability(self, fitness: float) -> float:
        """Convert fitness to selection probability (roulette wheel)."""
        total_fitness = sum(fitness for _ in range(self.population_size))
        if total_fitness <= 0:
            return 1.0 / self.population_size
        return fitness / total_fitness
    
    def _select_parent(self, population: List[List[int]]) -> List[int]:
        """Select a parent using tournament selection."""
        candidates = random.sample(
            population, 
            min(self.tournament_size, len(population))
        )
        best_candidate = max(candidates, key=lambda ind: self._evaluate_fitness(ind))
        return best_candidate
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Perform single-point crossover."""
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have equal length")
        
        # Uniform crossover for simplicity
        child1 = [
            parent1[i] if random.random() > 0.5 else parent2[i] 
            for i in range(len(parent1))
        ]
        child2 = [
            parent2[i] if random.random() > 0.5 else parent1[i] 
            for i in range(len(parent2))
        ]
        
        return child1, child2
    
    def _mutate(self, individual: List[int]) -> List[int]:
        """Apply mutation to an individual."""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]  # Flip the gene
        return mutated
    
    def _get_elite(self, population: List[List[int]]) -> List[Tuple[List[int], float]]:
        """Get elite individuals (top performers)."""
        fitness_scores = [
            (self._evaluate_fitness(ind), ind) 
            for ind in population
        ]
        # Sort by fitness (descending - higher is better)
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        return fitness_scores[:self.elite_size]
    
    def _evolve(self, population: List[List[int]]) -> List[List[int]]:
        """Evolve population to next generation."""
        new_population = []
        
        # Preserve elite individuals
        elites = self._get_elite(population)
        for _, ind in elites:
            # Apply small mutation chance to elites too
            if random.random() < self.mutation_rate * 0.5:
                ind = self._mutate(ind)
            new_population.append(ind)
        
        # Generate rest through crossover and selection
        while len(new_population) < self.population_size:
            # Select two parents
            parent1 = self._select_parent(population)
            parent2 = self._select_parent(population)
            
            # Crossover
            child1, child2 = self._crossover(parent1, parent2)
            
            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        return new_population
    
    def run(self, verbose: bool = True) -> Tuple[List[str], float]:
        """
        Run the genetic algorithm for feature selection.
        
        Args:
            verbose: If True, print progress information
        
        Returns:
            Tuple of (best_selected_features, best_evaluation_result)
        """
        population = self._create_population()
        
        if verbose:
            print(f"Starting GA Feature Selection...")
            print(f"Features: {len(self.features)}")
            print(f"Population size: {self.population_size}")
            print(f"Elite size: {self.elite_size}")
            print(f"Mutation rate: {self.mutation_rate}")
            print(f"Max generations: {self.max_generations}")
            print("-" * 50)
        
        best_individual = None
        best_fitness = 0.0
        best_evaluation_result = float('inf')
        
        for generation in range(self.max_generations):
            # Evaluate current population
            population_fitness = [self._evaluate_fitness(ind) for ind in population]
            
            # Track best individual found so far
            for i, (fitness, ind) in enumerate(zip(population_fitness, population)):
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = ind.copy()
                    # Convert to evaluation result
                    selected = [self.features[j] for j in range(len(self.features)) 
                              if ind[j] == 1]
                    if selected:
                        eval_result = self.evaluation_function(selected)
                        if eval_result is not None and eval_result >= 0:
                            best_evaluation_result = eval_result
            
            # Print progress
            if verbose and (generation % 10 == 0 or generation == self.max_generations - 1):
                avg_fitness = sum(population_fitness) / len(population_fitness)
                best_selection = [self.features[j] for j in range(len(best_individual)) 
                                if best_individual[j] == 1]
                print(f"Generation {generation}: "
                      f"Avg fitness: {avg_fitness:.4f}, "
                      f"Best fitness: {best_fitness:.4f}, "
                      f"Best features ({len(best_selection)}): {best_selection}")
            
            # Evolve to next generation
            population = self._evolve(population)
        
        # Return best features found
        selected_features = [self.features[j] for j in range(len(best_individual)) 
                           if best_individual[j] == 1]
        
        if verbose:
            print("-" * 50)
            print(f"Best solution after {self.max_generations} generations:")
            print(f"Selected features ({len(selected_features)}): {selected_features}")
            print(f"Evaluation result (error): {best_evaluation_result:.6f}")
        
        return selected_features, best_evaluation_result


def create_binary_classification_error(y_pred: List[float], y_true: List[float]) -> float:
    """
    Create an evaluation function for binary classification.
    
    Returns 1 - accuracy as the error metric (lower is better).
    
    Args:
        y_pred: Predicted labels
        y_true: True labels
    
    Returns:
        Error metric (0 = perfect, higher = worse)
    """
    if not y_pred or not y_true or len(y_pred) != len(y_true):
        return 1.0
    
    correct = sum(1 for p, t in zip(y_pred, y_true) if p == t)
    accuracy = correct / len(y_pred)
    return 1.0 - accuracy


def create_regression_error(y_pred: List[float], y_true: List[float]) -> float:
    """
    Create an evaluation function for regression.
    
    Returns normalized mean squared error (lower is better).
    
    Args:
        y_pred: Predicted values
        y_true: True values
    
    Returns:
        Normalized MSE
    """
    if not y_pred or not y_true or len(y_pred) != len(y_true):
        return float('inf')
    
    mse = sum((p - t) ** 2 for p, t in zip(y_pred, y_true)) / len(y_pred)
    # Normalize by variance of true values (or use a default scale)
    if len(y_true) >= 2:
        mean_y = sum(y_true) / len(y_true)
        variance = sum((y - mean_y) ** 2 for y in y_true) / len(y_true)
        normalized_mse = mse / (variance + 1e-6)
    else:
        normalized_mse = mse
    
    return normalized_mse


def create_custom_error(
    selected_features: List[str], 
    model_predictor: Callable[[List[str]], Tuple[List[float], List[float]]]
) -> float:
    """
    Create a flexible evaluation function for custom models.
    
    The model_predictor should take selected features and return (y_pred, y_true).
    This is useful when you want to test different models or metrics.
    
    Args:
        selected_features: Features to use (not actually used but kept for interface consistency)
        model_predictor: Function that returns (predictions, true_values)
    
    Returns:
        Error metric - can be customized
    """
    y_pred, y_true = model_predictor(selected_features)
    
    # Default to binary accuracy error
    if len(y_pred) > 0 and all(isinstance(y, (int, float)) for y in y_pred):
        correct = sum(1 for p, t in zip(y_pred, y_true) if abs(p - t) < 0.5)
        error = 1.0 - (correct / len(y_pred))
        return max(0.0, min(1.0, error))  # Clip to [0, 1]
    
    return float('inf')


# Convenience function for quick usage
def select_features_with_ga(
    features: List[str],
    evaluation_function: Callable[[List[str]], float],
    population_size: int = 50,
    mutation_rate: float = 0.01,
    max_generations: int = 100,
    verbose: bool = True
) -> Tuple[List[str], float]:
    """
    Quick wrapper to run GA feature selection.
    
    Args:
        features: List of feature names
        evaluation_function: Function taking feature list and returning error (lower is better)
        population_size: Size of population
        mutation_rate: Mutation probability
        max_generations: Maximum generations
        verbose: Print progress
    
    Returns:
        Tuple of (selected_features, evaluation_result)
    """
    selector = GeneticFeatureSelector(
        features=features,
        evaluation_function=evaluation_function,
        population_size=population_size,
        mutation_rate=mutation_rate,
        max_generations=max_generations
    )
    return selector.run(verbose=verbose)


if __name__ == "__main__":
    # Example usage demonstration
    print("=" * 60)
    print("GA Feature Selection Library Demo")
    print("=" * 60)
    
    # Create dummy data for demonstration
    def dummy_evaluation_function(features):
        """Dummy evaluation that returns random error based on number of features."""
        num_features = len(features)
        # Simulate: too few or too many features is bad, medium is best
        ideal_range = (5, 12)
        if num_features < ideal_range[0]:
            return max(0.3, 0.1 * (ideal_range[0] - num_features))
        elif num_features > ideal_range[1]:
            return max(0.3, 0.1 * (num_features - ideal_range[1]))
        else:
            return max(0.05, 0.02 * abs(num_features - 8))
    
    features = [f"feature_{i}" for i in range(20)]
    
    print(f"\nTotal available features: {len(features)}")
    print("Running GA feature selection...")
    
    selected, error = select_features_with_ga(
        features=features,
        evaluation_function=dummy_evaluation_function,
        population_size=30,
        mutation_rate=0.05,
        max_generations=20,
        verbose=True
    )
    
    print(f"\nFinal selection:")
    print(f"  Selected: {selected}")
    print(f"  Number of features: {len(selected)}")
    print(f"  Error metric: {error:.4f}")
