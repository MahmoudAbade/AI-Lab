import random
import string
import time
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional, Union, Callable
import copy
import warnings
warnings.filterwarnings("ignore")

# Configurable parameters
GA_POPSIZE = 2048           # Population size
GA_MAXITER = 16384          # Maximum iterations
GA_ELITRATE = 0.10          # Elitism rate
GA_MUTATIONRATE = 0.25      # Mutation rate
GA_TARGET = "Hello world!"  # Target string to evolve
GA_TOURNAMENT_SIZE = 5      # Size for tournament selection

# Enums for selection types
class CrossoverType:
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"

class SelectionMethod:
    ROULETTE_WHEEL = "roulette_wheel"
    STOCHASTIC_UNIVERSAL = "stochastic_universal"
    TOURNAMENT_DETERMINISTIC = "tournament_deterministic"
    TOURNAMENT_NONDETERMINISTIC = "tournament_nondeterministic"
    AGING = "aging"

class GAConfig:
    """Configuration for the genetic algorithm"""
    def __init__(self):
        self.use_crossover = True
        self.use_mutation = True
        self.crossover_type = CrossoverType.SINGLE_POINT
        self.selection_method = SelectionMethod.ROULETTE_WHEEL
        self.use_original_fitness = True  # If False, use LCS-based fitness
        self.aging_limit = 10             # For aging selection method
        self.tournament_size = GA_TOURNAMENT_SIZE
        
        # Added parameters for LCS fitness optimization
        self.lcs_weight = 10              # Weight for LCS length in fitness
        self.position_bonus_weight = 2    # Weight for position bonuses
        
    def __str__(self):
        return (
            f"Crossover: {self.use_crossover}\n"
            f"Mutation: {self.use_mutation}\n"
            f"Crossover Type: {self.crossover_type}\n"
            f"Selection Method: {self.selection_method}\n"
            f"Fitness Function: {'Original' if self.use_original_fitness else 'LCS-based'}\n"
            f"Aging Limit: {self.aging_limit}\n"
            f"Tournament Size: {self.tournament_size}\n"
            f"LCS Weight: {self.lcs_weight}\n"
            f"Position Bonus Weight: {self.position_bonus_weight}"
        )

class Individual:
    """Represents an individual in the genetic algorithm"""
    def __init__(self, chromosome=None):
        self.chromosome = chromosome if chromosome else ""
        self.fitness = 0
        self.age = 0
        self.selection_prob = 0.0
        self.inverted_fitness = 0.0  # Added for clarity
        
    def __str__(self):
        return f"Individual('{self.chromosome}', fitness={self.fitness})"
    
    def __repr__(self):
        return self.__str__()

class RuntimeMetrics:
    """Stores metrics for a single generation"""
    def __init__(self):
        self.best_fitness = 0
        self.worst_fitness = 0
        self.mean_fitness = 0
        self.std_dev_fitness = 0
        self.fitness_range = 0
        self.elapsed_time = 0
        self.is_converged = False
        self.selection_pressure = 0
        self.top_avg_selection_ratio = 0
        self.genetic_diversity = 0
        self.unique_alleles = 0
        self.shannon_entropy = 0
        # Added for fitness distribution visualization
        self.all_fitness_values = []  
    
    def to_dict(self):
        """Convert metrics to dictionary for pandas DataFrame"""
        return {
            "best_fitness": self.best_fitness,
            "worst_fitness": self.worst_fitness,
            "mean_fitness": self.mean_fitness,
            "std_dev_fitness": self.std_dev_fitness,
            "fitness_range": self.fitness_range,
            "elapsed_time": self.elapsed_time,
            "is_converged": self.is_converged,
            "selection_pressure": self.selection_pressure,
            "top_avg_selection_ratio": self.top_avg_selection_ratio,
            "genetic_diversity": self.genetic_diversity,
            "unique_alleles": self.unique_alleles,
            "shannon_entropy": self.shannon_entropy
        }


# LCS utility functions
def longest_common_subsequence(a: str, b: str) -> int:
    """Find the length of longest common subsequence between strings a and b"""
    m, n = len(a), len(b)
    # Create a table to store the LCS for subproblems
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Build the table in bottom-up fashion
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Return the length of LCS
    return dp[m][n]

def calculate_position_bonus(s: str, target: str) -> int:
    """Calculate a bonus for characters in correct positions"""
    bonus = 0
    for i in range(min(len(s), len(target))):
        if s[i] == target[i]:
            bonus += 1  # Base bonus for correct position
    return bonus

class GeneticAlgorithm:
    """Main class for the genetic algorithm implementation"""
    
    def __init__(self, target=GA_TARGET, pop_size=GA_POPSIZE, 
                 elite_rate=GA_ELITRATE, mutation_rate=GA_MUTATIONRATE,
                 config=None):
        self.target = target
        self.pop_size = pop_size
        self.elite_rate = elite_rate
        self.mutation_rate = mutation_rate
        self.config = config if config else GAConfig()
        self.population = []
        self.generation = 0
        self.metrics_history = []
        self.start_time = time.time()
        
    def initialize_population(self):
        """Create initial random population"""
        self.population = []
        for _ in range(self.pop_size):
            ind = Individual()
            # Create random string of same length as target
            ind.chromosome = ''.join(random.choice(string.printable[:-5]) 
                                     for _ in range(len(self.target)))
            self.population.append(ind)
        
    def calc_fitness_original(self):
        """Calculate fitness based on character distance (lower is better)"""
        for ind in self.population:
            fitness = 0
            for i in range(len(self.target)):
                fitness += abs(ord(ind.chromosome[i]) - ord(self.target[i]))
            ind.fitness = fitness
    
    def calc_fitness_lcs(self):
        """Calculate fitness using LCS and position bonuses"""
        for ind in self.population:
            # Get the LCS length
            lcs_length = longest_common_subsequence(ind.chromosome, self.target)
            
            # Get position bonuses
            position_bonus = calculate_position_bonus(ind.chromosome, self.target)
            
            # Calculate distance-based fitness (lower is better)
            distance_fitness = 0
            for i in range(len(self.target)):
                distance_fitness += abs(ord(ind.chromosome[i]) - ord(self.target[i]))
            
            # Combine metrics (we want to minimize fitness, so we subtract bonuses)
            # Use configurable weights for LCS and position bonuses
            ind.fitness = distance_fitness - (lcs_length * self.config.lcs_weight) - (position_bonus * self.config.position_bonus_weight)
            
            # Ensure fitness doesn't go negative
            if ind.fitness < 0:
                ind.fitness = 0
    
    def calculate_fitness(self):
        """Calculate fitness for the entire population"""
        if self.config.use_original_fitness:
            self.calc_fitness_original()
        else:
            self.calc_fitness_lcs()
    
    def sort_population(self):
        """Sort population by fitness (lower is better)"""
        self.population.sort(key=lambda x: x.fitness)
    
    def calculate_selection_probability(self):
        """Calculate selection probability for each individual for roulette wheel selection"""
        # Use fitness inversion because lower fitness is better
        max_fitness = max(ind.fitness for ind in self.population)
        total_inverted_fitness = 0
        
        for ind in self.population:
            # Add 1 to avoid zero probability for best individual
            ind.inverted_fitness = max_fitness - ind.fitness + 1
            total_inverted_fitness += ind.inverted_fitness
        
        for ind in self.population:
            ind.selection_prob = ind.inverted_fitness / total_inverted_fitness
    
    def roulette_wheel_selection(self) -> Individual:
        """Select individual based on fitness-proportionate selection"""
        r = random.random()
        sum_prob = 0
        
        for ind in self.population:
            sum_prob += ind.selection_prob
            if r <= sum_prob:
                return ind
        
        # Fallback to first individual (shouldn't happen)
        return self.population[0]
    
    def stochastic_universal_sampling(self, num_selections: int) -> List[Individual]:
        """Stochastic Universal Sampling - selects multiple individuals with a single random value"""
        selected = []
        distance = 1.0 / num_selections
        start = random.random() * distance
        
        for i in range(num_selections):
            pointer = start + i * distance
            sum_prob = 0
            
            for ind in self.population:
                sum_prob += ind.selection_prob
                if pointer <= sum_prob:
                    selected.append(ind)
                    break
        
        return selected
    
    def tournament_selection_deterministic(self) -> Individual:
        """Deterministic tournament selection - best individual always wins"""
        tournament = random.sample(self.population, self.config.tournament_size)
        return min(tournament, key=lambda x: x.fitness)
    
    def tournament_selection_nondeterministic(self) -> Individual:
        """Non-deterministic tournament selection - best individual usually wins but not always"""
        tournament = random.sample(self.population, self.config.tournament_size)
        best = min(tournament, key=lambda x: x.fitness)
        
        # With 10% probability, select a random individual instead
        if random.random() < 0.1:
            return random.choice(tournament)
        return best
    
    def select_parent(self) -> Individual:
        """Select a parent based on the configured selection method"""
        if self.config.selection_method == SelectionMethod.ROULETTE_WHEEL:
            return self.roulette_wheel_selection()
        elif self.config.selection_method == SelectionMethod.TOURNAMENT_DETERMINISTIC:
            return self.tournament_selection_deterministic()
        elif self.config.selection_method == SelectionMethod.TOURNAMENT_NONDETERMINISTIC:
            return self.tournament_selection_nondeterministic()
        else:
            # Default to roulette wheel
            return self.roulette_wheel_selection()
    
    def single_point_crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Perform single-point crossover between two parents"""
        child = Individual()
        point = random.randint(1, len(self.target) - 1)
        child.chromosome = parent1.chromosome[:point] + parent2.chromosome[point:]
        return child
    
    def two_point_crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Perform two-point crossover between two parents"""
        child = Individual()
        point1 = random.randint(1, len(self.target) - 2)
        point2 = random.randint(point1 + 1, len(self.target) - 1)
        child.chromosome = parent1.chromosome[:point1] + parent2.chromosome[point1:point2] + parent1.chromosome[point2:]
        return child
    
    def uniform_crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Perform uniform crossover between two parents"""
        child = Individual()
        child.chromosome = ''.join(
            parent1.chromosome[i] if random.random() < 0.5 else parent2.chromosome[i]
            for i in range(len(self.target))
        )
        return child
    
    def mutate(self, individual: Individual):
        """Mutate an individual by changing a random character"""
        if random.random() < self.mutation_rate:
            position = random.randint(0, len(self.target) - 1)
            # Generate a random printable character
            new_char = random.choice(string.printable[:-5])
            individual.chromosome = (
                individual.chromosome[:position] + 
                new_char + 
                individual.chromosome[position+1:]
            )

    def create_new_generation(self):
        """Create a new generation through selection, crossover, and mutation"""
        # Calculate elitism size
        elite_size = int(self.pop_size * self.elite_rate)
        new_population = []
        
        # Keep elite individuals
        for i in range(elite_size):
            elite = copy.deepcopy(self.population[i])
            elite.age += 1  # Increment age for elite individuals
            new_population.append(elite)
        
        # Calculate selection probabilities for selection methods that need it
        if self.config.selection_method in [SelectionMethod.ROULETTE_WHEEL, 
                                           SelectionMethod.STOCHASTIC_UNIVERSAL]:
            self.calculate_selection_probability()
        
        # Special case for SUS - get all parents at once
        sus_parents = None
        if self.config.selection_method == SelectionMethod.STOCHASTIC_UNIVERSAL:
            sus_parents = self.stochastic_universal_sampling(2 * (self.pop_size - elite_size))
        
        # Special case for aging-based selection
        age_sorted_population = None
        if self.config.selection_method == SelectionMethod.AGING:
            age_sorted_population = sorted(self.population, key=lambda x: x.age, reverse=True)
        
        # Breed the rest of the population
        for i in range(elite_size, self.pop_size):
            # Select parents
            parent1, parent2 = None, None
            
            if self.config.selection_method == SelectionMethod.AGING:
                # Select from oldest half of the population
                older_half = age_sorted_population[:len(age_sorted_population)//2]
                parent1 = random.choice(older_half)
                parent2 = random.choice(older_half)
            elif self.config.selection_method == SelectionMethod.STOCHASTIC_UNIVERSAL:
                # Use parents selected by SUS
                parent1 = sus_parents[(i - elite_size) * 2]
                parent2 = sus_parents[(i - elite_size) * 2 + 1]
            else:
                # Use configured selection method
                parent1 = self.select_parent()
                parent2 = self.select_parent()
            
            # Create child
            child = Individual()
            child.age = 0  # Reset age for new offspring
            
            # Apply crossover if enabled
            if self.config.use_crossover:
                if self.config.crossover_type == CrossoverType.SINGLE_POINT:
                    child = self.single_point_crossover(parent1, parent2)
                elif self.config.crossover_type == CrossoverType.TWO_POINT:
                    child = self.two_point_crossover(parent1, parent2)
                elif self.config.crossover_type == CrossoverType.UNIFORM:
                    child = self.uniform_crossover(parent1, parent2)
            else:
                # If no crossover, just copy one parent
                child.chromosome = parent1.chromosome
            
            # Apply mutation if enabled
            if self.config.use_mutation:
                self.mutate(child)
            
            # Check if individual should die due to age limit (for aging method)
            if (self.config.selection_method == SelectionMethod.AGING and
                child.age >= self.config.aging_limit):
                # Replace with a new random individual
                child.chromosome = ''.join(random.choice(string.printable[:-5]) 
                                          for _ in range(len(self.target)))
                child.age = 0
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
    
    def calculate_metrics(self) -> RuntimeMetrics:
        """Calculate and return metrics for the current generation"""
        metrics = RuntimeMetrics()
        
        # Fitness statistics
        fitness_values = [ind.fitness for ind in self.population]
        metrics.best_fitness = min(fitness_values)
        metrics.worst_fitness = max(fitness_values)
        metrics.mean_fitness = sum(fitness_values) / len(fitness_values)
        
        # Store all fitness values for boxplot visualization
        metrics.all_fitness_values = fitness_values.copy()
        
        # Standard deviation
        variance = sum((f - metrics.mean_fitness) ** 2 for f in fitness_values) / len(fitness_values)
        metrics.std_dev_fitness = math.sqrt(variance)
        
        # Fitness range
        metrics.fitness_range = metrics.worst_fitness - metrics.best_fitness
        
        # Runtime
        metrics.elapsed_time = time.time() - self.start_time
        
        # Convergence (if best individual is perfect)
        metrics.is_converged = (metrics.best_fitness == 0)
        
        # Selection pressure metrics
        if metrics.mean_fitness != 0:
            metrics.selection_pressure = metrics.fitness_range / metrics.mean_fitness
        
        # Top-Average Selection Ratio
        top_10_percent = sorted(fitness_values)[:int(self.pop_size * 0.1)]
        top_avg_fitness = sum(top_10_percent) / len(top_10_percent) if top_10_percent else 0
        if metrics.mean_fitness != 0:
            metrics.top_avg_selection_ratio = top_avg_fitness / metrics.mean_fitness
        
        # Genetic diversity - Hamming distance between individuals
        sample_size = min(100, self.pop_size)
        sampled_individuals = random.sample(self.population, sample_size)
        total_distance = 0
        comparisons = 0
        
        for i in range(sample_size):
            for j in range(i+1, sample_size):
                distance = sum(1 for a, b in zip(sampled_individuals[i].chromosome, 
                                               sampled_individuals[j].chromosome) if a != b)
                total_distance += distance
                comparisons += 1
        
        metrics.genetic_diversity = total_distance / comparisons if comparisons > 0 else 0
        
        # Count unique alleles (characters at each position)
        metrics.unique_alleles = 0
        for pos in range(len(self.target)):
            alleles = set(ind.chromosome[pos] for ind in self.population)
            metrics.unique_alleles += len(alleles)
        
        # Shannon entropy
        entropy = 0
        for pos in range(len(self.target)):
            allele_count = Counter(ind.chromosome[pos] for ind in self.population)
            pos_entropy = 0
            for count in allele_count.values():
                prob = count / self.pop_size
                pos_entropy -= prob * math.log2(prob) if prob > 0 else 0
            entropy += pos_entropy
        
        metrics.shannon_entropy = entropy / len(self.target)
        
        return metrics

    def run(self, max_generations=GA_MAXITER, print_interval=100):
        """Run the genetic algorithm"""
        print(f"Running GA with configuration:")
        print(self.config)
        print()
        
        self.generation = 0
        self.metrics_history = []
        self.start_time = time.time()
        
        # Initialize population
        self.initialize_population()
        
        # Loop through generations
        for gen in range(max_generations):
            # Calculate fitness
            self.calculate_fitness()
            
            # Sort by fitness
            self.sort_population()
            
            # Calculate and store metrics
            metrics = self.calculate_metrics()
            self.metrics_history.append(metrics)
            
            # Print best and metrics at intervals
            if gen % print_interval == 0 or metrics.is_converged:
                best = self.population[0]
                print(f"Generation {gen}: Best: '{best.chromosome}' (Fitness: {best.fitness})")
                if gen % (print_interval * 10) == 0:  # Print detailed metrics less frequently
                    self.print_metrics(metrics, gen)
            
            # Check if we've reached the target
            if metrics.is_converged:
                print(f"\nFound solution at generation {gen}")
                best = self.population[0]
                print(f"Best: '{best.chromosome}' (Fitness: {best.fitness})")
                self.print_metrics(metrics, gen)
                break
            
            # Create new generation
            self.create_new_generation()
        
        # Final results
        best = self.population[0]
        print(f"\nFinished run")
        print(f"Final solution: '{best.chromosome}'")
        print(f"Final fitness: {best.fitness}")
        print(f"Total generations: {self.generation}")
        print(f"Total time: {metrics.elapsed_time:.2f} seconds")
        print("----------------------------------------")
        
        # Save metrics
        return self.metrics_to_dataframe()
    
    def print_metrics(self, metrics: RuntimeMetrics, generation: int):
        """Print detailed metrics for a generation"""
        print(f"  Best Fitness: {metrics.best_fitness}")
        print(f"  Worst Fitness: {metrics.worst_fitness}")
        print(f"  Mean Fitness: {metrics.mean_fitness:.2f}")
        print(f"  Std Dev Fitness: {metrics.std_dev_fitness:.2f}")
        print(f"  Fitness Range: {metrics.fitness_range}")
        print(f"  Elapsed Time: {metrics.elapsed_time:.2f} seconds")
        print(f"  Selection Pressure: {metrics.selection_pressure:.2f}")
        print(f"  Genetic Diversity: {metrics.genetic_diversity:.2f}")
        print(f"  Unique Alleles: {metrics.unique_alleles}")
        print(f"  Shannon Entropy: {metrics.shannon_entropy:.2f}")
        print(f"  Converged: {'Yes' if metrics.is_converged else 'No'}")
        print()
    
    def metrics_to_dataframe(self) -> pd.DataFrame:
        """Convert metrics history to pandas DataFrame"""
        data = [metrics.to_dict() for metrics in self.metrics_history]
        df = pd.DataFrame(data)
        df['generation'] = range(len(df))
        return df

    def optimize_lcs_parameters(self, param_grid=None):
        """
        Optimize parameters for LCS-based fitness function
        This addresses requirement 7 to perform optimal parameterization
        """
        print("\n===== Optimizing LCS Parameters =====")
        
        if param_grid is None:
            # Default parameter grid
            param_grid = {
                'lcs_weight': [5, 10, 15, 20],
                'position_bonus_weight': [1, 2, 4, 8]
            }
        
        best_config = None
        best_generations = float('inf')
        best_time = float('inf')
        results = []
        
        # Save original config
        original_config = copy.deepcopy(self.config)
        
        # Try all parameter combinations
        for lcs_w in param_grid['lcs_weight']:
            for pos_w in param_grid['position_bonus_weight']:
                # Update config with test parameters
                self.config.use_original_fitness = False
                self.config.lcs_weight = lcs_w
                self.config.position_bonus_weight = pos_w
                
                print(f"\nTesting LCS weight={lcs_w}, Position bonus weight={pos_w}")
                
                # Run with limited generations
                self.generation = 0
                self.metrics_history = []
                self.start_time = time.time()
                
                # Initialize population
                self.initialize_population()
                
                found_solution = False
                for gen in range(500):  # Limit to 500 generations for optimization
                    # Calculate fitness
                    self.calculate_fitness()
                    
                    # Sort by fitness
                    self.sort_population()
                    
                    # Calculate metrics
                    metrics = self.calculate_metrics()
                    
                    # Check if we've reached the target
                    if metrics.is_converged:
                        elapsed_time = time.time() - self.start_time
                        print(f"  Found solution at generation {gen} (time: {elapsed_time:.2f}s)")
                        
                        # Record result
                        results.append({
                            'lcs_weight': lcs_w,
                            'position_bonus_weight': pos_w,
                            'generations': gen,
                            'time': elapsed_time
                        })
                        
                        # Update best if better
                        if gen < best_generations:
                            best_generations = gen
                            best_time = elapsed_time
                            best_config = {
                                'lcs_weight': lcs_w,
                                'position_bonus_weight': pos_w
                            }
                        
                        found_solution = True
                        break
                    
                    # Create new generation
                    self.create_new_generation()
                
                if not found_solution:
                    print("  No solution found within 500 generations")
                    # Record non-convergent result
                    results.append({
                        'lcs_weight': lcs_w,
                        'position_bonus_weight': pos_w,
                        'generations': 500,
                        'time': time.time() - self.start_time
                    })
        
        # Restore original config
        self.config = original_config
        
        # Print optimization results
        print("\nLCS Parameter Optimization Results:")
        if best_config:
            print(f"Best configuration: LCS weight={best_config['lcs_weight']}, Position bonus weight={best_config['position_bonus_weight']}")
            print(f"Generations to convergence: {best_generations}")
            print(f"Time to convergence: {best_time:.2f}s")
            
            # Update config with best parameters
            self.config.lcs_weight = best_config['lcs_weight']
            self.config.position_bonus_weight = best_config['position_bonus_weight']
        else:
            print("No configuration converged within the generation limit")
        
        # Generate visualization of results
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            plt.figure(figsize=(10, 6))
            for lcs_w in param_grid['lcs_weight']:
                subset = results_df[results_df['lcs_weight'] == lcs_w]
                if not subset.empty:
                    plt.plot(subset['position_bonus_weight'], subset['generations'], 
                             marker='o', label=f'LCS weight={lcs_w}')
            
            plt.xlabel('Position Bonus Weight')
            plt.ylabel('Generations to Convergence')
            plt.title('Effect of LCS Parameters on Convergence Speed')
            plt.legend()
            plt.grid(True)
            plt.savefig('lcs_parameter_optimization.png')
            plt.close()
        
        return results
          
# Visualization functions
def plot_fitness_evolution(metrics_df, run_label):
    """Plot fitness metrics over generations"""
    plt.figure(figsize=(12, 8))
    
    # Plot fitness evolution
    plt.subplot(2, 1, 1)
    plt.plot(metrics_df['generation'], metrics_df['best_fitness'], label='Best Fitness', color='green')
    plt.plot(metrics_df['generation'], metrics_df['mean_fitness'], label='Mean Fitness', color='blue')
    plt.plot(metrics_df['generation'], metrics_df['worst_fitness'], label='Worst Fitness', color='red')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'Fitness Evolution - {run_label}')
    plt.legend()
    plt.grid(True)
    
    # Plot diversity metrics
    plt.subplot(2, 1, 2)
    plt.plot(metrics_df['generation'], metrics_df['genetic_diversity'], label='Genetic Diversity', color='purple')
    plt.plot(metrics_df['generation'], metrics_df['shannon_entropy'], label='Shannon Entropy', color='orange')
    plt.xlabel('Generation')
    plt.ylabel('Diversity Metrics')
    plt.title('Diversity Metrics')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'fitness_evolution_{run_label}.png')
    plt.close()

def plot_fitness_boxplots(metrics_history, run_label):
    """
    Create boxplots of fitness distribution at regular intervals throughout evolution
    This addresses the boxplot visualization requirement (3b)
    """
    # Select a subset of generations for visualization
    num_gens = len(metrics_history)
    if num_gens <= 10:
        sample_indices = list(range(num_gens))
    else:
        # Take evenly spaced samples
        sample_indices = [int(i * num_gens / 10) for i in range(10)]
        if num_gens - 1 not in sample_indices:
            sample_indices.append(num_gens - 1)  # Always include the last generation
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Create data for boxplots
    boxplot_data = []
    labels = []
    
    for idx in sample_indices:
        if idx < len(metrics_history):
            boxplot_data.append(metrics_history[idx].all_fitness_values)
            labels.append(f'Gen {idx}')
    
    # Create boxplot
    plt.boxplot(boxplot_data, labels=labels, showfliers=True)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'Fitness Distribution Across Generations - {run_label}')
    plt.grid(True, axis='y')
    
    # Add explanation text
    plt.figtext(0.5, 0.01, 
                'Boxplots show the distribution of fitness values at different generations.\n'
                'The box represents the interquartile range, with the middle line showing the median.\n'
                'Whiskers show min/max values (excluding outliers), and dots represent outliers.',
                ha='center', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make room for the text
    plt.savefig(f'fitness_boxplots_{run_label}.png')
    plt.close()

def compare_crossover_operators(metrics_dfs, operators):
    """Compare different crossover operators"""
    plt.figure(figsize=(12, 6))
    
    for op, df in zip(operators, metrics_dfs):
        plt.plot(df['generation'], df['best_fitness'], label=f'Best Fitness - {op}')
    
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Comparison of Crossover Operators')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('crossover_comparison.png')
    plt.close()

def compare_selection_methods(metrics_dfs, methods):
    """Compare different selection methods"""
    plt.figure(figsize=(12, 6))
    
    for method, df in zip(methods, metrics_dfs):
        plt.plot(df['generation'], df['best_fitness'], label=f'Best Fitness - {method}')
    
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Comparison of Selection Methods')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('selection_comparison.png')
    plt.close()

def compare_mutation_crossover_configs(metrics_dfs, configs):
    """Compare different mutation/crossover configurations"""
    plt.figure(figsize=(12, 6))
    
    for config, df in zip(configs, metrics_dfs):
        plt.plot(df['generation'], df['best_fitness'], label=f'Best Fitness - {config}')
    
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Comparison of Mutation/Crossover Configurations')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('mutation_crossover_comparison.png')
    plt.close()

def compare_fitness_functions(original_df, lcs_df):
    """Compare original fitness function with LCS-based fitness function"""
    plt.figure(figsize=(12, 6))
    
    plt.plot(original_df['generation'], original_df['best_fitness'], label='Best Fitness - Original')
    plt.plot(lcs_df['generation'], lcs_df['best_fitness'], label='Best Fitness - LCS-based')
    
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Comparison of Fitness Functions')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('fitness_function_comparison.png')
    plt.close()

def plot_exploration_exploitation_balance(metrics_df, run_label):
    """
    Plot metrics related to exploration vs. exploitation balance
    This helps visualize requirement 5
    """
    plt.figure(figsize=(12, 8))
    
    # Plot selection pressure and diversity
    plt.subplot(2, 1, 1)
    plt.plot(metrics_df['generation'], metrics_df['selection_pressure'], 
            label='Selection Pressure', color='red')
    plt.plot(metrics_df['generation'], metrics_df['top_avg_selection_ratio'], 
            label='Top-Avg Selection Ratio', color='orange', linestyle='--')
    plt.xlabel('Generation')
    plt.ylabel('Selection Pressure')
    plt.title(f'Exploration vs. Exploitation Metrics - {run_label}')
    plt.legend()
    plt.grid(True)
    
    # Plot diversity metrics
    plt.subplot(2, 1, 2)
    plt.plot(metrics_df['generation'], metrics_df['genetic_diversity'], 
            label='Genetic Diversity', color='blue')
    plt.plot(metrics_df['generation'], metrics_df['shannon_entropy'], 
            label='Shannon Entropy', color='green')
    plt.plot(metrics_df['generation'], metrics_df['unique_alleles'] / (metrics_df['unique_alleles'].max() * 1.1), 
            label='Normalized Unique Alleles', color='purple', linestyle=':')
    plt.xlabel('Generation')
    plt.ylabel('Diversity Metrics')
    plt.legend()
    plt.grid(True)
    
    # Add explanation
    plt.figtext(0.5, 0.01, 
                'Higher selection pressure indicates more exploitation.\n'
                'Higher diversity metrics indicate more exploration.',
                ha='center', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f'exploration_exploitation_{run_label}.png')
    plt.close()

# ARC Challenge Implementation
class ARCProblem:
    """Implementation for the Abstraction and Reasoning Challenge problems"""
    
    def __init__(self, input_grid, target_grid, name="", description=""):
        self.input_grid = input_grid
        self.target_grid = target_grid
        self.name = name
        self.description = description
    
    def print_grid(self, grid):
        """Print a grid to console"""
        for row in grid:
            print(' '.join(str(cell) for cell in row))
    
    def visualize_grids(self, result=None):
        """Visualize input, target, and result grids with color"""
        fig, axs = plt.subplots(1, 3 if result else 2, figsize=(12, 4))
        
        # Define a colormap with distinct colors
        cmap = plt.cm.get_cmap('tab10', 10)
        
        # Plot input grid
        ax1 = axs[0]
        im1 = ax1.imshow(self.input_grid, cmap=cmap, interpolation='nearest')
        ax1.set_title('Input Grid')
        ax1.grid(True, color='white', linestyle='-', linewidth=1.5)
        ax1.set_xticks(np.arange(-.5, len(self.input_grid[0]), 1), minor=True)
        ax1.set_yticks(np.arange(-.5, len(self.input_grid), 1), minor=True)
        ax1.tick_params(which='minor', length=0)
        
        # Plot target grid
        ax2 = axs[1]
        im2 = ax2.imshow(self.target_grid, cmap=cmap, interpolation='nearest')
        ax2.set_title('Target Grid')
        ax2.grid(True, color='white', linestyle='-', linewidth=1.5)
        ax2.set_xticks(np.arange(-.5, len(self.target_grid[0]), 1), minor=True)
        ax2.set_yticks(np.arange(-.5, len(self.target_grid), 1), minor=True)
        ax2.tick_params(which='minor', length=0)
        
        # Plot result grid if provided
        if result:
            ax3 = axs[2]
            im3 = ax3.imshow(result, cmap=cmap, interpolation='nearest')
            ax3.set_title('Result Grid')
            ax3.grid(True, color='white', linestyle='-', linewidth=1.5)
            ax3.set_xticks(np.arange(-.5, len(result[0]), 1), minor=True)
            ax3.set_yticks(np.arange(-.5, len(result), 1), minor=True)
            ax3.tick_params(which='minor', length=0)
        
        # Add a title with problem description
        if self.name or self.description:
            title = f"{self.name}: {self.description}" if self.name and self.description else (self.name or self.description)
            plt.suptitle(title, fontsize=14)
            plt.subplots_adjust(top=0.85)
        
        plt.tight_layout()
        plt.savefig(f'arc_problem_{self.name.replace(" ", "_")}.png')
        plt.close()
    
    def find_color_transform(self):
        """Find a simple color transformation rule between input and target"""
        transformations = {}
        
        for i in range(len(self.input_grid)):
            for j in range(len(self.input_grid[i])):
                in_color = self.input_grid[i][j]
                target_color = self.target_grid[i][j]
                
                if in_color != target_color:
                    if in_color not in transformations:
                        transformations[in_color] = target_color
                    elif transformations[in_color] != target_color:
                        # Inconsistent mapping
                        return None
        
        return transformations if transformations else None
    
    def apply_transformation(self, transform_dict):
        """Apply a color transformation to input grid"""
        result = [row[:] for row in self.input_grid]  # Deep copy
        
        for i in range(len(result)):
            for j in range(len(result[i])):
                if result[i][j] in transform_dict:
                    result[i][j] = transform_dict[result[i][j]]
        
        return result
    
    def check_solution(self, result):
        """Check if a result grid matches the target grid"""
        if len(result) != len(self.target_grid):
            return False
        
        for i in range(len(result)):
            if len(result[i]) != len(self.target_grid[i]):
                return False
            
            for j in range(len(result[i])):
                if result[i][j] != self.target_grid[i][j]:
                    return False
        
        return True
    
    def solve_with_ga(self):
        """Solve the ARC problem using a genetic algorithm approach"""
        print(f"\nSolving ARC problem '{self.name}' with genetic algorithm")
        print(f"Description: {self.description}")
        
        # Find colors in input and target grids
        input_colors = set()
        for row in self.input_grid:
            input_colors.update(row)
        
        target_colors = set()
        for row in self.target_grid:
            target_colors.update(row)
        
        print(f"Input colors: {input_colors}")
        print(f"Target colors: {target_colors}")
        
        # Visualize the problem
        self.visualize_grids()
        
        # Try simple transformation first
        transform_dict = self.find_color_transform()
        if transform_dict:
            print(f"Found simple transformation: {transform_dict}")
            result = self.apply_transformation(transform_dict)
            if self.check_solution(result):
                print("Solution found!")
                print("Input grid:")
                self.print_grid(self.input_grid)
                print("\nResult grid:")
                self.print_grid(result)
                print("\nTarget grid:")
                self.print_grid(self.target_grid)
                
                # Visualize the solution
                self.visualize_grids(result)
                return True
        
        # If simple transformation didn't work, use GA
        # For ARC, we'll represent individuals as a mapping from input colors to target colors
        
        class ARCIndividual:
            def __init__(self, input_colors, target_colors):
                self.mapping = {}
                for color in input_colors:
                    self.mapping[color] = random.choice(list(target_colors))
                self.fitness = float('inf')
            
            def __str__(self):
                return f"ARCIndividual(mapping={self.mapping}, fitness={self.fitness})"
        
        def calculate_fitness(individual, problem):
            """Calculate how many cells differ from target after transformation"""
            result = problem.apply_transformation(individual.mapping)
            fitness = 0
            for i in range(len(problem.target_grid)):
                for j in range(len(problem.target_grid[i])):
                    if result[i][j] != problem.target_grid[i][j]:
                        fitness += 1
            return fitness
        
        # GA parameters
        pop_size = 100
        num_generations = 100
        mutation_rate = 0.2
        
        # Record metrics for visualization
        generation_history = []
        best_fitness_history = []
        avg_fitness_history = []
        
        # Initialize population
        population = [ARCIndividual(input_colors, target_colors) for _ in range(pop_size)]
        
        # Evaluate initial population
        for ind in population:
            ind.fitness = calculate_fitness(ind, self)
        
        # Main GA loop
        start_time = time.time()
        for generation in range(num_generations):
            # Sort by fitness
            population.sort(key=lambda x: x.fitness)
            
            # Record metrics
            generation_history.append(generation)
            best_fitness_history.append(population[0].fitness)
            avg_fitness = sum(ind.fitness for ind in population) / pop_size
            avg_fitness_history.append(avg_fitness)
            
            # Check if we found a solution
            if population[0].fitness == 0:
                end_time = time.time()
                print(f"Solution found at generation {generation}")
                print(f"Time to solution: {end_time - start_time:.2f} seconds")
                result = self.apply_transformation(population[0].mapping)
                print("Input grid:")
                self.print_grid(self.input_grid)
                print("\nResult grid:")
                self.print_grid(result)
                print("\nTarget grid:")
                self.print_grid(self.target_grid)
                print(f"\nTransformation: {population[0].mapping}")
                
                # Visualize the solution
                self.visualize_grids(result)
                
                # Plot fitness evolution
                plt.figure(figsize=(10, 6))
                plt.plot(generation_history, best_fitness_history, label='Best Fitness')
                plt.plot(generation_history, avg_fitness_history, label='Average Fitness')
                plt.xlabel('Generation')
                plt.ylabel('Fitness (Number of Different Cells)')
                plt.title(f'ARC Problem: {self.name} - Fitness Evolution')
                plt.legend()
                plt.grid(True)
                plt.savefig(f'arc_fitness_{self.name.replace(" ", "_")}.png')
                plt.close()
                
                return True
            
            # Progress report
            if generation % 10 == 0:
                print(f"Generation {generation}, best fitness: {population[0].fitness}")
                print(f"  Best mapping: {population[0].mapping}")
            
            # Create next generation
            next_population = []
            
            # Elitism - keep best individual
            next_population.append(copy.deepcopy(population[0]))
            
            # Create rest of population
            while len(next_population) < pop_size:
                # Tournament selection
                tournament_size = 5
                tourney = random.sample(population, tournament_size)
                parent1 = min(tourney, key=lambda x: x.fitness)
                
                tourney = random.sample(population, tournament_size)
                parent2 = min(tourney, key=lambda x: x.fitness)
                
                # Create child with crossover
                child = ARCIndividual(input_colors, target_colors)
                for color in input_colors:
                    if random.random() < 0.5:
                        child.mapping[color] = parent1.mapping[color]
                    else:
                        child.mapping[color] = parent2.mapping[color]
                
                # Mutation
                if random.random() < mutation_rate:
                    mutation_color = random.choice(list(input_colors))
                    child.mapping[mutation_color] = random.choice(list(target_colors))
                
                # Evaluate fitness
                child.fitness = calculate_fitness(child, self)
                
                next_population.append(child)
            
            population = next_population
        
        # If we reach here, we didn't find an exact solution
        end_time = time.time()
        print(f"No exact solution found in {num_generations} generations")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print("Returning best approximation")
        
        population.sort(key=lambda x: x.fitness)
        result = self.apply_transformation(population[0].mapping)
        
        print("Input grid:")
        self.print_grid(self.input_grid)
        print("\nBest result grid:")
        self.print_grid(result)
        print("\nTarget grid:")
        self.print_grid(self.target_grid)
        print(f"\nBest transformation: {population[0].mapping}")
        print(f"Fitness (cells different): {population[0].fitness}")
        
        # Visualize the best solution
        self.visualize_grids(result)
        
        # Plot fitness evolution
        plt.figure(figsize=(10, 6))
        plt.plot(generation_history, best_fitness_history, label='Best Fitness')
        plt.plot(generation_history, avg_fitness_history, label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Number of Different Cells)')
        plt.title(f'ARC Problem: {self.name} - Fitness Evolution')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'arc_fitness_{self.name.replace(" ", "_")}.png')
        plt.close()
        
        return False
    
# Bin Packing Problem Implementation
class BinPackingProblem:
    """Implementation for the Bin Packing Problem"""
    
    def __init__(self, items, bin_capacity):
        self.items = items
        self.bin_capacity = bin_capacity
    
    def first_fit(self):
        """Implement the First-Fit heuristic for bin packing"""
        bins = []  # List of bins, each bin is a list of items
        bin_remaining = []  # Remaining capacity in each bin
        
        for item in self.items:
            # Try to place item in an existing bin
            placed = False
            for i in range(len(bins)):
                if bin_remaining[i] >= item:
                    bins[i].append(item)
                    bin_remaining[i] -= item
                    placed = True
                    break
            
            # If item couldn't fit in any existing bin, create a new bin
            if not placed:
                bins.append([item])
                bin_remaining.append(self.bin_capacity - item)
        
        return bins
    
    def is_valid_solution(self, assignment):
        """Check if a bin assignment solution is valid (no bin overflow)"""
        if not assignment:
            return False
        
        num_bins = max(assignment) + 1
        bin_usage = [0] * num_bins
        
        for i, bin_idx in enumerate(assignment):
            bin_usage[bin_idx] += self.items[i]
            
            if bin_usage[bin_idx] > self.bin_capacity:
                return False  # Overflow in this bin
        
        return True
    
    def calculate_bin_packing_fitness(self, assignment):
        """Calculate fitness for a bin packing solution (number of bins used)"""
        # If the solution is invalid, return a high value
        if not assignment:
            return float('inf')
        
        if not self.is_valid_solution(assignment):
            return float('inf')
        
        # The fitness is the number of bins used (lower is better)
        return max(assignment) + 1
    
    def visualize_bins(self, bins, algorithm_name):
        """Visualize bin contents and usage without labels outside plot boundaries"""
        plt.figure(figsize=(14, 8))
        
        # Calculate bin usage
        bin_usage = [sum(bin_items) for bin_items in bins]
        usage_percentage = [usage / self.bin_capacity * 100 for usage in bin_usage]
        
        # Plot bin usage
        plt.subplot(1, 2, 1)
        plt.bar(range(len(bins)), usage_percentage)
        plt.axhline(y=100, color='r', linestyle='--', label='Capacity')
        plt.xlabel('Bin')
        plt.ylabel('Usage (%)')
        plt.title(f'Bin Usage - {algorithm_name}')
        plt.xticks(range(len(bins)))
        plt.ylim(0, 105)  # Give a little space above 100%
        plt.legend()
        plt.grid(True)
        
        # Plot item distribution
        plt.subplot(1, 2, 2)
        bottoms = [0] * len(bins)
        colors = plt.cm.get_cmap('tab20', len(self.items))
        
        # Create figure with fixed height
        for bin_idx, bin_items in enumerate(bins):
            bottom = 0
            for item_idx, item in enumerate(sorted(bin_items, reverse=True)):  # Sort items by size (largest first)
                height = item / self.bin_capacity * 100
                plt.bar(bin_idx, height, bottom=bottom, width=0.8, 
                        color=colors(item_idx % 20), alpha=0.7, edgecolor='white')
                
                # Only add text if there's enough space (minimum 10% of capacity)
                if height >= 10:
                    # Center the text in the colored section
                    text_y = bottom + height/2
                    plt.text(bin_idx, text_y, str(item), ha='center', va='center', 
                            fontsize=9, fontweight='bold')
                
                bottom += height
        
        plt.xlabel('Bin')
        plt.ylabel('Content (% of capacity)')
        plt.title(f'Bin Contents - {algorithm_name}')
        plt.xticks(range(len(bins)))
        plt.ylim(0, 105)  # Give a little space above 100%
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'bin_packing_{algorithm_name.replace(" ", "_")}.png')
        plt.close()
    
    def solve_with_ga(self):
        """Solve the bin packing problem using a genetic algorithm"""
        print("\nSolving Bin Packing Problem with genetic algorithm")
        print(f"Items: {self.items}")
        print(f"Bin capacity: {self.bin_capacity}")
        
        # First, solve using First-Fit (a greedy approach)
        ff_bins = self.first_fit()
        print(f"First-Fit solution uses {len(ff_bins)} bins")
        
        for i, bin_items in enumerate(ff_bins):
            bin_usage = sum(bin_items)
            print(f"  Bin {i} (usage: {bin_usage}/{self.bin_capacity}): {bin_items}")
        
        # Visualize First-Fit solution
        self.visualize_bins(ff_bins, "First-Fit")
        
        # Now solve using GA
        class BPIndividual:
            """Represents a bin packing solution"""
            def __init__(self, n_items, bin_capacity, items):
                # Randomly assign each item to a bin
                # Initially create a valid solution using First-Fit
                self.assignment = []
                bins_used = 0
                bin_remaining = []
                
                for item in items:
                    # Try to place in existing bin
                    placed = False
                    for i in range(bins_used):
                        if bin_remaining[i] >= item:
                            self.assignment.append(i)
                            bin_remaining[i] -= item
                            placed = True
                            break
                    
                    # If not placed, create new bin
                    if not placed:
                        self.assignment.append(bins_used)
                        bins_used += 1
                        bin_remaining.append(bin_capacity - item)
                
                # Add some randomness
                for i in range(len(self.assignment)):
                    if random.random() < 0.3:  # 30% chance to move an item
                        # Try to move to a different bin (including potentially new ones)
                        new_bin = random.randint(0, max(self.assignment) + 1)
                        self.assignment[i] = new_bin
                
                self.fitness = float('inf')
        
        # GA parameters
        pop_size = 200
        elite_size = int(pop_size * 0.1)  # Top 10% survive unchanged
        max_generations = 500
        mutation_rate = 0.2
        crossover_rate = 0.8
        tournament_size = 5
        
        # Initialize items and bin capacity from the problem
        items = self.items
        bin_capacity = self.bin_capacity
        
        # Create initial population
        population = [BPIndividual(len(items), bin_capacity, items) for _ in range(pop_size)]
        
        # Evaluate initial population
        for ind in population:
            ind.fitness = self.calculate_bin_packing_fitness(ind.assignment)
        
        # Track the best solution
        best_solution = None
        best_fitness = float('inf')
        
        # Track metrics for plotting
        generation_history = []
        best_fitness_history = []
        avg_fitness_history = []
        
        # Start timing
        start_time = time.time()
        
        # Main GA loop
        for generation in range(max_generations):
            # Sort population by fitness
            population.sort(key=lambda x: x.fitness)
            
            # Track the best solution
            if population[0].fitness < best_fitness:
                best_solution = copy.deepcopy(population[0])
                best_fitness = population[0].fitness
            
            # Print progress
            if generation % 50 == 0 or generation == max_generations - 1:
                valid_solutions = sum(1 for ind in population if ind.fitness < float('inf'))
                avg_fitness = sum(ind.fitness for ind in population if ind.fitness < float('inf')) / max(1, valid_solutions)
                print(f"Generation {generation}, best fitness: {population[0].fitness} bins used")
                print(f"  Valid solutions: {valid_solutions}/{pop_size}, avg fitness: {avg_fitness:.2f}")
            
            # Track metrics
            generation_history.append(generation)
            best_fitness_history.append(population[0].fitness)
            
            valid_solutions = [ind for ind in population if ind.fitness < float('inf')]
            avg_fitness = (sum(ind.fitness for ind in valid_solutions) / len(valid_solutions)) if valid_solutions else float('inf')
            avg_fitness_history.append(avg_fitness)
            
            # Create next generation
            next_population = []
            
            # Elitism - keep best individuals
            for i in range(elite_size):
                next_population.append(copy.deepcopy(population[i]))
            
            # Create the rest of the population
            while len(next_population) < pop_size:
                # Tournament selection for parents
                parent1 = None
                for _ in range(tournament_size):
                    candidate = random.choice(population)
                    if parent1 is None or candidate.fitness < parent1.fitness:
                        parent1 = candidate
                
                parent2 = None
                for _ in range(tournament_size):
                    candidate = random.choice(population)
                    if parent2 is None or candidate.fitness < parent2.fitness:
                        parent2 = candidate
                
                # Create child
                child = BPIndividual(len(items), bin_capacity, items)
                
                # Apply crossover with some probability
                if random.random() < crossover_rate:
                    # One-point crossover
                    crossover_point = random.randint(1, len(items) - 1)
                    
                    # Remap bin indices to maintain relative structure
                    p1_bins = {}
                    p2_bins = {}
                    next_bin1 = 0
                    next_bin2 = 0
                    
                    # Build remapping dictionaries
                    for i in range(crossover_point):
                        if parent1.assignment[i] not in p1_bins:
                            p1_bins[parent1.assignment[i]] = next_bin1
                            next_bin1 += 1
                    
                    for i in range(crossover_point, len(items)):
                        if parent2.assignment[i] not in p2_bins:
                            p2_bins[parent2.assignment[i]] = next_bin2
                            next_bin2 += 1
                    
                    # Apply crossover with remapping
                    child.assignment = []
                    for i in range(len(items)):
                        if i < crossover_point:
                            child.assignment.append(p1_bins[parent1.assignment[i]])
                        else:
                            child.assignment.append(p2_bins[parent2.assignment[i]] + next_bin1)
                else:
                    # Just copy the better parent
                    child.assignment = parent1.assignment.copy()
                
                # Apply mutation with some probability
                if random.random() < mutation_rate:
                    # Move a random item to a different bin
                    mutation_item = random.randint(0, len(items) - 1)
                    current_max_bin = max(child.assignment)
                    
                    # 70% chance to move to existing bin, 30% chance for a new bin
                    if random.random() < 0.7 and current_max_bin > 0:
                        new_bin = random.randint(0, current_max_bin)
                    else:
                        new_bin = current_max_bin + 1
                    
                    child.assignment[mutation_item] = new_bin
                
                # Calculate fitness of child
                child.fitness = self.calculate_bin_packing_fitness(child.assignment)
                
                # Add to new population
                next_population.append(child)
            
            # Replace old population
            population = next_population
        
        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Get the best solution from all generations
        population.sort(key=lambda x: x.fitness)
        best_individual = population[0] if population[0].fitness < best_fitness else best_solution
        
        print("\nGA solution:")
        print(f"Number of bins used: {best_individual.fitness}")
        print(f"Runtime: {elapsed_time:.2f} seconds")
        
        # Bin usage statistics
        bin_usage = [0] * best_individual.fitness
        bin_contents = [[] for _ in range(best_individual.fitness)]
        
        for i, bin_idx in enumerate(best_individual.assignment):
            bin_contents[bin_idx].append(self.items[i])
            bin_usage[bin_idx] += self.items[i]
        
        for i in range(best_individual.fitness):
            print(f"  Bin {i} (usage: {bin_usage[i]}/{self.bin_capacity}): {bin_contents[i]}")
        
        # Visualize GA solution
        self.visualize_bins(bin_contents, "Genetic Algorithm")
        
        # Plot fitness evolution
        plt.figure(figsize=(10, 6))
        plt.plot(generation_history, best_fitness_history, label='Best Fitness')
        plt.plot(generation_history, avg_fitness_history, label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Number of Bins)')
        plt.title('Bin Packing GA Evolution')
        plt.legend()
        plt.grid(True)
        plt.savefig('bin_packing_evolution.png')
        plt.close()
        
        # Compare First-Fit and GA results
        plt.figure(figsize=(10, 6))
        
        # Data for comparison
        algorithms = ['First-Fit', 'Genetic Algorithm']
        bins_used = [len(ff_bins), best_individual.fitness]
        
        # Calculate average bin utilization
        ff_utilization = sum(sum(bin_items) for bin_items in ff_bins) / (len(ff_bins) * self.bin_capacity) * 100
        ga_utilization = sum(bin_usage) / (best_individual.fitness * self.bin_capacity) * 100
        utilization = [ff_utilization, ga_utilization]
        
        # Plot bins used
        plt.subplot(1, 2, 1)
        plt.bar(algorithms, bins_used, color=['blue', 'green'])
        plt.ylabel('Bins Used')
        plt.title('Number of Bins')
        plt.grid(axis='y')
        
        # Plot bin utilization
        plt.subplot(1, 2, 2)
        plt.bar(algorithms, utilization, color=['blue', 'green'])
        plt.ylabel('Average Bin Utilization (%)')
        plt.title('Bin Utilization')
        plt.grid(axis='y')
        
        plt.tight_layout()
        plt.savefig('bin_packing_comparison.png')
        plt.close()
        
        # Return comparison for analysis
        return {
            'ff_bins_used': len(ff_bins),
            'ga_bins_used': best_individual.fitness,
            'ff_utilization': ff_utilization,
            'ga_utilization': ga_utilization,
            'ff_runtime': 0,  # First-Fit is fast enough to consider negligible
            'ga_runtime': elapsed_time,
            'ff_solution': ff_bins,
            'ga_solution': bin_contents
        }

# Main function to run all experiments
def main():
    # Run all the required experiments
    results = {}
    all_metrics_history = {}  # Store metrics history for boxplot visualization
    
    print("===== Testing Different Crossover Operators =====")
    
    # Test single-point crossover
    config1 = GAConfig()
    config1.use_crossover = True
    config1.use_mutation = True
    config1.crossover_type = CrossoverType.SINGLE_POINT
    config1.selection_method = SelectionMethod.ROULETTE_WHEEL
    config1.use_original_fitness = True
    config1.aging_limit = 10
    config1.tournament_size = GA_TOURNAMENT_SIZE
    
    ga1 = GeneticAlgorithm(config=config1)
    results['single_point'] = ga1.run(max_generations=1000)
    all_metrics_history['single_point'] = ga1.metrics_history
    
    # Test two-point crossover
    config2 = copy.deepcopy(config1)
    config2.crossover_type = CrossoverType.TWO_POINT
    
    ga2 = GeneticAlgorithm(config=config2)
    results['two_point'] = ga2.run(max_generations=1000)
    all_metrics_history['two_point'] = ga2.metrics_history
    
    # Test uniform crossover
    config3 = copy.deepcopy(config1)
    config3.crossover_type = CrossoverType.UNIFORM
    
    ga3 = GeneticAlgorithm(config=config3)
    results['uniform'] = ga3.run(max_generations=1000)
    all_metrics_history['uniform'] = ga3.metrics_history
    
    print("\n===== Testing Crossover vs Mutation =====")
    
    # Test with only crossover (no mutation)
    config4 = copy.deepcopy(config1)
    config4.use_mutation = False
    
    ga4 = GeneticAlgorithm(config=config4)
    results['only_crossover'] = ga4.run(max_generations=1000)
    all_metrics_history['only_crossover'] = ga4.metrics_history
    
    # Test with only mutation (no crossover)
    config5 = copy.deepcopy(config1)
    config5.use_crossover = False
    config5.use_mutation = True
    
    ga5 = GeneticAlgorithm(config=config5)
    results['only_mutation'] = ga5.run(max_generations=1000)
    all_metrics_history['only_mutation'] = ga5.metrics_history
    
    # Default has both crossover and mutation
    all_metrics_history['both'] = ga1.metrics_history  # Single point from earlier
    
    print("\n===== Optimizing LCS-based Fitness Parameters =====")
    
    # Default config but with LCS fitness
    config_lcs = copy.deepcopy(config1)
    config_lcs.use_original_fitness = False
    
    ga_lcs = GeneticAlgorithm(config=config_lcs)
    # Optimize parameters for LCS
    ga_lcs.optimize_lcs_parameters()
    
    # Run with optimized parameters
    results['lcs_fitness'] = ga_lcs.run(max_generations=1000)
    all_metrics_history['lcs_fitness'] = ga_lcs.metrics_history
    
    print("\n===== Testing Different Selection Methods =====")
    
    # Test stochastic universal sampling
    config7 = copy.deepcopy(config1)
    config7.selection_method = SelectionMethod.STOCHASTIC_UNIVERSAL
    
    ga7 = GeneticAlgorithm(config=config7)
    results['stochastic_universal'] = ga7.run(max_generations=1000)
    all_metrics_history['stochastic_universal'] = ga7.metrics_history
    
    # Test deterministic tournament selection
    config8 = copy.deepcopy(config1)
    config8.selection_method = SelectionMethod.TOURNAMENT_DETERMINISTIC
    
    ga8 = GeneticAlgorithm(config=config8)
    results['tournament_deterministic'] = ga8.run(max_generations=1000)
    all_metrics_history['tournament_deterministic'] = ga8.metrics_history
    
    # Test non-deterministic tournament selection
    config9 = copy.deepcopy(config1)
    config9.selection_method = SelectionMethod.TOURNAMENT_NONDETERMINISTIC
    
    ga9 = GeneticAlgorithm(config=config9)
    results['tournament_nondeterministic'] = ga9.run(max_generations=1000)
    all_metrics_history['tournament_nondeterministic'] = ga9.metrics_history
    
    # Test aging mechanism
    config10 = copy.deepcopy(config1)
    config10.selection_method = SelectionMethod.AGING
    
    ga10 = GeneticAlgorithm(config=config10)
    results['aging_mechanism'] = ga10.run(max_generations=1000)
    all_metrics_history['aging_mechanism'] = ga10.metrics_history
    
    # Create visualizations
    for name, df in results.items():
        plot_fitness_evolution(df, name)
        # Generate boxplots for each run
        plot_fitness_boxplots(all_metrics_history[name], name)
        # Plot exploration vs exploitation balance
        plot_exploration_exploitation_balance(df, name)
    
    # Compare crossover operators
    compare_crossover_operators(
        [results['single_point'], results['two_point'], results['uniform']],
        ['Single-Point', 'Two-Point', 'Uniform']
    )
    
    # Compare selection methods
    compare_selection_methods(
        [results['single_point'],  # This is roulette wheel 
         results['stochastic_universal'], 
         results['tournament_deterministic'], 
         results['tournament_nondeterministic'],
         results['aging_mechanism']],
        ['Roulette Wheel', 'Stochastic Universal', 'Tournament (Det)', 'Tournament (Non-Det)', 'Aging']
    )
    
    # Compare mutation and crossover configurations
    compare_mutation_crossover_configs(
        [results['single_point'], results['only_crossover'], results['only_mutation']],
        ['Crossover + Mutation', 'Only Crossover', 'Only Mutation']
    )
    
    # Compare fitness functions
    compare_fitness_functions(results['single_point'], results['lcs_fitness'])
    
    # Generate comprehensive exploration vs exploitation analysis
    
    # Run ARC challenge experiments
    print("\n===== Solving ARC Challenge Problems =====")
    
    # Define the example puzzles from the document
    puzzles = [
        {
            "name": "Checkerboard",
            "input_grid": [
                [1,0,1,0,1,0],
                [0,1,0,1,0,1],
                [1,0,1,0,1,0],
                [0,1,0,1,0,1],
                [1,0,1,0,1,0],
                [0,1,0,1,0,1]
            ],
            "target_grid": [
                [2,0,2,0,2,0],
                [0,2,0,2,0,2],
                [2,0,2,0,2,0],
                [0,2,0,2,0,2],
                [2,0,2,0,2,0],
                [0,2,0,2,0,2]
            ],
            "description": "Transform color 1 -> 2, leaving 0 alone"
        },
        {
            "name": "BoxInside",
            "input_grid": [
                [0,0,0,0,0],
                [0,1,1,1,0],
                [0,1,0,1,0],
                [0,1,1,1,0],
                [0,0,0,0,0]
            ],
            "target_grid": [
                [0,0,0,0,0],
                [0,1,1,1,0],
                [0,1,2,1,0],
                [0,1,1,1,0],
                [0,0,0,0,0]
            ],
            "description": "Fill center with color 2"
        },
        {
            "name": "RandomShapes1",
            "input_grid": [
                [0,0,1,1,0],
                [0,1,1,1,0],
                [1,1,0,1,1],
                [0,1,1,1,0],
                [0,0,1,1,0]
            ],
            "target_grid": [
                [0,0,2,2,0],
                [0,2,2,2,0],
                [2,2,0,2,2],
                [0,2,2,2,0],
                [0,0,2,2,0]
            ],
            "description": "Convert shape color from 1 -> 2"
        },
        {
            "name": "RandomShapes2",
            "input_grid": [
                [0,2,2,0,0],
                [0,2,1,2,0],
                [0,2,1,2,0],
                [0,2,1,2,0],
                [0,2,2,0,0]
            ],
            "target_grid": [
                [0,2,2,0,0],
                [0,2,0,2,0],
                [0,2,0,2,0],
                [0,2,0,2,0],
                [0,2,2,0,0]
            ],
            "description": "Remove color 1 => 0"
        },
        {
            "name": "LargerSquare",
            "input_grid": [
                [0,0,0,0,0],
                [0,1,1,1,0],
                [0,1,1,1,0],
                [0,1,1,1,0],
                [0,0,0,0,0]
            ],
            "target_grid": [
                [0,0,0,0,0],
                [0,0,1,0,0],
                [0,1,1,1,0],
                [0,0,1,0,0],
                [0,0,0,0,0]
            ],
            "description": "Shrink or recolor"
        }
    ]
    
    # Solve each puzzle
    for puzzle in puzzles:
        print(f"\n---- Solving {puzzle['name']} ({puzzle['description']}) ----")
        
        arc_problem = ARCProblem(puzzle["input_grid"], puzzle["target_grid"], 
                                puzzle["name"], puzzle["description"])
        arc_problem.solve_with_ga()
    
    # Run Bin Packing problem experiments
    print("\n===== Solving Bin Packing Problems =====")
    
    # Define some test instances
    bin_packing_instances = [
        {"name": "Instance 1", "items": [50, 30, 20, 10, 70, 40, 60, 80, 90, 15], "capacity": 100},
        {"name": "Instance 2", "items": [100, 150, 50, 75, 125, 80, 60, 40, 30, 20, 10], "capacity": 200},
        {"name": "Instance 3", "items": [500, 400, 300, 200, 100, 600, 700, 800, 900, 50, 150, 250, 350, 450], "capacity": 1000},
        {"name": "Instance 4", "items": [10, 20, 30, 40, 10, 10, 20, 20, 30, 30], "capacity": 50},
        {"name": "Instance 5", "items": [60, 70, 80, 90, 100, 110, 120, 30, 40, 50, 60, 70, 80, 90], "capacity": 150}
    ]
    
    bin_packing_results = []
    
    for i, instance in enumerate(bin_packing_instances):
        print(f"\n---- Solving Bin Packing {instance['name']} ----")
        
        bp_problem = BinPackingProblem(instance["items"], instance["capacity"])
        result = bp_problem.solve_with_ga()
        
        # Store results for comparison
        result['instance_name'] = instance['name']
        bin_packing_results.append(result)
    
    # Generate overall bin packing comparison
    if bin_packing_results:
        # Compare FF vs GA across all instances
        instance_names = [res['instance_name'] for res in bin_packing_results]
        ff_bins = [res['ff_bins_used'] for res in bin_packing_results]
        ga_bins = [res['ga_bins_used'] for res in bin_packing_results]
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(instance_names))
        width = 0.35
        
        plt.bar(x - width/2, ff_bins, width, label='First-Fit')
        plt.bar(x + width/2, ga_bins, width, label='Genetic Algorithm')
        
        plt.xlabel('Instance')
        plt.ylabel('Bins Used')
        plt.title('First-Fit vs Genetic Algorithm: Bins Used')
        plt.xticks(x, instance_names)
        plt.legend()
        plt.grid(axis='y')
        
        plt.tight_layout()
        plt.savefig('bin_packing_overall_comparison.png')
        plt.close()
        
        # Compare utilization
        ff_util = [res['ff_utilization'] for res in bin_packing_results]
        ga_util = [res['ga_utilization'] for res in bin_packing_results]
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, ff_util, width, label='First-Fit')
        plt.bar(x + width/2, ga_util, width, label='Genetic Algorithm')
        
        plt.xlabel('Instance')
        plt.ylabel('Bin Utilization (%)')
        plt.title('First-Fit vs Genetic Algorithm: Bin Utilization')
        plt.xticks(x, instance_names)
        plt.legend()
        plt.grid(axis='y')
        
        plt.tight_layout()
        plt.savefig('bin_packing_utilization_comparison.png')
        plt.close()
    
    print("\n===== Analysis of Exploration vs. Exploitation =====")
    
    # Create a comprehensive report on exploration vs. exploitation
    plt.figure(figsize=(12, 10))
    
    # Plot selection pressure across different selection methods
    plt.subplot(2, 1, 1)
    for method in ['single_point', 'stochastic_universal', 'tournament_deterministic', 
                   'tournament_nondeterministic', 'aging_mechanism']:
        if method in results:
            plt.plot(results[method]['generation'], results[method]['selection_pressure'], 
                    label=method.replace('_', ' ').title())
    
    plt.xlabel('Generation')
    plt.ylabel('Selection Pressure')
    plt.title('Selection Pressure Across Different Selection Methods')
    plt.legend()
    plt.grid(True)
    
    # Plot diversity metrics across different selection methods
    plt.subplot(2, 1, 2)
    for method in ['single_point', 'stochastic_universal', 'tournament_deterministic', 
                   'tournament_nondeterministic', 'aging_mechanism']:
        if method in results:
            plt.plot(results[method]['generation'], results[method]['genetic_diversity'], 
                    label=method.replace('_', ' ').title())
    
    plt.xlabel('Generation')
    plt.ylabel('Genetic Diversity')
    plt.title('Genetic Diversity Across Different Selection Methods')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('exploration_exploitation_analysis.png')
    plt.close()
    
    print("\nAll experiments completed successfully!")
    print("Check the generated visualizations for detailed analysis.")


if __name__ == "__main__":
    main()