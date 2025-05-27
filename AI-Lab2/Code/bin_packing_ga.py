# bin_packing_problem.py
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import heapq
from typing import List, Tuple, Dict, Any, Set
from collections import defaultdict

class Item:
    def __init__(self, size, item_id=None):
        self.size = size
        self.id = item_id
    
    def __repr__(self):
        return f"Item({self.id}: {self.size})"

class BinPackingIndividual:
    def __init__(self, bins=None):
        self.bins = bins if bins is not None else []  # List of lists, each inner list contains items in a bin
        self.fitness = 0.0
        self.age = 0
        self.bin_sizes = None  # Cache for bin sizes
        
    def copy(self):
        new_ind = BinPackingIndividual()
        new_ind.bins = [bin_items.copy() for bin_items in self.bins]
        new_ind.fitness = self.fitness
        new_ind.age = self.age
        return new_ind
    
    def get_bin_sizes(self):
        """Get the sizes of all bins, with caching"""
        if self.bin_sizes is None:
            self.bin_sizes = [sum(item.size for item in bin_items) for bin_items in self.bins]
        return self.bin_sizes
    
    def reset_bin_sizes(self):
        """Reset the cached bin sizes when bins are modified"""
        self.bin_sizes = None
    
    def __repr__(self):
        return f"BinPackingIndividual(bins={len(self.bins)}, fitness={self.fitness:.2f}, age={self.age})"

def generate_random_instance(num_items=100, min_size=1, max_size=100, bin_capacity=150):
    """Generate a random bin packing instance"""
    items = [Item(random.randint(min_size, max_size), i) for i in range(num_items)]
    return items, bin_capacity

def load_bin_packing_instance(filename):
    """Load bin packing instance from file"""
    items = []
    bin_capacity = None
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if i == 0:  # First line contains bin capacity
                bin_capacity = int(line)
            else:  # Remaining lines contain item sizes
                size = int(line)
                items.append(Item(size, i-1))
    
    return items, bin_capacity

def first_fit_heuristic(items, bin_capacity):
    """Solve bin packing using First Fit heuristic"""
    bins = []
    bin_sums = []  # Keep track of bin sums for faster lookup
    
    for item in items:
        placed = False
        
        # Try to place item in an existing bin
        for i, bin_sum in enumerate(bin_sums):
            if bin_sum + item.size <= bin_capacity:
                bins[i].append(item)
                bin_sums[i] += item.size
                placed = True
                break
        
        # If item couldn't fit in any existing bin, create a new one
        if not placed:
            bins.append([item])
            bin_sums.append(item.size)
    
    return bins

def initialize_population(items, bin_capacity, pop_size):
    """Initialize a population of bin packing solutions"""
    population = []
    
    # First individual using First Fit heuristic
    first_fit_bins = first_fit_heuristic(items, bin_capacity)
    population.append(BinPackingIndividual(first_fit_bins))
    
    # Randomly generate the rest
    for _ in range(pop_size - 1):
        # Random assignment of items to bins
        shuffled_items = items.copy()
        random.shuffle(shuffled_items)
        
        bins = []
        bin_sums = []  # Track bin sums for faster checking
        
        for item in shuffled_items:
            # Random choice: add to existing bin or create new bin
            if bins and random.random() < 0.7:  # 70% chance to add to existing bin
                # Select a bin where item fits
                valid_bins = [i for i, bin_sum in enumerate(bin_sums) if bin_sum + item.size <= bin_capacity]
                if valid_bins:
                    bin_idx = random.choice(valid_bins)
                    bins[bin_idx].append(item)
                    bin_sums[bin_idx] += item.size
                else:
                    # No bin has enough space, create new bin
                    bins.append([item])
                    bin_sums.append(item.size)
            else:  # 30% chance to create new bin
                bins.append([item])
                bin_sums.append(item.size)
        
        population.append(BinPackingIndividual(bins))
    
    return population

def repair_solution(bins, bin_capacity):
    """Repair a solution by resolving capacity violations"""
    repaired_bins = []
    bin_sums = []
    leftover_items = []
    
    # Check each bin for overload
    for bin_items in bins:
        bin_sum = sum(item.size for item in bin_items)
        
        if bin_sum <= bin_capacity:
            # Bin is valid, keep it
            repaired_bins.append(bin_items)
            bin_sums.append(bin_sum)
        else:
            # Bin is overloaded, sort items by size
            sorted_items = sorted(bin_items, key=lambda x: x.size, reverse=True)
            valid_bin = []
            current_sum = 0
            
            for item in sorted_items:
                if current_sum + item.size <= bin_capacity:
                    valid_bin.append(item)
                    current_sum += item.size
                else:
                    leftover_items.append(item)
            
            if valid_bin:
                repaired_bins.append(valid_bin)
                bin_sums.append(current_sum)
    
    # Sort leftover items by size (descending) for better packing
    leftover_items.sort(key=lambda x: x.size, reverse=True)
    
    # Repack leftover items using First Fit
    for item in leftover_items:
        placed = False
        for i, bin_sum in enumerate(bin_sums):
            if bin_sum + item.size <= bin_capacity:
                repaired_bins[i].append(item)
                bin_sums[i] += item.size
                placed = True
                break
        
        if not placed:
            repaired_bins.append([item])
            bin_sums.append(item.size)
    
    return repaired_bins

def evaluate_fitness(individual, bin_capacity):
    """Evaluate fitness of a bin packing solution
    Fitness goals:
    1. Minimize number of bins
    2. Maximize bin utilization (sum of items / capacity)
    """
    # Check if solution is valid
    valid = True
    bin_utilization = 0.0
    
    bin_sizes = individual.get_bin_sizes()
    if bin_sizes is None:
        bin_sizes = [sum(item.size for item in bin_items) for bin_items in individual.bins]
        individual.bin_sizes = bin_sizes
    
    for bin_size in bin_sizes:
        if bin_size > bin_capacity:
            valid = False
            break
        bin_utilization += bin_size / bin_capacity
    
    if not valid:
        individual.fitness = 0.0
        return 0.0
    
    # Calculate fitness based on number of bins and utilization
    num_bins = len(individual.bins)
    avg_utilization = bin_utilization / num_bins if num_bins > 0 else 0
    
    # We want to minimize bins and maximize utilization
    individual.fitness = 1.0 / (num_bins + 1) + 0.3 * avg_utilization
    return individual.fitness

def tournament_selection(population, tournament_size):
    """Select an individual using tournament selection"""
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda ind: ind.fitness)

def crossover_bin_packing(parent1, parent2, bin_capacity, crossover_rate=0.8):
    """Crossover operator for bin packing problem"""
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()
    
    # Create offspring
    child1 = BinPackingIndividual()
    child2 = BinPackingIndividual()
    
    # Group Crossover: Inherit some bins completely from each parent
    p1_bins = parent1.bins.copy()
    p2_bins = parent2.bins.copy()
    
    random.shuffle(p1_bins)
    random.shuffle(p2_bins)
    
    # Randomly choose some bins from each parent
    split_point1 = random.randint(1, max(1, len(p1_bins) - 1))
    split_point2 = random.randint(1, max(1, len(p2_bins) - 1))
    
    child1.bins = p1_bins[:split_point1] + p2_bins[split_point2:]
    child2.bins = p2_bins[:split_point2] + p1_bins[split_point1:]
    
    # Check which items are already included
    items_in_child1 = set()
    items_in_child2 = set()
    
    for bin_items in child1.bins:
        for item in bin_items:
            items_in_child1.add(item.id)
    
    for bin_items in child2.bins:
        for item in bin_items:
            items_in_child2.add(item.id)
    
    # Get all items
    all_items = set()
    for bin_items in parent1.bins:
        for item in bin_items:
            all_items.add(item)
    
    # Find missing items for each child
    missing_items1 = [item for item in all_items if item.id not in items_in_child1]
    missing_items2 = [item for item in all_items if item.id not in items_in_child2]
    
    # Find duplicate items for each child
    duplicate_items1 = []
    items_seen1 = set()
    
    for bin_idx, bin_items in enumerate(child1.bins):
        for i, item in enumerate(bin_items):
            if item.id in items_seen1:
                duplicate_items1.append((bin_idx, i))
            else:
                items_seen1.add(item.id)
    
    duplicate_items2 = []
    items_seen2 = set()
    
    for bin_idx, bin_items in enumerate(child2.bins):
        for i, item in enumerate(bin_items):
            if item.id in items_seen2:
                duplicate_items2.append((bin_idx, i))
            else:
                items_seen2.add(item.id)
    
    # Remove duplicates with safer approach
    duplicate_items1.sort(reverse=True)  # Start from end to avoid index problems
    for bin_idx, item_idx in duplicate_items1:
        if bin_idx < len(child1.bins) and item_idx < len(child1.bins[bin_idx]):
            del child1.bins[bin_idx][item_idx]
    
    duplicate_items2.sort(reverse=True)
    for bin_idx, item_idx in duplicate_items2:
        if bin_idx < len(child2.bins) and item_idx < len(child2.bins[bin_idx]):
            del child2.bins[bin_idx][item_idx]
    
    # Remove empty bins
    child1.bins = [bin_items for bin_items in child1.bins if bin_items]
    child2.bins = [bin_items for bin_items in child2.bins if bin_items]
    
    # Get child bin sizes for faster packing
    child1_bin_sums = [sum(item.size for item in bin_items) for bin_items in child1.bins]
    child2_bin_sums = [sum(item.size for item in bin_items) for bin_items in child2.bins]
    
    # Add missing items using First Fit
    for item in missing_items1:
        placed = False
        for i, bin_sum in enumerate(child1_bin_sums):
            if bin_sum + item.size <= bin_capacity:
                child1.bins[i].append(item)
                child1_bin_sums[i] += item.size
                placed = True
                break
        
        if not placed:
            child1.bins.append([item])
            child1_bin_sums.append(item.size)
    
    for item in missing_items2:
        placed = False
        for i, bin_sum in enumerate(child2_bin_sums):
            if bin_sum + item.size <= bin_capacity:
                child2.bins[i].append(item)
                child2_bin_sums[i] += item.size
                placed = True
                break
        
        if not placed:
            child2.bins.append([item])
            child2_bin_sums.append(item.size)
    
    # Remove empty bins (again, just to be sure)
    child1.bins = [bin_items for bin_items in child1.bins if bin_items]
    child2.bins = [bin_items for bin_items in child2.bins if bin_items]
    
    # Reset cached bin sizes
    child1.reset_bin_sizes()
    child2.reset_bin_sizes()
    
    return child1, child2

def mutate_bin_packing(individual, bin_capacity, mutation_rate=0.05):
    """Mutation operator for bin packing problem"""
    if random.random() > mutation_rate:
        return individual.copy()
    
    mutated_ind = individual.copy()
    num_bins = len(mutated_ind.bins)
    
    if num_bins <= 1:
        return mutated_ind
    
    # Choose mutation type
    mutation_type = random.choice(["swap", "move", "split", "merge"])
    
    # Get bin sizes for faster operations
    bin_sizes = mutated_ind.get_bin_sizes()
    if bin_sizes is None:
        bin_sizes = [sum(item.size for item in bin_items) for bin_items in mutated_ind.bins]
    
    if mutation_type == "swap":
        # Swap items between two bins
        bin1_idx = random.randint(0, num_bins - 1)
        bin2_idx = random.randint(0, num_bins - 1)
        
        if bin1_idx != bin2_idx and mutated_ind.bins[bin1_idx] and mutated_ind.bins[bin2_idx]:
            item1_idx = random.randint(0, len(mutated_ind.bins[bin1_idx]) - 1)
            item2_idx = random.randint(0, len(mutated_ind.bins[bin2_idx]) - 1)
            
            item1 = mutated_ind.bins[bin1_idx][item1_idx]
            item2 = mutated_ind.bins[bin2_idx][item2_idx]
            
            # Check if swap is valid
            new_bin1_sum = bin_sizes[bin1_idx] - item1.size + item2.size
            new_bin2_sum = bin_sizes[bin2_idx] - item2.size + item1.size
            
            if new_bin1_sum <= bin_capacity and new_bin2_sum <= bin_capacity:
                mutated_ind.bins[bin1_idx][item1_idx] = item2
                mutated_ind.bins[bin2_idx][item2_idx] = item1
                
                # Update bin sizes
                bin_sizes[bin1_idx] = new_bin1_sum
                bin_sizes[bin2_idx] = new_bin2_sum
                mutated_ind.bin_sizes = bin_sizes
    
    elif mutation_type == "move":
        # Move an item from one bin to another
        src_bin_idx = random.randint(0, num_bins - 1)
        
        if mutated_ind.bins[src_bin_idx]:
            item_idx = random.randint(0, len(mutated_ind.bins[src_bin_idx]) - 1)
            item = mutated_ind.bins[src_bin_idx][item_idx]
            
            # Find a valid target bin
            valid_targets = []
            for dst_bin_idx in range(num_bins):
                if dst_bin_idx != src_bin_idx:
                    if bin_sizes[dst_bin_idx] + item.size <= bin_capacity:
                        valid_targets.append(dst_bin_idx)
            
            if valid_targets:
                dst_bin_idx = random.choice(valid_targets)
                
                # Move the item
                mutated_ind.bins[dst_bin_idx].append(item)
                del mutated_ind.bins[src_bin_idx][item_idx]
                
                # Update bin sizes
                bin_sizes[dst_bin_idx] += item.size
                bin_sizes[src_bin_idx] -= item.size
                
                # Remove empty bins
                if not mutated_ind.bins[src_bin_idx]:
                    del mutated_ind.bins[src_bin_idx]
                    del bin_sizes[src_bin_idx]
                
                mutated_ind.bin_sizes = bin_sizes
    
    elif mutation_type == "split":
        # Split a bin into two
        if num_bins > 1:
            bin_idx = random.randint(0, num_bins - 1)
            
            if len(mutated_ind.bins[bin_idx]) > 1:
                # Randomly split the bin
                split_point = random.randint(1, len(mutated_ind.bins[bin_idx]) - 1)
                new_bin = mutated_ind.bins[bin_idx][split_point:]
                mutated_ind.bins[bin_idx] = mutated_ind.bins[bin_idx][:split_point]
                mutated_ind.bins.append(new_bin)
                
                # Update bin sizes
                new_bin_size = sum(item.size for item in new_bin)
                bin_sizes[bin_idx] -= new_bin_size
                bin_sizes.append(new_bin_size)
                mutated_ind.bin_sizes = bin_sizes
    
    elif mutation_type == "merge":
        # Try to merge two bins
        if num_bins > 1:
            bin1_idx = random.randint(0, num_bins - 1)
            bin2_idx = random.randint(0, num_bins - 1)
            
            if bin1_idx != bin2_idx:
                bin1_sum = bin_sizes[bin1_idx]
                bin2_sum = bin_sizes[bin2_idx]
                
                if bin1_sum + bin2_sum <= bin_capacity:
                    # Merge the bins
                    mutated_ind.bins[bin1_idx].extend(mutated_ind.bins[bin2_idx])
                    del mutated_ind.bins[bin2_idx]
                    
                    # Update bin sizes
                    bin_sizes[bin1_idx] += bin2_sum
                    del bin_sizes[bin2_idx]
                    mutated_ind.bin_sizes = bin_sizes
    
    # Reset bin sizes cache after significant changes
    mutated_ind.reset_bin_sizes()
    
    return mutated_ind

def bin_packing_ga(items, bin_capacity, population_size=100, generations=500, elite_size=10,
                  tournament_size=5, crossover_rate=0.8, mutation_rate=0.05,
                  mutation_policy="fixed", fitness_policy="standard", 
                  diversity_policy=None, policy_params=None, early_stop_generations=50):
    """Run the genetic algorithm for bin packing"""
    if policy_params is None:
        policy_params = {}
        
    # Initialize population
    population = initialize_population(items, bin_capacity, population_size)
    
    # Calculate initial fitness
    for individual in population:
        evaluate_fitness(individual, bin_capacity)
    
    best_individual = max(population, key=lambda ind: ind.fitness)
    best_fitness_history = [best_individual.fitness]
    avg_fitness_history = [sum(ind.fitness for ind in population) / len(population)]
    best_bins_history = [len(best_individual.bins)]
    
    # For early stopping
    generations_without_improvement = 0
    best_fitness_so_far = best_individual.fitness
    
    for generation in range(generations):
        # Apply diversity policy if specified
        if diversity_policy == "niching":
            radius = policy_params.get("fitness_radius", 0.2)
            # Implement niching for bin packing (similar to TSP version)
            # This is a simplified version for speed
            modified_population = copy.deepcopy(population)
            for i, ind1 in enumerate(modified_population):
                niche_count = 1
                for ind2 in population:
                    if ind1 != ind2:
                        # Simple similarity measure: ratio of bins with same items
                        similarity = 0.0
                        if similarity > (1 - radius):
                            niche_count += 1
                modified_population[i].fitness = ind1.fitness / max(1, niche_count)
            population = modified_population
        
        # Calculate current generation statistics
        current_best = max(population, key=lambda ind: ind.fitness)
        avg_fitness = sum(ind.fitness for ind in population) / len(population)
        
        # Create new generation
        next_population = []
        
        # Add elites
        elites = sorted(population, key=lambda ind: ind.fitness, reverse=True)[:elite_size]
        next_population.extend([elite.copy() for elite in elites])
        
        # Create rest of the population
        while len(next_population) < population_size:
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            
            child1, child2 = crossover_bin_packing(parent1, parent2, bin_capacity, crossover_rate)
            
            # Apply mutation based on policy
            if mutation_policy == "adaptive":
                # Adaptive mutation rate based on fitness
                rate1 = adaptive_mutation_rate(parent1, avg_fitness, current_best.fitness, mutation_rate)
                rate2 = adaptive_mutation_rate(parent2, avg_fitness, current_best.fitness, mutation_rate)
                
                child1 = mutate_bin_packing(child1, bin_capacity, rate1)
                child2 = mutate_bin_packing(child2, bin_capacity, rate2)
            
            elif mutation_policy == "hypermutation":
                # Hypermutation policy
                threshold = policy_params.get("threshold", 0.7)
                high_rate = policy_params.get("high_rate", 0.3)
                current_rate = hypermutation_policy(generation, generations, threshold, mutation_rate, high_rate)
                
                child1 = mutate_bin_packing(child1, bin_capacity, current_rate)
                child2 = mutate_bin_packing(child2, bin_capacity, current_rate)
            
            elif mutation_policy == "age_based":
                # Age-based mutation
                max_age = policy_params.get("max_age", 20)
                min_rate = policy_params.get("min_rate", 0.01)
                max_rate = policy_params.get("max_rate", 0.2)
                
                child1.age = parent1.age + 1
                child2.age = parent2.age + 1
                
                rate1 = age_based_mutation(child1, max_age, min_rate, max_rate)
                rate2 = age_based_mutation(child2, max_age, min_rate, max_rate)
                
                child1 = mutate_bin_packing(child1, bin_capacity, rate1)
                child2 = mutate_bin_packing(child2, bin_capacity, rate2)
            
            else:  # Default fixed mutation rate
                child1 = mutate_bin_packing(child1, bin_capacity, mutation_rate)
                child2 = mutate_bin_packing(child2, bin_capacity, mutation_rate)
            
            # Evaluate fitness and add to next generation
            evaluate_fitness(child1, bin_capacity)
            next_population.append(child1)
            
            if len(next_population) < population_size:
                evaluate_fitness(child2, bin_capacity)
                next_population.append(child2)
        
        # Apply fitness policy if specified
        if fitness_policy == "novelty":
            # Simplified novelty-based fitness for bin packing
            for ind in next_population:
                base_fitness = evaluate_fitness(ind, bin_capacity)
                novelty_bonus = 0.01 * random.random()  # Simplified for speed
                ind.fitness = base_fitness + novelty_bonus
        
        elif fitness_policy == "age_based":
            # Age-based fitness for bin packing
            for ind in next_population:
                base_fitness = evaluate_fitness(ind, bin_capacity)
                age_factor = min(1.0, ind.age / policy_params.get("max_age", 20))
                ind.fitness = base_fitness + age_factor * 0.01  # Small age bonus
        
        else:  # Default standard fitness
            for ind in next_population:
                evaluate_fitness(ind, bin_capacity)
        
        # Increment age for all individuals
        for ind in next_population:
            ind.age += 1
        
        # Update population
        population = next_population
        
        # Update statistics
        best_individual = max(population, key=lambda ind: ind.fitness)
        best_fitness_history.append(best_individual.fitness)
        avg_fitness_history.append(avg_fitness)
        best_bins_history.append(len(best_individual.bins))
        
        # Check for early stopping
        if best_individual.fitness > best_fitness_so_far:
            best_fitness_so_far = best_individual.fitness
            best_individual_so_far = best_individual.copy()
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1
            
        if generations_without_improvement >= early_stop_generations:
            print(f"Early stopping at generation {generation} after {early_stop_generations} generations without improvement")
            best_individual = best_individual_so_far  # Ensure we return the best ever found
            break
        
        # Optional: Print progress
        if generation % 50 == 0 or generation == generations - 1:
            print(f"Generation {generation}: Best bins = {len(best_individual.bins)}, Avg fitness = {avg_fitness:.6f}")
    
    # Return the best individual and history
    return best_individual, best_fitness_history, avg_fitness_history, best_bins_history

def adaptive_mutation_rate(individual, avg_fitness, best_fitness, base_rate=0.05):
    """Adjust mutation rate based on individual's fitness relative to population"""
    if best_fitness == avg_fitness:
        return base_rate
        
    relative_fitness = (individual.fitness - avg_fitness) / (best_fitness - avg_fitness)
    
    # Higher mutation for lower fitness
    return base_rate * (1.5 - relative_fitness * 0.5)

def hypermutation_policy(generation, max_generations, threshold=0.7, base_rate=0.05, high_rate=0.3):
    """Increase mutation rate when reaching a certain threshold of generations"""
    if generation > max_generations * threshold:
        return high_rate
    return base_rate

def age_based_mutation(individual, max_age=20, min_rate=0.01, max_rate=0.2):
    """Adjust mutation rate based on individual's age"""
    age_factor = min(1.0, individual.age / max_age)
    return min_rate + age_factor * (max_rate - min_rate)

def plot_bin_packing_results(best_fitness_history, avg_fitness_history, best_bins_history, title="Bin Packing Genetic Algorithm"):
    """Plot bin packing results"""
    plt.figure(figsize=(15, 10))
    
    # Plot fitness history
    plt.subplot(2, 1, 1)
    plt.plot(best_fitness_history, label='Best Fitness')
    plt.plot(avg_fitness_history, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (higher is better)')
    plt.title(f'{title} - Fitness')
    plt.legend()
    plt.grid(True)
    
    # Plot bins history
    plt.subplot(2, 1, 2)
    plt.plot(best_bins_history, label='Best Solution Bins')
    plt.xlabel('Generation')
    plt.ylabel('Number of Bins (lower is better)')
    plt.title(f'{title} - Bins')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def visualize_bin_packing_solution(individual, bin_capacity, title="Bin Packing Solution"):
    """Visualize a bin packing solution"""
    bins = individual.bins
    num_bins = len(bins)
    
    plt.figure(figsize=(15, 8))
    
    # Plot bin utilization as a bar chart
    bin_utilization = [sum(item.size for item in bin_items) / bin_capacity * 100 for bin_items in bins]
    bin_labels = [f"Bin {i+1}" for i in range(num_bins)]
    
    plt.bar(bin_labels, bin_utilization)
    plt.axhline(y=100, color='r', linestyle='--', label='Capacity')
    
    plt.xlabel('Bin')
    plt.ylabel('Utilization (%)')
    plt.title(f'{title} - {num_bins} Bins')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()
    
    # Also visualize the items in each bin
    plt.figure(figsize=(15, 10))
    
    # Define a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, 10))
    
    # Limit bins to show for large solutions
    bins_to_show = min(num_bins, 30)
    if bins_to_show < num_bins:
        print(f"Solution has {num_bins} bins, showing only first {bins_to_show} bins in visualization")
    
    for i, bin_items in enumerate(bins[:bins_to_show]):
        y_pos = bins_to_show - i - 1  # Start from top
        
        # Plot bin capacity
        plt.plot([0, bin_capacity], [y_pos, y_pos], 'k-', linewidth=2)
        
        # Plot items in the bin
        x_pos = 0
        for item in bin_items:
            plt.bar(x_pos + item.size/2, 0.8, width=item.size, bottom=y_pos-0.4, 
                   color=colors[item.size % 10], alpha=0.7)
            
            # Add item label if there's enough space
            if item.size > bin_capacity * 0.05:
                plt.text(x_pos + item.size/2, y_pos, str(item.id), 
                        ha='center', va='center', fontsize=8)
            
            x_pos += item.size
    
    plt.yticks(range(bins_to_show), bin_labels[:bins_to_show][::-1])
    plt.xlabel('Size')
    plt.title(f'{title} - Items in Bins')
    plt.grid(True)
    
    plt.savefig(f"{title.replace(' ', '_')}_items.png")
    plt.close()

def compare_bin_packing_policies(items, bin_capacity, policies, generations=300, runs=3, 
                                population_size=100, elite_size=10, tournament_size=5, 
                                early_stop_generations=50):
    """Compare different policies for bin packing problem"""
    results = {}
    
    for policy_name, (policy_type, policy, params) in policies.items():
        print(f"Testing {policy_name} policy...")
        policy_best_history = []
        policy_avg_history = []
        policy_bins_history = []
        
        for run in range(runs):
            if policy_type == "mutation":
                best_ind, best_history, avg_history, bins_history = bin_packing_ga(
                    items, bin_capacity, population_size=population_size, generations=generations,
                    elite_size=elite_size, tournament_size=tournament_size, crossover_rate=0.8, mutation_rate=0.05,
                    mutation_policy=policy, policy_params=params, early_stop_generations=early_stop_generations
                )
            elif policy_type == "fitness":
                best_ind, best_history, avg_history, bins_history = bin_packing_ga(
                    items, bin_capacity, population_size=population_size, generations=generations,
                    elite_size=elite_size, tournament_size=tournament_size, crossover_rate=0.8, mutation_rate=0.05,
                    fitness_policy=policy, policy_params=params, early_stop_generations=early_stop_generations
                )
            elif policy_type == "diversity":
                best_ind, best_history, avg_history, bins_history = bin_packing_ga(
                    items, bin_capacity, population_size=population_size, generations=generations,
                    elite_size=elite_size, tournament_size=tournament_size, crossover_rate=0.8, mutation_rate=0.05,
                    diversity_policy=policy, policy_params=params, early_stop_generations=early_stop_generations
                )
            
            # Normalize histories if they have different lengths
            min_length = min(len(best_history), len(avg_history), len(bins_history))
            policy_best_history.append(best_history[:min_length])
            policy_avg_history.append(avg_history[:min_length])
            policy_bins_history.append(bins_history[:min_length])
        
        # Average across runs
        # Find minimum length across all runs
        min_len = min(len(hist) for hist in policy_best_history)
        
        avg_best_history = [sum(hist[i] for hist in policy_best_history) / runs 
                           for i in range(min_len)]
        avg_avg_history = [sum(hist[i] for hist in policy_avg_history) / runs 
                          for i in range(min_len)]
        avg_bins_history = [sum(hist[i] for hist in policy_bins_history) / runs 
                           for i in range(min_len)]
        
        results[policy_name] = {
            'best_history': avg_best_history,
            'avg_history': avg_avg_history,
            'bins_history': avg_bins_history
        }
    
    # Plot comparison of fitness
    plt.figure(figsize=(12, 8))
    
    for policy_name, data in results.items():
        plt.plot(data['best_history'], label=f"{policy_name} (Best)")
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness (higher is better)')
    plt.title('Comparison of Policies - Fitness')
    plt.legend()
    plt.grid(True)
    plt.savefig("Bin_Packing_Policies_Fitness_Comparison.png")
    plt.close()
    
    # Plot comparison of bins
    plt.figure(figsize=(12, 8))
    
    for policy_name, data in results.items():
        plt.plot(data['bins_history'], label=f"{policy_name}")
    
    plt.xlabel('Generation')
    plt.ylabel('Number of Bins (lower is better)')
    plt.title('Comparison of Policies - Bins')
    plt.legend()
    plt.grid(True)
    plt.savefig("Bin_Packing_Policies_Bins_Comparison.png")
    plt.close()
    
    return results

def main():
    # Test with random instance
    items, bin_capacity = generate_random_instance(num_items=50, min_size=10, max_size=70, bin_capacity=100)
    
    # Run basic GA
    print("\nRunning basic genetic algorithm for bin packing...")
    best_individual, best_fitness_history, avg_fitness_history, best_bins_history = bin_packing_ga(
        items, bin_capacity, population_size=50, generations=100,
        elite_size=5, tournament_size=5, crossover_rate=0.8, mutation_rate=0.05,
        early_stop_generations=20
    )
    
    # Print results
    print(f"\nBest solution found:")
    print(f"Number of bins: {len(best_individual.bins)}")
    print(f"Fitness: {best_individual.fitness:.6f}")
    
    # Plot results
    plot_bin_packing_results(best_fitness_history, avg_fitness_history, best_bins_history)
    visualize_bin_packing_solution(best_individual, bin_capacity)
    
    # Compare mutation policies
    print("\nComparing mutation policies...")
    mutation_policies = {
        "Fixed": ("mutation", "fixed", {}),
        "Adaptive": ("mutation", "adaptive", {}),
        "Hypermutation": ("mutation", "hypermutation", {"threshold": 0.7, "high_rate": 0.3}),
        "Age-based": ("mutation", "age_based", {"max_age": 20, "min_rate": 0.01, "max_rate": 0.2})
    }
    
    compare_bin_packing_policies(items, bin_capacity, mutation_policies, generations=50, runs=1)
    
    # Compare fitness policies
    print("\nComparing fitness policies...")
    fitness_policies = {
        "Standard": ("fitness", "standard", {}),
        "Novelty": ("fitness", "novelty", {"k_neighbors": 5}),
        "Age-based": ("fitness", "age_based", {"max_age": 20})
    }
    
    compare_bin_packing_policies(items, bin_capacity, fitness_policies, generations=50, runs=1)
    
    # Compare diversity methods
    print("\nComparing diversity methods...")
    diversity_methods = {
        "None": ("diversity", None, {}),
        "Niching": ("diversity", "niching", {"fitness_radius": 0.2}),
    }
    
    compare_bin_packing_policies(items, bin_capacity, diversity_methods, generations=50, runs=1)

if __name__ == "__main__":
    main()