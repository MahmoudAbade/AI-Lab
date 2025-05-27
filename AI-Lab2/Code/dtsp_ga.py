import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import time
import os
from typing import List, Tuple, Dict, Any, Callable
import copy
import heapq
from collections import defaultdict

# Constants (reduced for speed)
POPULATION_SIZE = 100
ELITE_SIZE = 10
GENERATIONS = 200
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 5

class City:
    def __init__(self, x, y, city_id=None):
        self.x = x
        self.y = y
        self.id = city_id
    
    def distance(self, city):
        return math.sqrt((self.x - city.x)**2 + (self.y - city.y)**2)
    
    def __repr__(self):
        return f"City({self.id}: {self.x}, {self.y})"

class Individual:
    def __init__(self, chromosome1=None, chromosome2=None):
        self.chromosome1 = chromosome1  # First path
        self.chromosome2 = chromosome2  # Second path
        self.fitness = 0.0
        self.age = 0
        
    def __repr__(self):
        return f"Individual(fitness={self.fitness:.2f}, age={self.age})"

# Store distances between cities to avoid recalculation
distance_cache = {}

def clear_distance_cache():
    global distance_cache
    distance_cache = {}

def cached_distance(city1, city2):
    """Get distance with caching"""
    key = (id(city1), id(city2))
    if key not in distance_cache:
        distance_cache[key] = city1.distance(city2)
    return distance_cache[key]

def load_cities_from_file(filename):
    cities = []
    # Check if the file is in TSP or TSPLIB format
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Check if it's TSPLIB format
    if "NODE_COORD_SECTION" in ''.join(lines):
        processing_coords = False
        city_id = 1
        for line in lines:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                processing_coords = True
                continue
            if processing_coords and line not in ["EOF", "-1"]:
                parts = line.split()
                if len(parts) >= 3:
                    # TSPLIB format: ID X Y
                    city_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    cities.append(City(x, y, city_id))
            if line == "EOF":
                break
    else:
        # Assume simple format: x,y on each line
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    cities.append(City(x, y, i+1))
    
    # Clear cache for new cities
    clear_distance_cache()
    return cities

def precompute_distances(cities):
    """Precompute all pairwise distances between cities"""
    n = len(cities)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):  # Only compute upper triangle
            dist = cities[i].distance(cities[j])
            distances[i, j] = dist
            distances[j, i] = dist  # Matrix is symmetric
    return distances

def calculate_path_length(path, cities, distances=None):
    """Calculate total distance of a path with optional precomputed distances"""
    if distances is not None:
        # Use precomputed distances
        return sum(distances[path[i], path[(i+1) % len(path)]] for i in range(len(path)))
    else:
        # Calculate on the fly with caching
        return sum(cached_distance(cities[path[i]], cities[path[(i+1) % len(path)]]) for i in range(len(path)))

def generate_random_tour(num_cities):
    """Generate a random permutation of city indices"""
    tour = list(range(num_cities))
    random.shuffle(tour)
    return tour

def check_shared_edges(path1, path2):
    """Check if two paths share any edges using sets for speed"""
    edges1 = set()
    
    # Add edges from path1
    for i in range(len(path1)):
        city1 = path1[i]
        city2 = path1[(i+1) % len(path1)]
        edges1.add((min(city1, city2), max(city1, city2)))
    
    # Check edges from path2
    for i in range(len(path2)):
        city1 = path2[i]
        city2 = path2[(i+1) % len(path2)]
        edge = (min(city1, city2), max(city1, city2))
        
        if edge in edges1:
            return True
    
    return False

def initialize_population(cities, pop_size):
    """Initialize a population of individuals with random chromosomes"""
    population = []
    num_cities = len(cities)
    
    for _ in range(pop_size):
        # Generate two random tours
        chromosome1 = generate_random_tour(num_cities)
        chromosome2 = generate_random_tour(num_cities)
        
        # Ensure they don't share edges by regenerating chromosome2 if needed
        max_attempts = 20  # Limit attempts to prevent infinite loop
        attempts = 0
        while check_shared_edges(chromosome1, chromosome2) and attempts < max_attempts:
            chromosome2 = generate_random_tour(num_cities)
            attempts += 1
        
        # If we can't find a solution after many attempts, just use a simple repair function
        if attempts >= max_attempts:
            chromosome2 = repair_chromosomes(chromosome1, chromosome2, num_cities)
        
        individual = Individual(chromosome1, chromosome2)
        population.append(individual)
    
    return population

def repair_chromosomes(chromosome1, chromosome2, num_cities):
    """Repair the second chromosome to ensure no shared edges with the first"""
    # Create a set of edges in chromosome1 for quick lookup
    edges1 = set()
    for i in range(num_cities):
        city1 = chromosome1[i]
        city2 = chromosome1[(i+1) % num_cities]
        edges1.add((min(city1, city2), max(city1, city2)))
    
    # Make a copy of chromosome2 for modification
    repaired = chromosome2.copy()
    
    # Check each edge in repaired and swap if it's shared
    for i in range(num_cities):
        city1 = repaired[i]
        city2 = repaired[(i+1) % num_cities]
        edge = (min(city1, city2), max(city1, city2))
        
        if edge in edges1:
            # Find a swap that resolves this conflict
            for j in range(num_cities):
                if j != i and j != (i+1) % num_cities:
                    # Try swap
                    repaired[i], repaired[j] = repaired[j], repaired[i]
                    
                    # Check if this swap resolves the conflict
                    new_edge1 = (min(repaired[i], repaired[(i+1) % num_cities]), 
                                max(repaired[i], repaired[(i+1) % num_cities]))
                    new_edge2 = (min(repaired[j], repaired[(j+1) % num_cities]), 
                                max(repaired[j], repaired[(j+1) % num_cities]))
                    
                    if new_edge1 not in edges1 and new_edge2 not in edges1:
                        break
                    
                    # If not, revert the swap
                    repaired[i], repaired[j] = repaired[j], repaired[i]
    
    # If there are still shared edges, try a more aggressive approach
    if check_shared_edges(chromosome1, repaired):
        # Start with a random permutation and remove shared edges one by one
        repaired = list(range(num_cities))
        random.shuffle(repaired)
        
        # Keep removing shared edges until none remain or max iterations reached
        max_iterations = 100
        iterations = 0
        while check_shared_edges(chromosome1, repaired) and iterations < max_iterations:
            # Find one shared edge
            for i in range(num_cities):
                city1 = repaired[i]
                city2 = repaired[(i+1) % num_cities]
                edge = (min(city1, city2), max(city1, city2))
                
                if edge in edges1:
                    # Swap this city with a random city
                    j = random.randint(0, num_cities - 1)
                    repaired[i], repaired[j] = repaired[j], repaired[i]
                    break
            
            iterations += 1
    
    return repaired

def evaluate_fitness(individual, cities, distances=None):
    """Calculate fitness based on the length of the longer path"""
    length1 = calculate_path_length(individual.chromosome1, cities, distances)
    length2 = calculate_path_length(individual.chromosome2, cities, distances)
    
    # Fitness is negative of the longer path (since we want to minimize)
    individual.fitness = -max(length1, length2)
    return individual.fitness

def tournament_selection(population, tournament_size):
    """Select an individual using tournament selection"""
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda ind: ind.fitness)

def ordered_crossover(parent1, parent2):
    """Optimized ordered crossover for permutation representation"""
    size = len(parent1)
    
    # Select random start and end points for the crossover
    a = random.randint(0, size - 1)
    b = random.randint(0, size - 1)
    
    if a > b:
        a, b = b, a
        
    # Create child with the segment from parent1
    child = [-1] * size
    for i in range(a, b + 1):
        child[i] = parent1[i]
    
    # Create a set of cities already in the child for quick lookup
    used = set(parent1[a:b+1])
    
    # Fill the remaining positions with cities from parent2
    j = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[j] in used:
                j += 1
            child[i] = parent2[j]
            j += 1
    
    return child

def swap_mutation(chromosome, mutation_rate=0.05):
    """Swap mutation for permutation representation"""
    size = len(chromosome)
    for i in range(size):
        if random.random() < mutation_rate:
            j = random.randint(0, size - 1)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

def inversion_mutation(chromosome, mutation_rate=0.05):
    """Inversion mutation for permutation representation"""
    if random.random() < mutation_rate:
        size = len(chromosome)
        a = random.randint(0, size - 1)
        b = random.randint(0, size - 1)
        
        if a > b:
            a, b = b, a
            
        # Reverse the segment
        chromosome[a:b+1] = chromosome[a:b+1][::-1]
    
    return chromosome

def crossover_individuals(parent1, parent2, crossover_rate=0.8):
    """Crossover two individuals to produce two offspring"""
    if random.random() > crossover_rate:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)
    
    # Create the offspring
    child1 = Individual()
    child2 = Individual()
    
    # Apply crossover for both paths
    child1.chromosome1 = ordered_crossover(parent1.chromosome1, parent2.chromosome1)
    child1.chromosome2 = ordered_crossover(parent1.chromosome2, parent2.chromosome2)
    
    child2.chromosome1 = ordered_crossover(parent2.chromosome1, parent1.chromosome1)
    child2.chromosome2 = ordered_crossover(parent2.chromosome2, parent1.chromosome2)

    # Check if the crossover created shared edges and repair if needed
    if check_shared_edges(child1.chromosome1, child1.chromosome2):
        child1.chromosome2 = repair_chromosomes(child1.chromosome1, child1.chromosome2, len(child1.chromosome1))
    if check_shared_edges(child2.chromosome1, child2.chromosome2):
        child2.chromosome2 = repair_chromosomes(child2.chromosome1, child2.chromosome2, len(child2.chromosome1))
    
    return child1, child2

def mutate_individual(individual, mutation_rate=0.05):
    """Apply mutation to an individual"""
    # Make a copy to avoid modifying the original
    mutated = copy.deepcopy(individual)
    
    # Apply mutation to both chromosomes
    mutated.chromosome1 = swap_mutation(mutated.chromosome1, mutation_rate)
    mutated.chromosome2 = swap_mutation(mutated.chromosome2, mutation_rate)
    
    # Check if mutation created shared edges and repair if needed
    if check_shared_edges(mutated.chromosome1, mutated.chromosome2):
        mutated.chromosome2 = repair_chromosomes(mutated.chromosome1, mutated.chromosome2, len(mutated.chromosome1))
    
    return mutated

def elitism(population, elites_count):
    """Select the top individuals from the population"""
    return sorted(population, key=lambda ind: ind.fitness, reverse=True)[:elites_count]

def create_new_generation(population, cities, elite_size, tournament_size, crossover_rate, mutation_rate, distances=None):
    """Create a new generation through selection, crossover, and mutation"""
    next_population = []
    
    # Add elites
    elites = elitism(population, elite_size)
    next_population.extend(copy.deepcopy(elites))
    
    # Create rest of the population
    while len(next_population) < len(population):
        parent1 = tournament_selection(population, tournament_size)
        parent2 = tournament_selection(population, tournament_size)
        
        child1, child2 = crossover_individuals(parent1, parent2, crossover_rate)
        
        child1 = mutate_individual(child1, mutation_rate)
        child2 = mutate_individual(child2, mutation_rate)
        
        evaluate_fitness(child1, cities, distances)
        evaluate_fitness(child2, cities, distances)
        
        next_population.append(child1)
        
        if len(next_population) < len(population):
            next_population.append(child2)
    
    # Increment age for all individuals
    for ind in next_population:
        ind.age += 1
    
    return next_population

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

def novelty_based_fitness(individual, population, cities, k=5, distances=None):
    """Calculate fitness based on novelty (how different the individual is from others)"""
    # Calculate base fitness
    base_fitness = evaluate_fitness(individual, cities, distances)
    
    # Calculate average distance to k nearest neighbors
    distances_to_others = []
    for other in population:
        if other is not individual:
            similarity = gene_similarity(individual, other)
            distances_to_others.append(1.0 - similarity)  # Convert similarity to distance
    
    distances_to_others.sort()
    avg_distance = sum(distances_to_others[:k]) / k if len(distances_to_others) >= k else 0
    
    # Combine base fitness with novelty bonus
    novelty_bonus = avg_distance * abs(base_fitness * 0.1)  # Scale novelty by fitness
    return base_fitness + novelty_bonus

def age_based_fitness(individual, max_age=20, base_fitness=None):
    """Calculate fitness with age bonus to prevent premature convergence"""
    if base_fitness is None:
        base_fitness = individual.fitness
        
    age_bonus = (individual.age / max_age) * abs(base_fitness * 0.05)
    return base_fitness + age_bonus

def gene_similarity(individual1, individual2):
    """Measure similarity between two individuals' genes"""
    path1_similarity = path_similarity(individual1.chromosome1, individual2.chromosome1)
    path2_similarity = path_similarity(individual1.chromosome2, individual2.chromosome2)
    
    # Check cross similarity too (path1 vs path2)
    path_cross1 = path_similarity(individual1.chromosome1, individual2.chromosome2)
    path_cross2 = path_similarity(individual1.chromosome2, individual2.chromosome1)
    
    # Take maximum similarity (either direct or cross)
    direct_sim = (path1_similarity + path2_similarity) / 2
    cross_sim = (path_cross1 + path_cross2) / 2
    
    return max(direct_sim, cross_sim)

def path_similarity(path1, path2):
    """Calculate similarity between two paths (percentage of identical edges)"""
    edges1 = set()
    edges2 = set()
    
    # Get edges from path1
    for i in range(len(path1)):
        city1 = path1[i]
        city2 = path1[(i+1) % len(path1)]
        edges1.add((min(city1, city2), max(city1, city2)))
    
    # Get edges from path2
    for i in range(len(path2)):
        city1 = path2[i]
        city2 = path2[(i+1) % len(path2)]
        edges2.add((min(city1, city2), max(city1, city2)))
    
    # Calculate Jaccard similarity
    intersection = len(edges1.intersection(edges2))
    union = len(edges1.union(edges2))
    
    return intersection / union if union > 0 else 0.0

def niching_algorithm(population, fitness_radius=0.2):
    """Apply fitness sharing based on similarity within a radius"""
    modified_population = copy.deepcopy(population)
    
    # Calculate shared fitness for each individual
    for i, ind1 in enumerate(modified_population):
        niche_count = 0
        
        for j, ind2 in enumerate(population):
            similarity = gene_similarity(ind1, ind2)
            
            # If within radius, add to niche count
            if similarity > (1 - fitness_radius):
                niche_count += 1
        
        # Adjust fitness by dividing by niche count
        modified_population[i].fitness = ind1.fitness / max(1, niche_count)
    
    return modified_population

def speciation_algorithm(population, cities, similarity_threshold=0.3, target_species=5, distances=None):
    """Group individuals into species and adjust fitness accordingly"""
    # Recalculate fitness for each individual
    for ind in population:
        evaluate_fitness(ind, cities, distances)
    
    # Adaptive threshold to get the desired number of species
    current_threshold = similarity_threshold
    max_adjustments = 5
    adjustments = 0
    
    while adjustments < max_adjustments:
        # Assign individuals to species
        species = []
        unassigned = list(population)
        
        while unassigned:
            # Start a new species with the first unassigned individual
            current_species = [unassigned.pop(0)]
            i = 0
            
            # Process the current species
            while i < len(current_species):
                representative = current_species[i]
                j = 0
                
                # Check remaining unassigned individuals
                while j < len(unassigned):
                    if gene_similarity(representative, unassigned[j]) >= current_threshold:
                        current_species.append(unassigned.pop(j))
                    else:
                        j += 1
                
                i += 1
            
            species.append(current_species)
        
        # If we have the right number of species, or we can't adjust further, break
        if len(species) == target_species or adjustments == max_adjustments - 1:
            break
            
        # Adjust threshold if needed
        if len(species) < target_species:
            current_threshold += 0.05  # Make it harder to be in same species
        else:
            current_threshold -= 0.05  # Make it easier to be in same species
            
        current_threshold = max(0.1, min(0.9, current_threshold))
        adjustments += 1
    
    # Adjust fitness within each species
    for specie in species:
        # Find best fitness in species
        best_fitness = max(ind.fitness for ind in specie)
        
        # Adjust all fitnesses in the species relative to the best
        for ind in specie:
            # Scale by species size to promote smaller species
            scaling_factor = math.sqrt(len(population) / len(specie))
            ind.fitness = ind.fitness * scaling_factor
    
    # Combine all species back into one population
    result_population = []
    for specie in species:
        result_population.extend(specie)
    
    return result_population

def run_genetic_algorithm(cities, population_size=100, generations=500, elite_size=10, 
                         tournament_size=5, crossover_rate=0.8, mutation_rate=0.05,
                         mutation_policy="fixed", fitness_policy="standard", 
                         diversity_policy=None, policy_params=None, early_stop_generations=50):
    """Run the genetic algorithm with specified parameters and policies"""
    if policy_params is None:
        policy_params = {}
    
    # Precompute distances for faster fitness calculation
    distances = precompute_distances(cities)
    
    # Initialize population
    population = initialize_population(cities, population_size)
    
    # Calculate initial fitness
    for individual in population:
        evaluate_fitness(individual, cities, distances)
    
    best_individual = max(population, key=lambda ind: ind.fitness)
    best_fitness_history = [-best_individual.fitness]  # Negative to convert back to distance
    avg_fitness_history = [-sum(ind.fitness for ind in population) / len(population)]
    
    # For early stopping
    generations_without_improvement = 0
    best_fitness_so_far = best_individual.fitness
    
    for generation in range(generations):
        # Apply diversity policy if specified
        if diversity_policy == "niching":
            radius = policy_params.get("fitness_radius", 0.2)
            population = niching_algorithm(population, radius)
        elif diversity_policy == "speciation":
            threshold = policy_params.get("similarity_threshold", 0.3)
            target_species = policy_params.get("target_species", 5)
            population = speciation_algorithm(population, cities, threshold, target_species, distances)
        
        # Calculate current generation statistics
        current_best = max(population, key=lambda ind: ind.fitness)
        avg_fitness = sum(ind.fitness for ind in population) / len(population)
        
        # Create new generation
        if mutation_policy == "adaptive":
            # For adaptive mutation, we need to calculate new rates for each individual
            next_population = []
            
            # Add elites
            elites = elitism(population, elite_size)
            next_population.extend(copy.deepcopy(elites))
            
            # Create rest of the population with adaptive mutation
            while len(next_population) < len(population):
                parent1 = tournament_selection(population, tournament_size)
                parent2 = tournament_selection(population, tournament_size)
                
                child1, child2 = crossover_individuals(parent1, parent2, crossover_rate)
                
                # Calculate adaptive mutation rates
                rate1 = adaptive_mutation_rate(parent1, avg_fitness, current_best.fitness, mutation_rate)
                rate2 = adaptive_mutation_rate(parent2, avg_fitness, current_best.fitness, mutation_rate)
                
                child1 = mutate_individual(child1, rate1)
                child2 = mutate_individual(child2, rate2)
                
                # Evaluate fitness
                evaluate_fitness(child1, cities, distances)
                evaluate_fitness(child2, cities, distances)
                
                next_population.append(child1)
                if len(next_population) < len(population):
                    next_population.append(child2)
            
            # Increment age
            for ind in next_population:
                ind.age += 1
                
            population = next_population
        
        elif mutation_policy == "hypermutation":
            # Use hypermutation policy
            threshold = policy_params.get("threshold", 0.7)
            high_rate = policy_params.get("high_rate", 0.3)
            current_rate = hypermutation_policy(generation, generations, threshold, mutation_rate, high_rate)
            
            population = create_new_generation(population, cities, elite_size, tournament_size, 
                                              crossover_rate, current_rate, distances)
        
        elif mutation_policy == "age_based":
            # For age-based mutation, calculate rate based on individual's age
            next_population = []
            
            # Add elites
            elites = elitism(population, elite_size)
            next_population.extend(copy.deepcopy(elites))
            
            # Create rest of the population with age-based mutation
            while len(next_population) < len(population):
                parent1 = tournament_selection(population, tournament_size)
                parent2 = tournament_selection(population, tournament_size)
                
                child1, child2 = crossover_individuals(parent1, parent2, crossover_rate)
                
                # Calculate age-based mutation rates
                max_age = policy_params.get("max_age", 20)
                min_rate = policy_params.get("min_rate", 0.01)
                max_rate = policy_params.get("max_rate", 0.2)
                
                rate1 = age_based_mutation(parent1, max_age, min_rate, max_rate)
                rate2 = age_based_mutation(parent2, max_age, min_rate, max_rate)
                
                child1 = mutate_individual(child1, rate1)
                child2 = mutate_individual(child2, rate2)
                
                # Evaluate fitness
                evaluate_fitness(child1, cities, distances)
                evaluate_fitness(child2, cities, distances)
                
                next_population.append(child1)
                if len(next_population) < len(population):
                    next_population.append(child2)
            
            # Increment age
            for ind in next_population:
                ind.age += 1
                
            population = next_population
        
        else:  # Default fixed mutation rate
            population = create_new_generation(population, cities, elite_size, tournament_size, 
                                              crossover_rate, mutation_rate, distances)
        
        # Apply fitness policy if specified
        if fitness_policy == "novelty":
            for ind in population:
                ind.fitness = novelty_based_fitness(ind, population, cities, 
                                                  policy_params.get("k_neighbors", 5), distances)
        elif fitness_policy == "age_based":
            for ind in population:
                base_fitness = evaluate_fitness(ind, cities, distances)
                ind.fitness = age_based_fitness(ind, policy_params.get("max_age", 20), base_fitness)
        else:  # Default standard fitness
            for ind in population:
                evaluate_fitness(ind, cities, distances)
        
        # Update statistics
        best_individual = max(population, key=lambda ind: ind.fitness)
        best_fitness_history.append(-best_individual.fitness)
        avg_fitness_history.append(-avg_fitness)
        
        # Check for early stopping
        if best_individual.fitness > best_fitness_so_far:
            best_fitness_so_far = best_individual.fitness
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1
            
        if generations_without_improvement >= early_stop_generations:
            print(f"Early stopping at generation {generation} after {early_stop_generations} generations without improvement")
            break
        
        # Optional: Print progress
        if generation % 50 == 0 or generation == generations - 1:
            print(f"Generation {generation}: Best fitness = {-best_individual.fitness:.2f}, Avg fitness = {-avg_fitness:.2f}")
    
    # Return the best individual and history
    return best_individual, best_fitness_history, avg_fitness_history

def plot_results(best_fitness_history, avg_fitness_history, title="Fitness Evolution"):
    """Plot the evolution of fitness over generations"""
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_history, label='Best Fitness')
    plt.plot(avg_fitness_history, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Tour Length (smaller is better)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def plot_tour(cities, tour, title="TSP Tour"):
    """Plot the tour on a 2D map"""
    plt.figure(figsize=(10, 8))
    
    # Plot all cities
    x = [city.x for city in cities]
    y = [city.y for city in cities]
    plt.scatter(x, y, c='blue', s=50)
    
    # Plot tour
    for i in range(len(tour)):
        city1 = cities[tour[i]]
        city2 = cities[tour[(i+1) % len(tour)]]
        plt.plot([city1.x, city2.x], [city1.y, city2.y], 'r-', alpha=0.6)
    
    # Add city labels if not too many cities
    if len(cities) <= 100:
        for i, city in enumerate(cities):
            plt.text(city.x, city.y, str(city.id if city.id is not None else i), fontsize=9)
    
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def plot_double_tour(cities, tour1, tour2, title="Double TSP Solution"):
    """Plot two tours on the same map with different colors"""
    plt.figure(figsize=(10, 8))
    
    # Plot all cities
    x = [city.x for city in cities]
    y = [city.y for city in cities]
    plt.scatter(x, y, c='blue', s=50)
    
    # Plot first tour in red
    for i in range(len(tour1)):
        city1 = cities[tour1[i]]
        city2 = cities[tour1[(i+1) % len(tour1)]]
        plt.plot([city1.x, city2.x], [city1.y, city2.y], 'r-', alpha=0.6)
    
    # Plot second tour in green
    for i in range(len(tour2)):
        city1 = cities[tour2[i]]
        city2 = cities[tour2[(i+1) % len(tour2)]]
        plt.plot([city1.x, city2.x], [city1.y, city2.y], 'g--', alpha=0.6)
    
    # Add city labels if not too many cities
    if len(cities) <= 100:
        for i, city in enumerate(cities):
            plt.text(city.x, city.y, str(city.id if city.id is not None else i), fontsize=9)
    
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(['Cities', 'Tour 1', 'Tour 2'])
    plt.grid(True)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def compare_mutation_policies(cities, policies, generations=300, runs=3, population_size=None, early_stop_generations=50):
    """Compare different mutation policies"""
    if population_size is None:
        population_size = POPULATION_SIZE
        
    results = {}
    
    for policy_name, (policy, params) in policies.items():
        print(f"Testing {policy_name} mutation policy...")
        policy_best_history = []
        policy_avg_history = []
        
        for run in range(runs):
            best_ind, best_history, avg_history = run_genetic_algorithm(
                cities, population_size=population_size, generations=generations,
                elite_size=int(population_size*0.1), tournament_size=TOURNAMENT_SIZE,
                crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE,
                mutation_policy=policy, policy_params=params,
                early_stop_generations=early_stop_generations
            )
            
            policy_best_history.append(best_history)
            policy_avg_history.append(avg_history)
        
        # Average across runs
        avg_best_history = [sum(hist[i] for hist in policy_best_history) / runs 
                           for i in range(min(len(h) for h in policy_best_history))]
        avg_avg_history = [sum(hist[i] for hist in policy_avg_history) / runs 
                          for i in range(min(len(h) for h in policy_avg_history))]
        
        results[policy_name] = {
            'best_history': avg_best_history,
            'avg_history': avg_avg_history
        }
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    for policy_name, data in results.items():
        plt.plot(data['best_history'], label=f"{policy_name} (Best)")
    
    plt.xlabel('Generation')
    plt.ylabel('Tour Length (smaller is better)')
    plt.title('Comparison of Mutation Policies - Best Fitness')
    plt.legend()
    plt.grid(True)
    plt.savefig("Mutation_Policies_Comparison.png")
    plt.close()
    
    return results

def compare_fitness_policies(cities, policies, generations=300, runs=3, population_size=None, early_stop_generations=50):
    """Compare different fitness policies"""
    if population_size is None:
        population_size = POPULATION_SIZE
        
    results = {}
    
    for policy_name, (policy, params) in policies.items():
        print(f"Testing {policy_name} fitness policy...")
        policy_best_history = []
        policy_avg_history = []
        
        for run in range(runs):
            best_ind, best_history, avg_history = run_genetic_algorithm(
                cities, population_size=population_size, generations=generations,
                elite_size=int(population_size*0.1), tournament_size=TOURNAMENT_SIZE,
                crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE,
                fitness_policy=policy, policy_params=params,
                early_stop_generations=early_stop_generations
            )
            
            policy_best_history.append(best_history)
            policy_avg_history.append(avg_history)
        
        # Average across runs
        avg_best_history = [sum(hist[i] for hist in policy_best_history) / runs 
                           for i in range(min(len(h) for h in policy_best_history))]
        avg_avg_history = [sum(hist[i] for hist in policy_avg_history) / runs 
                          for i in range(min(len(h) for h in policy_avg_history))]
        
        results[policy_name] = {
            'best_history': avg_best_history,
            'avg_history': avg_avg_history
        }
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    for policy_name, data in results.items():
        plt.plot(data['best_history'], label=f"{policy_name} (Best)")
    
    plt.xlabel('Generation')
    plt.ylabel('Tour Length (smaller is better)')
    plt.title('Comparison of Fitness Policies - Best Fitness')
    plt.legend()
    plt.grid(True)
    plt.savefig("Fitness_Policies_Comparison.png")
    plt.close()
    
    return results

def compare_diversity_methods(cities, methods, generations=300, runs=3, population_size=None, early_stop_generations=50):
    """Compare different diversity maintenance methods"""
    if population_size is None:
        population_size = POPULATION_SIZE
        
    results = {}
    
    for method_name, (method, params) in methods.items():
        print(f"Testing {method_name} diversity method...")
        method_best_history = []
        method_avg_history = []
        
        for run in range(runs):
            best_ind, best_history, avg_history = run_genetic_algorithm(
                cities, population_size=population_size, generations=generations,
                elite_size=int(population_size*0.1), tournament_size=TOURNAMENT_SIZE,
                crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE,
                diversity_policy=method, policy_params=params,
                early_stop_generations=early_stop_generations
            )
            
            method_best_history.append(best_history)
            method_avg_history.append(avg_history)
        
        # Average across runs
        max_len = min(len(h) for h in method_best_history)
        avg_best_history = [sum(hist[i] for hist in method_best_history) / runs 
                           for i in range(max_len)]
        avg_avg_history = [sum(hist[i] for hist in method_avg_history) / runs 
                          for i in range(max_len)]
        
        results[method_name] = {
            'best_history': avg_best_history,
            'avg_history': avg_avg_history
        }
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    for method_name, data in results.items():
        plt.plot(data['best_history'], label=f"{method_name} (Best)")
    
    plt.xlabel('Generation')
    plt.ylabel('Tour Length (smaller is better)')
    plt.title('Comparison of Diversity Methods - Best Fitness')
    plt.legend()
    plt.grid(True)
    plt.savefig("Diversity_Methods_Comparison.png")
    plt.close()
    
    return results

def parameter_sensitivity_analysis(cities, parameter_name, parameter_values, generations=200, runs=1):
    """Analyze the sensitivity of the algorithm to different parameter values"""
    results = {}
    
    for value in parameter_values:
        print(f"Testing {parameter_name} = {value}...")
        value_best_history = []
        
        for run in range(runs):
            # Set up the right parameters based on the parameter name
            if parameter_name == "mutation_rate":
                best_ind, best_history, _ = run_genetic_algorithm(
                    cities, population_size=POPULATION_SIZE, generations=generations,
                    elite_size=ELITE_SIZE, tournament_size=TOURNAMENT_SIZE,
                    crossover_rate=CROSSOVER_RATE, mutation_rate=value,
                    early_stop_generations=int(generations/4)
                )
            elif parameter_name == "crossover_rate":
                best_ind, best_history, _ = run_genetic_algorithm(
                    cities, population_size=POPULATION_SIZE, generations=generations,
                    elite_size=ELITE_SIZE, tournament_size=TOURNAMENT_SIZE,
                    crossover_rate=value, mutation_rate=MUTATION_RATE,
                    early_stop_generations=int(generations/4)
                )
            elif parameter_name == "elite_size":
                best_ind, best_history, _ = run_genetic_algorithm(
                    cities, population_size=POPULATION_SIZE, generations=generations,
                    elite_size=value, tournament_size=TOURNAMENT_SIZE,
                    crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE,
                    early_stop_generations=int(generations/4)
                )
            elif parameter_name == "tournament_size":
                best_ind, best_history, _ = run_genetic_algorithm(
                    cities, population_size=POPULATION_SIZE, generations=generations,
                    elite_size=ELITE_SIZE, tournament_size=value,
                    crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE,
                    early_stop_generations=int(generations/4)
                )
            elif parameter_name == "population_size":
                best_ind, best_history, _ = run_genetic_algorithm(
                    cities, population_size=value, generations=generations,
                    elite_size=min(ELITE_SIZE, int(value * 0.1)),
                    tournament_size=TOURNAMENT_SIZE,
                    crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE,
                    early_stop_generations=int(generations/4)
                )
            elif parameter_name == "fitness_radius":
                best_ind, best_history, _ = run_genetic_algorithm(
                    cities, population_size=POPULATION_SIZE, generations=generations,
                    elite_size=ELITE_SIZE, tournament_size=TOURNAMENT_SIZE,
                    crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE,
                    diversity_policy="niching", policy_params={"fitness_radius": value},
                    early_stop_generations=int(generations/4)
                )
            elif parameter_name == "similarity_threshold":
                best_ind, best_history, _ = run_genetic_algorithm(
                    cities, population_size=POPULATION_SIZE, generations=generations,
                    elite_size=ELITE_SIZE, tournament_size=TOURNAMENT_SIZE,
                    crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE,
                    diversity_policy="speciation", policy_params={"similarity_threshold": value},
                    early_stop_generations=int(generations/4)
                )
            
            value_best_history.append(best_history)
        
        # Average across runs
        min_len = min(len(h) for h in value_best_history)
        avg_best_history = [sum(hist[i] for hist in value_best_history) / runs 
                           for i in range(min_len)]
        
        results[value] = {
            'best_history': avg_best_history,
            'final_best': avg_best_history[-1]
        }
    
    # Plot comparison of evolution
    plt.figure(figsize=(12, 8))
    
    for value, data in results.items():
        plt.plot(data['best_history'], label=f"{parameter_name} = {value}")
    
    plt.xlabel('Generation')
    plt.ylabel('Tour Length (smaller is better)')
    plt.title(f'Sensitivity Analysis - {parameter_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Sensitivity_{parameter_name}_Evolution.png")
    plt.close()
    
    # Plot final best fitness vs parameter value
    plt.figure(figsize=(10, 6))
    
    x_values = list(results.keys())
    y_values = [data['final_best'] for data in results.values()]
    
    plt.plot(x_values, y_values, 'o-')
    plt.xlabel(parameter_name)
    plt.ylabel('Final Best Tour Length')
    plt.title(f'Final Performance vs {parameter_name}')
    plt.grid(True)
    plt.savefig(f"Sensitivity_{parameter_name}_Final.png")
    plt.close()
    
    return results

def run_baldwin_effect_experiment(target_length=20, population_size=1000, generations=100, 
                                 learning_attempts=1000, crossover_rate=0.8, mutation_rate=0.02):
    """Implement the Baldwin Effect experiment as described by Hinton and Nolan"""
    # Initialize target genome
    random.seed(42)  # For reproducibility
    target_genome = []
    for _ in range(target_length):
        target_genome.append(random.choice(['0', '1', '?']))
    
    # Initialize population
    population = []
    for _ in range(population_size):
        genome = []
        for i in range(target_length):
            r = random.random()
            if r < 0.25:  # 25% correct
                genome.append(target_genome[i] if target_genome[i] != '?' else random.choice(['0', '1']))
            elif r < 0.5:  # 25% incorrect
                if target_genome[i] == '0':
                    genome.append('1')
                elif target_genome[i] == '1':
                    genome.append('0')
                else:  # If target is '?'
                    genome.append(random.choice(['0', '1']))
            else:  # 50% unknown
                genome.append('?')
        population.append(genome)
    
    # Track statistics
    mismatches_history = []
    correct_positions_history = []
    learned_bits_history = []
    
    for generation in range(generations):
        print(f"Generation {generation}")
        
        # Evaluate fitness through learning
        fitness_scores = []
        learned_genome_counts = []
        
        for genome in population:
            # Make a copy for learning
            learned_genome = genome.copy()
            
            # Count initial mismatches
            initial_mismatches = 0
            for i in range(target_length):
                if target_genome[i] != '?' and learned_genome[i] != '?' and learned_genome[i] != target_genome[i]:
                    initial_mismatches += 1
            
            # Learning phase - make random guesses
            learned_bits = 0
            for _ in range(learning_attempts):
                # Randomly select an unknown position
                unknown_positions = [i for i in range(target_length) if learned_genome[i] == '?']
                if not unknown_positions:
                    break
                
                pos = random.choice(unknown_positions)
                guess = random.choice(['0', '1'])
                
                # Check if guess is correct
                if target_genome[pos] == '?' or target_genome[pos] == guess:
                    learned_genome[pos] = guess
                    learned_bits += 1
            
            # Count final mismatches after learning
            final_mismatches = 0
            for i in range(target_length):
                if target_genome[i] != '?' and learned_genome[i] != '?' and learned_genome[i] != target_genome[i]:
                    final_mismatches += 1
            
            # Fitness is inversely proportional to mismatches
            fitness = 1.0 / (1.0 + final_mismatches)
            fitness_scores.append(fitness)
            learned_genome_counts.append(learned_bits)
        
        # Track statistics
        avg_mismatches = sum(1 for genome in population for i in range(target_length) 
                            if target_genome[i] != '?' and genome[i] != '?' and genome[i] != target_genome[i]) / population_size
        
        avg_correct = sum(1 for genome in population for i in range(target_length) 
                         if target_genome[i] != '?' and genome[i] == target_genome[i]) / population_size
        
        avg_learned_bits = sum(learned_genome_counts) / population_size
        
        mismatches_history.append(avg_mismatches)
        correct_positions_history.append(avg_correct)
        learned_bits_history.append(avg_learned_bits)
        
        # Create next generation
        next_population = []
        
        # Use tournament selection and crossover
        while len(next_population) < population_size:
            # Tournament selection
            tournament_size = 5
            parent1_idx = random.randrange(population_size)
            for _ in range(tournament_size - 1):
                idx = random.randrange(population_size)
                if fitness_scores[idx] > fitness_scores[parent1_idx]:
                    parent1_idx = idx
            
            parent2_idx = random.randrange(population_size)
            for _ in range(tournament_size - 1):
                idx = random.randrange(population_size)
                if fitness_scores[idx] > fitness_scores[parent2_idx]:
                    parent2_idx = idx
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            if random.random() < crossover_rate:
                crossover_point = random.randint(1, target_length - 1)
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()
            
            # Mutation
            for i in range(target_length):
                if random.random() < mutation_rate:
                    child1[i] = random.choice(['0', '1', '?'])
                if random.random() < mutation_rate:
                    child2[i] = random.choice(['0', '1', '?'])
            
            next_population.append(child1)
            if len(next_population) < population_size:
                next_population.append(child2)
        
        population = next_population
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(mismatches_history, label='Average Mismatches')
    plt.plot(correct_positions_history, label='Average Correct Positions')
    plt.plot(learned_bits_history, label='Average Bits Learned')
    plt.xlabel('Generation')
    plt.ylabel('Count')
    plt.title('Baldwin Effect Experiment')
    plt.legend()
    plt.grid(True)
    plt.savefig("Baldwin_Effect_Experiment.png")
    plt.close()
    
    # Return statistics history
    return {
        'mismatches': mismatches_history,
        'correct_positions': correct_positions_history,
        'learned_bits': learned_bits_history
    }

def main():
    # Load problem instances
    tsp_files = ['all/eil51.tsp', 'all/st70.tsp', 'all/pr76.tsp', 'all/kroA100.tsp']
    
    for tsp_file in tsp_files:
        print(f"\nSolving {tsp_file}...")
        cities = load_cities_from_file(tsp_file)
        
        # Run basic GA with reduced parameters for speed
        print("\nRunning basic genetic algorithm...")
        best_individual, best_history, avg_history = run_genetic_algorithm(
            cities, population_size=100, generations=100,
            elite_size=10, tournament_size=5,
            crossover_rate=0.8, mutation_rate=0.05,
            early_stop_generations=20
        )
        
        # Print results
        length1 = calculate_path_length(best_individual.chromosome1, cities)
        length2 = calculate_path_length(best_individual.chromosome2, cities)
        longer_length = max(length1, length2)
        
        print(f"\nBest solution found:")
        print(f"Path 1 length: {length1:.2f}")
        print(f"Path 2 length: {length2:.2f}")
        print(f"Longer path length (objective): {longer_length:.2f}")
        
        # Plot results
        plot_results(best_history, avg_history, f"{tsp_file.split('.')[0]} - Fitness Evolution")
        plot_double_tour(cities, best_individual.chromosome1, best_individual.chromosome2, 
                        f"{tsp_file.split('.')[0]} - Double TSP Solution")
        
        # Run only selected experiments for demonstration
        if tsp_file == tsp_files[0]:  # Only run for the first problem
            # Compare mutation policies
            print("\nComparing mutation policies...")
            mutation_policies = {
                "Fixed": ("fixed", {}),
                "Adaptive": ("adaptive", {}),
                "Hypermutation": ("hypermutation", {"threshold": 0.7, "high_rate": 0.3}),
                "Age-based": ("age_based", {"max_age": 20, "min_rate": 0.01, "max_rate": 0.2})
            }
            
            mutation_results = compare_mutation_policies(cities, mutation_policies, generations=100, runs=1)
            
            # Compare fitness policies
            print("\nComparing fitness policies...")
            fitness_policies = {
                "Standard": ("standard", {}),
                "Novelty": ("novelty", {"k_neighbors": 5}),
                "Age-based": ("age_based", {"max_age": 20})
            }
            
            fitness_results = compare_fitness_policies(cities, fitness_policies, generations=100, runs=1)
            
            # Compare diversity methods
            print("\nComparing diversity methods...")
            diversity_methods = {
                "None": (None, {}),
                "Niching": ("niching", {"fitness_radius": 0.2}),
                "Speciation": ("speciation", {"similarity_threshold": 0.3, "target_species": 5})
            }
            
            diversity_results = compare_diversity_methods(cities, diversity_methods, generations=100, runs=1)
            
            # Parameter sensitivity analysis
            print("\nPerforming parameter sensitivity analysis...")
            mutation_rates = [0.01, 0.05, 0.1]
            mutation_sensitivity = parameter_sensitivity_analysis(cities, "mutation_rate", mutation_rates, generations=50)
    
    # Run Baldwin Effect experiment with reduced parameters
    print("\nRunning Baldwin Effect experiment...")
    baldwin_results = run_baldwin_effect_experiment(
        target_length=10, 
        population_size=200, 
        generations=20, 
        learning_attempts=200, 
        crossover_rate=0.8, 
        mutation_rate=0.02
    )
    
    print("\nAll experiments completed!")

if __name__ == "__main__":
    main()