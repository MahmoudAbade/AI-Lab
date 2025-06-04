import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import copy
from typing import List, Tuple, Dict
from dataclasses import dataclass
import itertools
from collections import deque
import heapq

@dataclass
class CVRPInstance:
    """CVRP Instance data structure"""
    name: str
    dimension: int
    capacity: int
    coordinates: np.ndarray
    demands: np.ndarray
    depot: int = 0
    
    def distance_matrix(self):
        """Calculate Euclidean distance matrix"""
        n = self.dimension
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i, j] = np.sqrt((self.coordinates[i, 0] - self.coordinates[j, 0])**2 + 
                                   (self.coordinates[i, 1] - self.coordinates[j, 1])**2)
        return dist

@dataclass 
class Solution:
    """Solution representation"""
    routes: List[List[int]]
    cost: float
    feasible: bool = True
    
    def copy(self):
        return Solution([route.copy() for route in self.routes], self.cost, self.feasible)

class CVRPReader:
    """Read CVRP instances from standard format"""
    
    @staticmethod
    def read_instance(filename: str) -> CVRPInstance:
        """Read CVRP instance from file"""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        name = ""
        dimension = 0
        capacity = 0
        coordinates = []
        demands = []
        
        section = None
        for line in lines:
            line = line.strip()
            if line.startswith("NAME"):
                name = line.split(":")[1].strip()
            elif line.startswith("DIMENSION"):
                dimension = int(line.split(":")[1])
            elif line.startswith("CAPACITY"):
                capacity = int(line.split(":")[1])
            elif line == "NODE_COORD_SECTION":
                section = "COORD"
            elif line == "DEMAND_SECTION":
                section = "DEMAND"
            elif line == "DEPOT_SECTION":
                section = "DEPOT"
            elif line == "EOF":
                break
            elif section == "COORD" and line:
                parts = line.split()
                if len(parts) >= 3:
                    coordinates.append([float(parts[1]), float(parts[2])])
            elif section == "DEMAND" and line:
                parts = line.split()
                if len(parts) >= 2:
                    demands.append(int(parts[1]))
        
        return CVRPInstance(
            name=name,
            dimension=dimension,
            capacity=capacity,
            coordinates=np.array(coordinates),
            demands=np.array(demands)
        )

class CVRPSolver:
    """Base class for CVRP solvers"""
    
    def __init__(self, instance: CVRPInstance):
        self.instance = instance
        self.distance_matrix = instance.distance_matrix()
        self.best_solution = None
        self.best_cost = float('inf')
        
    def calculate_route_cost(self, route: List[int]) -> float:
        """Calculate cost of a single route"""
        if not route:
            return 0.0
        
        cost = self.distance_matrix[0][route[0]]  # depot to first
        for i in range(len(route) - 1):
            cost += self.distance_matrix[route[i]][route[i + 1]]
        cost += self.distance_matrix[route[-1]][0]  # last to depot
        return cost
    
    def calculate_solution_cost(self, solution: Solution) -> float:
        """Calculate total cost of solution"""
        return sum(self.calculate_route_cost(route) for route in solution.routes)
    
    def is_feasible(self, route: List[int]) -> bool:
        """Check if route is feasible (capacity constraint)"""
        total_demand = sum(self.instance.demands[i] for i in route)
        return total_demand <= self.instance.capacity
    
    def create_random_solution(self) -> Solution:
        """Create random feasible solution"""
        customers = list(range(1, self.instance.dimension))
        random.shuffle(customers)
        
        routes = []
        current_route = []
        current_load = 0
        
        for customer in customers:
            demand = self.instance.demands[customer]
            if current_load + demand <= self.instance.capacity:
                current_route.append(customer)
                current_load += demand
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [customer]
                current_load = demand
        
        if current_route:
            routes.append(current_route)
        
        solution = Solution(routes, 0)
        solution.cost = self.calculate_solution_cost(solution)
        return solution

# 1. Multi-stage Constructive Heuristic
class GreedyConstructiveHeuristic(CVRPSolver):
    """Multi-stage greedy constructive heuristic"""
    
    def solve(self) -> Solution:
        """
        Multi-stage approach:
        1. Cluster customers by geographic proximity
        2. Apply savings algorithm within clusters
        3. Optimize individual routes
        """
        start_time = time.time()
        
        # Stage 1: Geographic clustering
        clusters = self._geographic_clustering()
        
        # Stage 2: Apply savings algorithm within clusters
        routes = []
        for cluster in clusters:
            if cluster:
                cluster_routes = self._savings_algorithm(cluster)
                routes.extend(cluster_routes)
        
        # Stage 3: Local optimization
        routes = self._optimize_routes(routes)
        
        solution = Solution(routes, 0)
        solution.cost = self.calculate_solution_cost(solution)
        
        self.execution_time = time.time() - start_time
        return solution
    
    def _geographic_clustering(self, k: int = 4) -> List[List[int]]:
        """Simple geographic clustering using k-means-like approach"""
        customers = list(range(1, self.instance.dimension))
        if len(customers) <= k:
            return [customers]
        
        # Initialize cluster centers
        centers = random.sample(customers, k)
        clusters = [[] for _ in range(k)]
        
        # Assign customers to nearest center
        for customer in customers:
            min_dist = float('inf')
            best_cluster = 0
            for i, center in enumerate(centers):
                dist = self.distance_matrix[customer][center]
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = i
            clusters[best_cluster].append(customer)
        
        return [cluster for cluster in clusters if cluster]
    
    def _savings_algorithm(self, customers: List[int]) -> List[List[int]]:
        """Classic Clarke-Wright savings algorithm"""
        if not customers:
            return []
        
        # Calculate savings
        savings = []
        for i in range(len(customers)):
            for j in range(i + 1, len(customers)):
                c1, c2 = customers[i], customers[j]
                saving = (self.distance_matrix[0][c1] + self.distance_matrix[0][c2] - 
                         self.distance_matrix[c1][c2])
                savings.append((saving, c1, c2))
        
        # Sort by savings (descending)
        savings.sort(reverse=True)
        
        # Initialize routes (each customer in its own route)
        routes = [[c] for c in customers]
        route_of_customer = {c: i for i, c in enumerate(customers)}
        
        # Merge routes based on savings
        for saving, c1, c2 in savings:
            r1, r2 = route_of_customer[c1], route_of_customer[c2]
            if r1 != r2:  # Different routes
                route1, route2 = routes[r1], routes[r2]
                
                # Check if merge is feasible
                total_demand = (sum(self.instance.demands[c] for c in route1) + 
                              sum(self.instance.demands[c] for c in route2))
                
                if total_demand <= self.instance.capacity:
                    # Determine merge order
                    if c1 == route1[0] and c2 == route2[-1]:
                        merged = route2 + route1
                    elif c1 == route1[-1] and c2 == route2[0]:
                        merged = route1 + route2
                    elif c1 == route1[0] and c2 == route2[0]:
                        merged = list(reversed(route2)) + route1
                    elif c1 == route1[-1] and c2 == route2[-1]:
                        merged = route1 + list(reversed(route2))
                    else:
                        continue  # Can't merge in middle
                    
                    # Update routes
                    routes[r1] = merged
                    routes[r2] = []
                    for c in merged:
                        route_of_customer[c] = r1
        
        return [route for route in routes if route]
    
    def _optimize_routes(self, routes: List[List[int]]) -> List[List[int]]:
        """Apply 2-opt to each route"""
        optimized = []
        for route in routes:
            optimized.append(self._two_opt(route))
        return optimized
    
    def _two_opt(self, route: List[int]) -> List[int]:
        """2-opt local search for TSP"""
        if len(route) < 3:
            return route
        
        best_route = route[:]
        best_cost = self.calculate_route_cost(best_route)
        improved = True
        
        while improved:
            improved = False
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route)):
                    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                    new_cost = self.calculate_route_cost(new_route)
                    if new_cost < best_cost:
                        best_route = new_route
                        best_cost = new_cost
                        route = new_route
                        improved = True
        
        return best_route

# 2. Tabu Search
class TabuSearch(CVRPSolver):
    """Tabu Search for CVRP"""
    
    def __init__(self, instance: CVRPInstance, tabu_tenure: int = 10, max_iterations: int = 1000):
        super().__init__(instance)
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.tabu_list = {}
    
    def solve(self, initial_solution: Solution = None) -> Solution:
        if initial_solution is None:
            current = self.create_random_solution()
        else:
            current = initial_solution.copy()
        
        best = current.copy()
        self.tabu_list = {}
        
        for iteration in range(self.max_iterations):
            neighbors = self._generate_neighbors(current)
            
            # Find best non-tabu neighbor
            best_neighbor = None
            best_neighbor_cost = float('inf')
            
            for neighbor in neighbors:
                if not self._is_tabu(neighbor, iteration):
                    cost = self.calculate_solution_cost(neighbor)
                    if cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = cost
            
            if best_neighbor is None:
                break
            
            # Update tabu list
            self._update_tabu_list(current, best_neighbor, iteration)
            current = best_neighbor
            current.cost = best_neighbor_cost
            
            # Update best solution
            if current.cost < best.cost:
                best = current.copy()
        
        return best
    
    def _generate_neighbors(self, solution: Solution) -> List[Solution]:
        """Generate neighbors using various operators"""
        neighbors = []
        
        # Intra-route 2-opt
        for i, route in enumerate(solution.routes):
            if len(route) > 2:
                for j in range(len(route) - 1):
                    for k in range(j + 2, len(route)):
                        new_route = route[:j+1] + route[j+1:k+1][::-1] + route[k+1:]
                        new_routes = solution.routes[:]
                        new_routes[i] = new_route
                        if self.is_feasible(new_route):
                            neighbors.append(Solution(new_routes, 0))
        
        # Inter-route relocate
        for i, route1 in enumerate(solution.routes):
            for j, route2 in enumerate(solution.routes):
                if i != j:
                    for k, customer in enumerate(route1):
                        # Remove customer from route1
                        new_route1 = route1[:k] + route1[k+1:]
                        # Try inserting in all positions of route2
                        for pos in range(len(route2) + 1):
                            new_route2 = route2[:pos] + [customer] + route2[pos:]
                            if self.is_feasible(new_route2):
                                new_routes = solution.routes[:]
                                new_routes[i] = new_route1
                                new_routes[j] = new_route2
                                neighbors.append(Solution(new_routes, 0))
        
        return neighbors[:50]  # Limit number of neighbors
    
    def _is_tabu(self, solution: Solution, iteration: int) -> bool:
        """Check if solution is tabu"""
        solution_key = self._solution_key(solution)
        return solution_key in self.tabu_list and self.tabu_list[solution_key] > iteration
    
    def _update_tabu_list(self, old_solution: Solution, new_solution: Solution, iteration: int):
        """Update tabu list"""
        old_key = self._solution_key(old_solution)
        self.tabu_list[old_key] = iteration + self.tabu_tenure
    
    def _solution_key(self, solution: Solution) -> str:
        """Create a key for solution (for tabu list)"""
        return str(sorted([tuple(sorted(route)) for route in solution.routes]))

# 3. Simulated Annealing
class SimulatedAnnealing(CVRPSolver):
    """Simulated Annealing for CVRP"""
    
    def __init__(self, instance: CVRPInstance, initial_temp: float = 1000, 
                 cooling_rate: float = 0.95, min_temp: float = 0.1):
        super().__init__(instance)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
    
    def solve(self, initial_solution: Solution = None) -> Solution:
        if initial_solution is None:
            current = self.create_random_solution()
        else:
            current = initial_solution.copy()
        
        best = current.copy()
        temperature = self.initial_temp
        
        while temperature > self.min_temp:
            for _ in range(100):  # Inner loop
                neighbor = self._get_neighbor(current)
                
                delta = neighbor.cost - current.cost
                
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current = neighbor
                    
                    if current.cost < best.cost:
                        best = current.copy()
            
            temperature *= self.cooling_rate
        
        return best
    
    def _get_neighbor(self, solution: Solution) -> Solution:
        """Get a random neighbor"""
        neighbor = solution.copy()
        
        if random.random() < 0.5:
            # Intra-route 2-opt
            if neighbor.routes:
                route_idx = random.randint(0, len(neighbor.routes) - 1)
                route = neighbor.routes[route_idx]
                if len(route) > 2:
                    i = random.randint(0, len(route) - 2)
                    j = random.randint(i + 2, len(route))
                    neighbor.routes[route_idx] = route[:i+1] + route[i+1:j][::-1] + route[j:]
        else:
            # Inter-route relocate
            if len(neighbor.routes) > 1:
                route1_idx = random.randint(0, len(neighbor.routes) - 1)
                route2_idx = random.randint(0, len(neighbor.routes) - 1)
                while route2_idx == route1_idx:
                    route2_idx = random.randint(0, len(neighbor.routes) - 1)
                
                route1 = neighbor.routes[route1_idx]
                route2 = neighbor.routes[route2_idx]
                
                if route1:
                    customer_idx = random.randint(0, len(route1) - 1)
                    customer = route1[customer_idx]
                    
                    # Remove from route1
                    new_route1 = route1[:customer_idx] + route1[customer_idx+1:]
                    
                    # Add to route2
                    pos = random.randint(0, len(route2))
                    new_route2 = route2[:pos] + [customer] + route2[pos:]
                    
                    if self.is_feasible(new_route2):
                        neighbor.routes[route1_idx] = new_route1
                        neighbor.routes[route2_idx] = new_route2
        
        neighbor.cost = self.calculate_solution_cost(neighbor)
        return neighbor

# 4. Ant Colony Optimization
class AntColonyOptimization(CVRPSolver):
    """Discrete Ant Colony Optimization for CVRP"""
    
    def __init__(self, instance: CVRPInstance, n_ants: int = 20, n_iterations: int = 100,
                 alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1, q0: float = 0.9):
        super().__init__(instance)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # pheromone importance
        self.beta = beta    # heuristic importance
        self.rho = rho      # evaporation rate
        self.q0 = q0        # exploitation vs exploration
        
        # Initialize pheromone matrix
        self.pheromone = np.ones((instance.dimension, instance.dimension)) * 0.1
        
        # Heuristic information (inverse of distance)
        self.eta = np.zeros((instance.dimension, instance.dimension))
        for i in range(instance.dimension):
            for j in range(instance.dimension):
                if i != j and self.distance_matrix[i][j] > 0:
                    self.eta[i][j] = 1.0 / self.distance_matrix[i][j]
    
    def solve(self) -> Solution:
        best_solution = None
        best_cost = float('inf')
        
        for iteration in range(self.n_iterations):
            # Generate solutions with ants
            solutions = []
            for ant in range(self.n_ants):
                solution = self._construct_solution()
                solutions.append(solution)
                
                if solution.cost < best_cost:
                    best_solution = solution.copy()
                    best_cost = solution.cost
            
            # Update pheromones
            self._update_pheromones(solutions)
        
        return best_solution
    
    def _construct_solution(self) -> Solution:
        """Construct solution using ant's probabilistic rules"""
        unvisited = set(range(1, self.instance.dimension))
        routes = []
        
        while unvisited:
            route = []
            current_load = 0
            current_node = 0  # Start from depot
            
            while unvisited:
                # Find customers that can be visited (capacity constraint)
                feasible = []
                for customer in unvisited:
                    if current_load + self.instance.demands[customer] <= self.instance.capacity:
                        feasible.append(customer)
                
                if not feasible:
                    break
                
                # Select next customer using ACO rules
                next_customer = self._select_next_customer(current_node, feasible)
                
                route.append(next_customer)
                current_load += self.instance.demands[next_customer]
                current_node = next_customer
                unvisited.remove(next_customer)
            
            if route:
                routes.append(route)
        
        solution = Solution(routes, 0)
        solution.cost = self.calculate_solution_cost(solution)
        return solution
    
    def _select_next_customer(self, current: int, feasible: List[int]) -> int:
        """Select next customer using ACO probability rules"""
        if not feasible:
            return None
        
        if random.random() < self.q0:
            # Exploitation: select best
            best_value = -1
            best_customer = feasible[0]
            
            for customer in feasible:
                value = (self.pheromone[current][customer] ** self.alpha * 
                        self.eta[current][customer] ** self.beta)
                if value > best_value:
                    best_value = value
                    best_customer = customer
            
            return best_customer
        else:
            # Exploration: probabilistic selection
            probabilities = []
            total = 0
            
            for customer in feasible:
                prob = (self.pheromone[current][customer] ** self.alpha * 
                       self.eta[current][customer] ** self.beta)
                probabilities.append(prob)
                total += prob
            
            if total == 0:
                return random.choice(feasible)
            
            probabilities = [p / total for p in probabilities]
            
            # Roulette wheel selection
            r = random.random()
            cumulative = 0
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    return feasible[i]
            
            return feasible[-1]
    
    def _update_pheromones(self, solutions: List[Solution]):
        """Update pheromone trails"""
        # Evaporation
        self.pheromone *= (1 - self.rho)
        
        # Deposit pheromones
        for solution in solutions:
            if solution.cost > 0:
                deposit = 1.0 / solution.cost
                
                for route in solution.routes:
                    if route:
                        # Depot to first customer
                        self.pheromone[0][route[0]] += deposit
                        
                        # Between customers
                        for i in range(len(route) - 1):
                            self.pheromone[route[i]][route[i+1]] += deposit
                        
                        # Last customer to depot
                        self.pheromone[route[-1]][0] += deposit

# 5. Genetic Algorithm with Island Model
class GeneticAlgorithm(CVRPSolver):
    """Genetic Algorithm with Island Model for CVRP"""
    
    def __init__(self, instance: CVRPInstance, population_size: int = 100, 
                 n_generations: int = 500, mutation_rate: float = 0.1,
                 n_islands: int = 4, migration_rate: int = 50):
        super().__init__(instance)
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.n_islands = n_islands
        self.migration_rate = migration_rate
    
    def solve(self) -> Solution:
        # Initialize islands
        islands = []
        for _ in range(self.n_islands):
            population = [self.create_random_solution() for _ in range(self.population_size // self.n_islands)]
            islands.append(population)
        
        best_solution = None
        best_cost = float('inf')
        
        for generation in range(self.n_generations):
            # Evolve each island
            for i, island in enumerate(islands):
                islands[i] = self._evolve_population(island)
                
                # Update best solution
                for individual in islands[i]:
                    if individual.cost < best_cost:
                        best_solution = individual.copy()
                        best_cost = individual.cost
            
            # Migration between islands
            if generation % self.migration_rate == 0:
                self._migrate(islands)
        
        return best_solution
    
    def _evolve_population(self, population: List[Solution]) -> List[Solution]:
        """Evolve a single population"""
        new_population = []
        
        # Elitism: keep best solutions
        population.sort(key=lambda x: x.cost)
        elite_size = len(population) // 10
        new_population.extend(population[:elite_size])
        
        # Generate offspring
        while len(new_population) < len(population):
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            offspring = self._crossover(parent1, parent2)
            
            if random.random() < self.mutation_rate:
                offspring = self._mutate(offspring)
            
            offspring.cost = self.calculate_solution_cost(offspring)
            new_population.append(offspring)
        
        return new_population
    
    def _tournament_selection(self, population: List[Solution], tournament_size: int = 3) -> Solution:
        """Tournament selection"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return min(tournament, key=lambda x: x.cost)
    
    def _crossover(self, parent1: Solution, parent2: Solution) -> Solution:
        """Order crossover adapted for CVRP"""
        all_customers1 = [c for route in parent1.routes for c in route]
        all_customers2 = [c for route in parent2.routes for c in route]
        
        if not all_customers1:
            return parent2.copy()
        
        # Select crossover points
        start = random.randint(0, len(all_customers1) - 1)
        end = random.randint(start, len(all_customers1) - 1)
        
        # Create offspring
        offspring_customers = [-1] * len(all_customers1)
        offspring_customers[start:end+1] = all_customers1[start:end+1]
        
        # Fill remaining positions from parent2
        remaining = [c for c in all_customers2 if c not in offspring_customers]
        pos = 0
        for i in range(len(offspring_customers)):
            if offspring_customers[i] == -1:
                if pos < len(remaining):
                    offspring_customers[i] = remaining[pos]
                    pos += 1
        
        # Convert back to routes
        routes = self._customers_to_routes(offspring_customers)
        return Solution(routes, 0)
    
    def _mutate(self, solution: Solution) -> Solution:
        """Mutation operator"""
        mutated = solution.copy()
        
        if random.random() < 0.5:
            # Intra-route mutation (2-opt)
            if mutated.routes:
                route_idx = random.randint(0, len(mutated.routes) - 1)
                route = mutated.routes[route_idx]
                if len(route) > 2:
                    i = random.randint(0, len(route) - 2)
                    j = random.randint(i + 1, len(route) - 1)
                    mutated.routes[route_idx] = route[:i] + route[i:j+1][::-1] + route[j+1:]
        else:
            # Inter-route mutation (relocate)
            if len(mutated.routes) > 1:
                route1_idx = random.randint(0, len(mutated.routes) - 1)
                route2_idx = random.randint(0, len(mutated.routes) - 1)
                while route2_idx == route1_idx and len(mutated.routes) > 1:
                    route2_idx = random.randint(0, len(mutated.routes) - 1)
                
                route1 = mutated.routes[route1_idx]
                if route1:
                    customer_idx = random.randint(0, len(route1) - 1)
                    customer = route1.pop(customer_idx)
                    
                    route2 = mutated.routes[route2_idx]
                    pos = random.randint(0, len(route2))
                    route2.insert(pos, customer)
                    
                    if not self.is_feasible(route2):
                        # Revert if infeasible
                        route2.pop(pos)
                        route1.insert(customer_idx, customer)
        
        return mutated
    
    def _customers_to_routes(self, customers: List[int]) -> List[List[int]]:
        """Convert customer list to feasible routes"""
        routes = []
        current_route = []
        current_load = 0
        
        for customer in customers:
            if customer > 0:  # Valid customer
                demand = self.instance.demands[customer]
                if current_load + demand <= self.instance.capacity:
                    current_route.append(customer)
                    current_load += demand
                else:
                    if current_route:
                        routes.append(current_route)
                    current_route = [customer]
                    current_load = demand
        
        if current_route:
            routes.append(current_route)
        
        return routes
    
    def _migrate(self, islands: List[List[Solution]]):
        """Migration between islands"""
        for i in range(self.n_islands):
            source = islands[i]
            target_idx = (i + 1) % self.n_islands
            target = islands[target_idx]
           
            # Select best individuals from source
            source.sort(key=lambda x: x.cost)
            migrants = source[:2]  # Top 2 individuals
            
            # Replace worst individuals in target
            target.sort(key=lambda x: x.cost)
            target[-len(migrants):] = [m.copy() for m in migrants]

# 6. Adaptive Large Neighborhood Search
class AdaptiveLargeNeighborhoodSearch(CVRPSolver):
   """Adaptive Large Neighborhood Search for CVRP"""
   
   def __init__(self, instance: CVRPInstance, max_iterations: int = 1000):
       super().__init__(instance)
       self.max_iterations = max_iterations
       self.destroy_operators = [
           self._random_removal,
           self._worst_removal,
           self._related_removal
       ]
       self.repair_operators = [
           self._greedy_insertion,
           self._regret_insertion
       ]
       
       # Adaptive weights
       self.destroy_weights = [1.0] * len(self.destroy_operators)
       self.repair_weights = [1.0] * len(self.repair_operators)
       self.operator_scores = {'destroy': [0] * len(self.destroy_operators),
                              'repair': [0] * len(self.repair_operators)}
       self.operator_uses = {'destroy': [0] * len(self.destroy_operators),
                            'repair': [0] * len(self.repair_operators)}
   
   def solve(self, initial_solution: Solution = None) -> Solution:
       if initial_solution is None:
           current = self.create_random_solution()
       else:
           current = initial_solution.copy()
       
       best = current.copy()
       temperature = 100.0
       cooling_rate = 0.99995
       
       for iteration in range(self.max_iterations):
           # Select operators based on adaptive weights
           destroy_idx = self._roulette_wheel_selection(self.destroy_weights)
           repair_idx = self._roulette_wheel_selection(self.repair_weights)
           
           # Apply destroy and repair
           destroyed = self.destroy_operators[destroy_idx](current.copy())
           repaired = self.repair_operators[repair_idx](destroyed)
           
           # Acceptance criterion (simulated annealing)
           delta = repaired.cost - current.cost
           score = 0
           
           if repaired.cost < best.cost:
               best = repaired.copy()
               current = repaired
               score = 3  # New best solution
           elif delta < 0:
               current = repaired
               score = 2  # Improving solution
           elif random.random() < math.exp(-delta / temperature):
               current = repaired
               score = 1  # Accepted worse solution
           
           # Update operator scores
           self.operator_scores['destroy'][destroy_idx] += score
           self.operator_scores['repair'][repair_idx] += score
           self.operator_uses['destroy'][destroy_idx] += 1
           self.operator_uses['repair'][repair_idx] += 1
           
           # Update weights periodically
           if iteration % 100 == 0:
               self._update_weights()
           
           temperature *= cooling_rate
       
       return best
   
   def _roulette_wheel_selection(self, weights: List[float]) -> int:
       """Roulette wheel selection based on weights"""
       total = sum(weights)
       if total == 0:
           return random.randint(0, len(weights) - 1)
       
       r = random.uniform(0, total)
       cumulative = 0
       for i, weight in enumerate(weights):
           cumulative += weight
           if r <= cumulative:
               return i
       return len(weights) - 1
   
   def _random_removal(self, solution: Solution, removal_size: int = None) -> Solution:
       """Randomly remove customers"""
       if removal_size is None:
           removal_size = random.randint(1, min(10, len([c for route in solution.routes for c in route]) // 4))
       
       all_customers = [(i, j, customer) for i, route in enumerate(solution.routes) 
                       for j, customer in enumerate(route)]
       
       if len(all_customers) <= removal_size:
           return solution
       
       to_remove = random.sample(all_customers, removal_size)
       to_remove.sort(key=lambda x: (x[0], x[1]), reverse=True)  # Remove from back to front
       
       removed_customers = []
       for route_idx, customer_idx, customer in to_remove:
           if route_idx < len(solution.routes) and customer_idx < len(solution.routes[route_idx]):
               removed_customers.append(solution.routes[route_idx].pop(customer_idx))
       
       # Clean empty routes
       solution.routes = [route for route in solution.routes if route]
       solution.cost = self.calculate_solution_cost(solution)
       
       # Store removed customers for repair
       solution.removed_customers = removed_customers
       return solution
   
   def _worst_removal(self, solution: Solution, removal_size: int = None) -> Solution:
       """Remove customers with highest removal cost"""
       if removal_size is None:
           removal_size = random.randint(1, min(10, len([c for route in solution.routes for c in route]) // 4))
       
       # Calculate removal costs
       removal_costs = []
       for i, route in enumerate(solution.routes):
           for j, customer in enumerate(route):
               cost = self._calculate_removal_cost(route, j)
               removal_costs.append((cost, i, j, customer))
       
       if len(removal_costs) <= removal_size:
           return solution
       
       # Sort by removal cost (descending) and remove worst
       removal_costs.sort(reverse=True)
       to_remove = removal_costs[:removal_size]
       to_remove.sort(key=lambda x: (x[1], x[2]), reverse=True)
       
       removed_customers = []
       for cost, route_idx, customer_idx, customer in to_remove:
           if route_idx < len(solution.routes) and customer_idx < len(solution.routes[route_idx]):
               removed_customers.append(solution.routes[route_idx].pop(customer_idx))
       
       solution.routes = [route for route in solution.routes if route]
       solution.cost = self.calculate_solution_cost(solution)
       solution.removed_customers = removed_customers
       return solution
   
   def _related_removal(self, solution: Solution, removal_size: int = None) -> Solution:
       """Remove related customers (close to each other)"""
       if removal_size is None:
           removal_size = random.randint(1, min(10, len([c for route in solution.routes for c in route]) // 4))
       
       all_customers = [customer for route in solution.routes for customer in route]
       if not all_customers:
           return solution
       
       # Select seed customer
       seed = random.choice(all_customers)
       
       # Find customers closest to seed
       distances = [(self.distance_matrix[seed][customer], customer) for customer in all_customers if customer != seed]
       distances.sort()
       
       to_remove = [seed] + [customer for _, customer in distances[:removal_size-1]]
       
       # Remove customers
       removed_customers = []
       for i in range(len(solution.routes)-1, -1, -1):
           for j in range(len(solution.routes[i])-1, -1, -1):
               if solution.routes[i][j] in to_remove:
                   removed_customers.append(solution.routes[i].pop(j))
       
       solution.routes = [route for route in solution.routes if route]
       solution.cost = self.calculate_solution_cost(solution)
       solution.removed_customers = removed_customers
       return solution
   
   def _calculate_removal_cost(self, route: List[int], position: int) -> float:
       """Calculate cost of removing customer at position"""
       if len(route) <= 1:
           return 0
       
       customer = route[position]
       
       # Cost before removal
       cost_before = 0
       if position == 0:
           cost_before = self.distance_matrix[0][customer] + self.distance_matrix[customer][route[1] if len(route) > 1 else 0]
       elif position == len(route) - 1:
           cost_before = self.distance_matrix[route[position-1]][customer] + self.distance_matrix[customer][0]
       else:
           cost_before = (self.distance_matrix[route[position-1]][customer] + 
                         self.distance_matrix[customer][route[position+1]])
       
       # Cost after removal
       cost_after = 0
       if len(route) > 1:
           if position == 0 and len(route) > 1:
               cost_after = self.distance_matrix[0][route[1]]
           elif position == len(route) - 1 and len(route) > 1:
               cost_after = self.distance_matrix[route[position-1]][0]
           elif 0 < position < len(route) - 1:
               cost_after = self.distance_matrix[route[position-1]][route[position+1]]
       
       return cost_before - cost_after
   
   def _greedy_insertion(self, solution: Solution) -> Solution:
       """Greedily insert removed customers"""
       if not hasattr(solution, 'removed_customers'):
           return solution
       
       for customer in solution.removed_customers:
           best_cost = float('inf')
           best_position = None
           
           # Try inserting in existing routes
           for i, route in enumerate(solution.routes):
               for j in range(len(route) + 1):
                   # Check capacity
                   route_demand = sum(self.instance.demands[c] for c in route) + self.instance.demands[customer]
                   if route_demand <= self.instance.capacity:
                       # Calculate insertion cost
                       cost = self._calculate_insertion_cost(route, j, customer)
                       if cost < best_cost:
                           best_cost = cost
                           best_position = (i, j)
           
           # Create new route if no feasible insertion found
           if best_position is None:
               solution.routes.append([customer])
           else:
               route_idx, pos = best_position
               solution.routes[route_idx].insert(pos, customer)
       
       solution.cost = self.calculate_solution_cost(solution)
       delattr(solution, 'removed_customers')
       return solution
   
   def _regret_insertion(self, solution: Solution) -> Solution:
       """Insert customers using regret-based strategy"""
       if not hasattr(solution, 'removed_customers'):
           return solution
       
       remaining = solution.removed_customers[:]
       
       while remaining:
           best_regret = -1
           best_customer = None
           best_position = None
           
           for customer in remaining:
               # Find two best insertion positions
               costs = []
               positions = []
               
               for i, route in enumerate(solution.routes):
                   for j in range(len(route) + 1):
                       route_demand = sum(self.instance.demands[c] for c in route) + self.instance.demands[customer]
                       if route_demand <= self.instance.capacity:
                           cost = self._calculate_insertion_cost(route, j, customer)
                           costs.append(cost)
                           positions.append((i, j))
               
               # Add option to create new route
               costs.append(self.distance_matrix[0][customer] * 2)  # Round trip to depot
               positions.append((-1, -1))  # New route indicator
               
               if len(costs) >= 2:
                   costs_positions = list(zip(costs, positions))
                   costs_positions.sort()
                   
                   regret = costs_positions[1][0] - costs_positions[0][0]
                   if regret > best_regret:
                       best_regret = regret
                       best_customer = customer
                       best_position = costs_positions[0][1]
           
           if best_customer is not None:
               remaining.remove(best_customer)
               
               if best_position == (-1, -1):
                   solution.routes.append([best_customer])
               else:
                   route_idx, pos = best_position
                   solution.routes[route_idx].insert(pos, best_customer)
           else:
               break
       
       # Add any remaining customers to new routes
       for customer in remaining:
           solution.routes.append([customer])
       
       solution.cost = self.calculate_solution_cost(solution)
       delattr(solution, 'removed_customers')
       return solution
   
   def _calculate_insertion_cost(self, route: List[int], position: int, customer: int) -> float:
       """Calculate cost of inserting customer at position"""
       if not route:
           return self.distance_matrix[0][customer] * 2  # Round trip
       
       if position == 0:
           return (self.distance_matrix[0][customer] + self.distance_matrix[customer][route[0]] - 
                  self.distance_matrix[0][route[0]])
       elif position == len(route):
           return (self.distance_matrix[route[-1]][customer] + self.distance_matrix[customer][0] - 
                  self.distance_matrix[route[-1]][0])
       else:
           return (self.distance_matrix[route[position-1]][customer] + 
                  self.distance_matrix[customer][route[position]] - 
                  self.distance_matrix[route[position-1]][route[position]])
   
   def _update_weights(self):
       """Update operator weights based on performance"""
       for op_type in ['destroy', 'repair']:
           for i in range(len(self.operator_scores[op_type])):
               if self.operator_uses[op_type][i] > 0:
                   avg_score = self.operator_scores[op_type][i] / self.operator_uses[op_type][i]
                   if op_type == 'destroy':
                       self.destroy_weights[i] = 0.8 * self.destroy_weights[i] + 0.2 * avg_score
                   else:
                       self.repair_weights[i] = 0.8 * self.repair_weights[i] + 0.2 * avg_score
               
               # Reset scores and uses
               self.operator_scores[op_type][i] = 0
               self.operator_uses[op_type][i] = 0

# 7. Branch and Bound with Limited Discrepancy Search
class BranchAndBound(CVRPSolver):
   """Branch and Bound with Limited Discrepancy Search for CVRP"""
   
   def __init__(self, instance: CVRPInstance, max_discrepancies: int = 3, time_limit: float = 300):
       super().__init__(instance)
       self.max_discrepancies = max_discrepancies
       self.time_limit = time_limit
       self.start_time = None
       self.nodes_explored = 0
   
   def solve(self) -> Solution:
       self.start_time = time.time()
       self.best_solution = None
       self.best_cost = float('inf')
       
       # Get initial upper bound
       greedy = GreedyConstructiveHeuristic(self.instance)
       initial = greedy.solve()
       self.best_solution = initial
       self.best_cost = initial.cost
       
       # Limited Discrepancy Search with increasing discrepancy limit
       for max_disc in range(self.max_discrepancies + 1):
           if time.time() - self.start_time > self.time_limit:
               break
           self._lds([], set(range(1, self.instance.dimension)), 0, 0, 0, max_disc)
       
       return self.best_solution if self.best_solution else initial
   
   def _lds(self, current_route: List[int], unvisited: set, current_load: int, 
           current_cost: float, discrepancies: int, max_discrepancies: int):
       """Limited Discrepancy Search"""
       
       if time.time() - self.start_time > self.time_limit:
           return
       
       self.nodes_explored += 1
       
       # Pruning
       if current_cost >= self.best_cost:
           return
       
       if not unvisited:
           # Complete solution found
           if current_route:
               cost_to_depot = self.distance_matrix[current_route[-1]][0]
               total_cost = current_cost + cost_to_depot
               
               if total_cost < self.best_cost:
                   # Create complete solution (this is just one route)
                   # In practice, you'd need to handle multiple routes
                   solution = Solution([current_route], total_cost)
                   self.best_solution = solution
                   self.best_cost = total_cost
           return
       
       # Get candidates sorted by heuristic (nearest neighbor)
       candidates = []
       current_node = current_route[-1] if current_route else 0
       
       for customer in unvisited:
           if current_load + self.instance.demands[customer] <= self.instance.capacity:
               distance = self.distance_matrix[current_node][customer]
               candidates.append((distance, customer))
       
       candidates.sort()  # Sort by distance (heuristic choice)
       
       for i, (distance, customer) in enumerate(candidates):
           new_route = current_route + [customer]
           new_unvisited = unvisited - {customer}
           new_load = current_load + self.instance.demands[customer]
           new_cost = current_cost + distance
           
           # Decide on discrepancy
           is_discrepancy = (i > 0)  # Not choosing the best (first) option
           
           if is_discrepancy and discrepancies >= max_discrepancies:
               continue
           
           new_discrepancies = discrepancies + (1 if is_discrepancy else 0)
           
           self._lds(new_route, new_unvisited, new_load, new_cost, 
                    new_discrepancies, max_discrepancies)

# Ackley Function Optimization (for testing algorithms)
class AckleyFunction:
   """Ackley function for testing optimization algorithms"""
   
   def __init__(self, dimensions: int = 10, a: float = 20, b: float = 0.2, c: float = 2*math.pi):
       self.dimensions = dimensions
       self.a = a
       self.b = b
       self.c = c
       self.bounds = (-32.768, 32.768)
   
   def evaluate(self, x: np.ndarray) -> float:
       """Evaluate Ackley function"""
       d = len(x)
       sum1 = np.sum(x**2)
       sum2 = np.sum(np.cos(self.c * x))
       
       term1 = -self.a * np.exp(-self.b * np.sqrt(sum1 / d))
       term2 = -np.exp(sum2 / d)
       
       return term1 + term2 + self.a + np.exp(1)
   
   def test_algorithm(self, algorithm_name: str, n_runs: int = 10):
       """Test optimization algorithm on Ackley function"""
       print(f"\nTesting {algorithm_name} on Ackley function:")
       
       results = []
       times = []
       
       for run in range(n_runs):
           start_time = time.time()
           
           # Simple random search for demonstration
           best_x = None
           best_value = float('inf')
           
           for _ in range(1000):  # 1000 evaluations
               x = np.random.uniform(self.bounds[0], self.bounds[1], self.dimensions)
               value = self.evaluate(x)
               
               if value < best_value:
                   best_value = value
                   best_x = x
           
           end_time = time.time()
           
           results.append(best_value)
           times.append(end_time - start_time)
       
       print(f"  Average result: {np.mean(results):.6f}")
       print(f"  Best result: {min(results):.6f}")
       print(f"  Std deviation: {np.std(results):.6f}")
       print(f"  Average time: {np.mean(times):.4f}s")
       
       return results, times

# Visualization and Analysis Tools
class CVRPVisualizer:
    """Visualization tools for CVRP solutions"""
    
    @staticmethod
    def plot_instance(instance: CVRPInstance, title: str = None, save_path: str = None):
        """Plot CVRP instance (depot and customers only)"""
        plt.figure(figsize=(10, 8))
        
        if title is None:
            title = f"{instance.name} (n={instance.dimension-1}, Q={instance.capacity})"
        
        # Plot depot
        depot_coords = instance.coordinates[0]
        plt.plot(depot_coords[0], depot_coords[1], 'bs', markersize=12, label='depot')
        
        # Plot customers
        for i in range(1, instance.dimension):
            plt.plot(instance.coordinates[i, 0], instance.coordinates[i, 1], 'ro', markersize=8, label='customers' if i == 1 else "")
            # Add customer number and demand
            plt.annotate(f'{i}', 
                        (instance.coordinates[i, 0], instance.coordinates[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
            
        # Add demand annotation for one customer as example (like in your image)
        if instance.dimension > 5:  # Show demand for customer 5 as example
            customer_idx = min(5, instance.dimension - 1)
            plt.annotate(f'Customer:{customer_idx}, Demand:{instance.demands[customer_idx]}', 
                        xy=(instance.coordinates[customer_idx, 0], instance.coordinates[customer_idx, 1]),
                        xycoords='data',
                        xytext=(20, 20), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))
        
        plt.title(title)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved instance plot: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_solution(instance: CVRPInstance, solution: Solution, title: str = "CVRP Solution", save_path: str = None):
        """Plot CVRP solution with routes"""
        plt.figure(figsize=(12, 8))
        
        # Plot depot
        depot_coords = instance.coordinates[0]
        plt.plot(depot_coords[0], depot_coords[1], 'ks', markersize=15, label='Depot')
        
        # Plot customers
        for i in range(1, instance.dimension):
            plt.plot(instance.coordinates[i, 0], instance.coordinates[i, 1], 'bo', markersize=8)
            plt.annotate(str(i), (instance.coordinates[i, 0], instance.coordinates[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot routes
        colors = plt.cm.Set1(np.linspace(0, 1, len(solution.routes)))
        
        for route_idx, route in enumerate(solution.routes):
            if not route:
                continue
                
            color = colors[route_idx % len(colors)]
            
            # Plot route
            x_coords = [depot_coords[0]]
            y_coords = [depot_coords[1]]
            
            for customer in route:
                x_coords.append(instance.coordinates[customer, 0])
                y_coords.append(instance.coordinates[customer, 1])
            
            x_coords.append(depot_coords[0])
            y_coords.append(depot_coords[1])
            
            plt.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7,
                    label=f'Route {route_idx + 1}')
        
        plt.title(f"{title}\nCost: {solution.cost:.2f}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved solution plot: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_convergence(costs: List[float], title: str = "Algorithm Convergence", save_path: str = None):
        """Plot algorithm convergence"""
        plt.figure(figsize=(10, 6))
        plt.plot(costs, linewidth=2)
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Best Cost")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved convergence plot: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_comparison_chart(results: Dict, instance_name: str, save_path: str = None):
        """Plot performance comparison chart for algorithms"""
        plt.figure(figsize=(12, 8))
        
        algorithms = list(results.keys())
        avg_costs = [results[alg]['avg_cost'] for alg in algorithms]
        best_costs = [results[alg]['best_cost'] for alg in algorithms]
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        plt.bar(x - width/2, avg_costs, width, label='Average Cost', alpha=0.7)
        plt.bar(x + width/2, best_costs, width, label='Best Cost', alpha=0.7)
        
        plt.xlabel('Algorithm')
        plt.ylabel('Cost')
        plt.title(f'Performance Comparison - {instance_name}')
        plt.xticks(x, algorithms, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved comparison chart: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_route_details(instance: CVRPInstance, solution: Solution, save_path: str = None):
        """Plot solution with detailed route information"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))  
        
        # Left plot: Solution visualization
        depot_coords = instance.coordinates[0]
        ax1.plot(depot_coords[0], depot_coords[1], 'ks', markersize=15, label='Depot')
        
        for i in range(1, instance.dimension):
            ax1.plot(instance.coordinates[i, 0], instance.coordinates[i, 1], 'bo', markersize=8)
            ax1.annotate(f'{i}\n({instance.demands[i]})', 
                        (instance.coordinates[i, 0], instance.coordinates[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8,
                        ha='left', va='bottom')
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(solution.routes)))
        
        for route_idx, route in enumerate(solution.routes):
            if not route:
                continue
                
            color = colors[route_idx % len(colors)]
            
            x_coords = [depot_coords[0]]
            y_coords = [depot_coords[1]]
            
            for customer in route:
                x_coords.append(instance.coordinates[customer, 0])
                y_coords.append(instance.coordinates[customer, 1])
            
            x_coords.append(depot_coords[0])
            y_coords.append(depot_coords[1])
            
            ax1.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7,
                    label=f'Route {route_idx + 1}')
        
        ax1.set_title(f"{instance.name}\nTotal Cost: {solution.cost:.2f}")
        ax1.set_xlabel("X Coordinate")
        ax1.set_ylabel("Y Coordinate")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Route details table
        ax2.axis('tight')
        ax2.axis('off')
        
        table_data = []
        headers = ['Route', 'Customers', 'Load', 'Cost']
        
        def format_customer_list(customers, max_per_line=8):
            """Format customer list into multiple lines if needed"""
            if len(customers) <= max_per_line:
                return ' → '.join([str(c) for c in customers])
            else:
                lines = []
                for i in range(0, len(customers), max_per_line):
                    chunk = customers[i:i+max_per_line]
                    lines.append(' → '.join([str(c) for c in chunk]))
                return '\n'.join(lines)
        
        for route_idx, route in enumerate(solution.routes):
            if not route:
                continue
            
            route_load = sum(instance.demands[customer] for customer in route)
            route_cost = 0
            
            # Calculate route cost
            current = 0  # depot
            for customer in route:
                route_cost += np.sqrt((instance.coordinates[current, 0] - instance.coordinates[customer, 0])**2 + 
                                    (instance.coordinates[current, 1] - instance.coordinates[customer, 1])**2)
                current = customer
            # Return to depot
            route_cost += np.sqrt((instance.coordinates[current, 0] - instance.coordinates[0, 0])**2 + 
                                (instance.coordinates[current, 1] - instance.coordinates[0, 1])**2)
            
            customers_str = format_customer_list(route)
            table_data.append([f'Route {route_idx + 1}', customers_str, f'{route_load}/{instance.capacity}', f'{route_cost:.2f}'])
        
        table = ax2.table(cellText=table_data, colLabels=headers, cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)  # Slightly smaller font
        table.scale(1.4, 2.0)  # Increased height scaling for multi-line cells
        
        # Color table headers
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Adjust column widths
        cellDict = table.get_celld()
        for i in range(len(table_data) + 1):  # +1 for header
            cellDict[(i, 0)].set_width(0.15)  # Route column - narrower
            cellDict[(i, 1)].set_width(0.50)  # Customers column - wider
            cellDict[(i, 2)].set_width(0.15)  # Load column
            cellDict[(i, 3)].set_width(0.20)  # Cost column
            
            # Set text alignment and wrapping for customer column
            if i > 0:  # Skip header
                cellDict[(i, 1)].set_text_props(wrap=True, verticalalignment='center')
        
        ax2.set_title("Route Details", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved route details plot: {save_path}")
        
        plt.show()

# Performance Analysis
class PerformanceAnalyzer:
    """Analyze and compare algorithm performance"""
    
    @staticmethod
    def compare_algorithms(instance: CVRPInstance, algorithms: Dict, n_runs: int = 5) -> Dict:
        """Compare multiple algorithms"""
        results = {}
        
        for alg_name, alg_class in algorithms.items():
            print(f"\nRunning {alg_name}...")
            
            costs = []
            times = []
            solutions = []  # Store all solutions
            
            for run in range(n_runs):
                solver = alg_class(instance)
                
                start_time = time.time()
                solution = solver.solve()
                end_time = time.time()
                
                costs.append(solution.cost)
                times.append(end_time - start_time)
                solutions.append(solution)  # Store solution
                
                print(f"  Run {run + 1}: Cost = {solution.cost:.2f}, Time = {times[-1]:.2f}s")
            
            results[alg_name] = {
                'costs': costs,
                'times': times,
                'solutions': solutions,  # Include solutions
                'avg_cost': np.mean(costs),
                'best_cost': min(costs),
                'std_cost': np.std(costs),
                'avg_time': np.mean(times)
            }
        
        return results
    
    @staticmethod
    def get_best_solution_from_algorithm(instance: CVRPInstance, algorithm_class, algorithm_results) -> Solution:
        """Get the best solution from the algorithm that has the best average performance"""
        # Find the index of the best cost in this algorithm's runs
        best_cost_index = algorithm_results['costs'].index(algorithm_results['best_cost'])
        
        # Return the corresponding solution
        if 'solutions' in algorithm_results:
            return algorithm_results['solutions'][best_cost_index]
        else:
            # Fallback: re-run the algorithm to get a solution
            solver = algorithm_class(instance)
            return solver.solve()
    
    @staticmethod
    def print_comparison_table(results: Dict):
        """Print comparison table"""
        print("\n" + "="*90)
        print(f"{'Algorithm':<25} {'Avg Cost':<12} {'Best Cost':<12} {'Std Dev':<12} {'Avg Time':<12} {'Rank':<8}")
        print("="*90)
        
        # Sort algorithms by average cost for ranking
        sorted_algs = sorted(results.items(), key=lambda x: x[1]['avg_cost'])
        
        for rank, (alg_name, data) in enumerate(sorted_algs, 1):
            print(f"{alg_name:<25} {data['avg_cost']:<12.2f} {data['best_cost']:<12.2f} "
                  f"{data['std_cost']:<12.2f} {data['avg_time']:<12.2f} {rank:<8}")
        
        print("="*90)
        print(f"Best algorithm by average cost: {sorted_algs[0][0]} (avg: {sorted_algs[0][1]['avg_cost']:.2f})")

# Main Execution Framework
def main():
   """Main execution function"""
   
   # Example: Create a small test instance
   test_instance = CVRPInstance(
       name="Test Instance",
       dimension=5,
       capacity=10,
       coordinates=np.array([[0, 0], [2, 3], [5, 1], [1, 6], [8, 4]]),
       demands=np.array([0, 3, 4, 2, 3])
   )
   
   print("CVRP Solver Framework")
   print("=" * 50)
   
   # Test Ackley function first
   ackley = AckleyFunction()
   ackley.test_algorithm("Random Search")
   
   # Test algorithms on CVRP
   algorithms = {
       'Greedy Constructive': GreedyConstructiveHeuristic,
       'Tabu Search': TabuSearch,
       'Simulated Annealing': SimulatedAnnealing,
       'Ant Colony Optimization': AntColonyOptimization,
       'Genetic Algorithm': GeneticAlgorithm,
       'ALNS': AdaptiveLargeNeighborhoodSearch
   }
   
   # Run comparison
   results = PerformanceAnalyzer.compare_algorithms(test_instance, algorithms, n_runs=3)
   PerformanceAnalyzer.print_comparison_table(results)
   
   # Visualize best solution
   best_algorithm = min(results.keys(), key=lambda x: results[x]['best_cost'])
   print(f"\nBest algorithm: {best_algorithm}")
   
   solver = algorithms[best_algorithm](test_instance)
   solution = solver.solve()
   
   CVRPVisualizer.plot_solution(test_instance, solution, f"Best Solution ({best_algorithm})")

if __name__ == "__main__":
   main()