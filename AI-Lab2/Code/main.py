import os
import argparse
import sys
import time
import datetime
import multiprocessing as mp
from functools import partial

# Import the main TSP module
from dtsp_ga import (
    load_cities_from_file, run_genetic_algorithm, plot_results, plot_double_tour,
    compare_mutation_policies, compare_fitness_policies, compare_diversity_methods,
    parameter_sensitivity_analysis, run_baldwin_effect_experiment, calculate_path_length
)

# Import the bin packing module
from bin_packing_ga import (
    generate_random_instance, bin_packing_ga, plot_bin_packing_results,
    visualize_bin_packing_solution, compare_bin_packing_policies
)

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def create_output_dir(base_dir="results"):
    """Create output directory with timestamp"""
    timestamp = get_timestamp()
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def solve_dtsp_problem(problem_info, output_dir, runs=3, debug=False):
    """Solve a single DTSP problem"""
    tsp_file, generations = problem_info
    
    # Extract just the base name without path or extension
    tsp_name = os.path.splitext(os.path.basename(tsp_file))[0]
    print(f"\n{'-'*50}")
    print(f"Solving DTSP for {tsp_name}...")
    print(f"{'-'*50}")
    
    # Create problem-specific directory
    problem_dir = os.path.join(output_dir, tsp_name)
    os.makedirs(problem_dir, exist_ok=True)
    
    # Save current directory
    original_dir = os.getcwd()
    
    try:
        # Load cities before changing directory
        cities = load_cities_from_file(tsp_file)
        
        if debug and len(cities) > 30:
            # For debug mode, limit to 30 cities
            import random
            cities = random.sample(cities, 30)
        
        # Change to problem directory for output
        os.chdir(problem_dir)
        
        # Run basic GA with fewer generations in debug mode
        actual_generations = 20 if debug else generations
        actual_runs = 1 if debug else runs
        actual_population = 50 if debug else 200
        
        print(f"\nRunning basic genetic algorithm with {actual_generations} generations and {actual_runs} runs...")
        
        best_individual, best_history, avg_history = run_genetic_algorithm(
            cities, population_size=actual_population, generations=actual_generations,
            elite_size=int(actual_population*0.1), tournament_size=5,
            crossover_rate=0.8, mutation_rate=0.05,
            early_stop_generations=10 if debug else 50
        )
        
        # Save results
        length1 = calculate_path_length(best_individual.chromosome1, cities)
        length2 = calculate_path_length(best_individual.chromosome2, cities)
        longer_length = max(length1, length2)
        
        with open("basic_results.txt", "w") as f:
            f.write(f"Problem: {tsp_name}\n")
            f.write(f"Path 1 length: {length1:.2f}\n")
            f.write(f"Path 2 length: {length2:.2f}\n")
            f.write(f"Longer path length (objective): {longer_length:.2f}\n")
        
        # Plot results
        plot_results(best_history, avg_history, f"{tsp_name} - Fitness Evolution")
        plot_double_tour(cities, best_individual.chromosome1, best_individual.chromosome2, 
                        f"{tsp_name} - Double TSP Solution")
        
        # If not in debug mode, run more detailed experiments
        if not debug:
            # Compare mutation policies (with fewer generations and runs)
            print("\nComparing mutation policies...")
            mutation_policies = {
                "Fixed": ("fixed", {}),
                "Adaptive": ("adaptive", {}),
                "Hypermutation": ("hypermutation", {"threshold": 0.7, "high_rate": 0.3}),
                "Age-based": ("age_based", {"max_age": 20, "min_rate": 0.01, "max_rate": 0.2})
            }
            
            mutation_results = compare_mutation_policies(
                cities, mutation_policies, 
                generations=100, runs=actual_runs, 
                population_size=actual_population,
                early_stop_generations=50
            )
            
            # Compare fitness policies
            print("\nComparing fitness policies...")
            fitness_policies = {
                "Standard": ("standard", {}),
                "Novelty": ("novelty", {"k_neighbors": 5}),
                "Age-based": ("age_based", {"max_age": 20})
            }
            
            fitness_results = compare_fitness_policies(
                cities, fitness_policies, 
                generations=100, runs=actual_runs,
                population_size=actual_population,
                early_stop_generations=50
            )
            
            # Compare diversity methods
            print("\nComparing diversity methods...")
            diversity_methods = {
                "None": (None, {}),
                "Niching": ("niching", {"fitness_radius": 0.2}),
                "Speciation": ("speciation", {"similarity_threshold": 0.3, "target_species": 5})
            }
            
            diversity_results = compare_diversity_methods(
                cities, diversity_methods, 
                generations=100, runs=actual_runs,
                population_size=actual_population,
                early_stop_generations=50
            )
            
            # Parameter sensitivity analysis (with reduced parameter sets)
            print("\nPerforming parameter sensitivity analysis...")
            mutation_rates = [0.01, 0.1, 0.3]  # Reduced for speed
            mutation_sensitivity = parameter_sensitivity_analysis(
                cities, "mutation_rate", mutation_rates,
                generations=50
            )
            
            fitness_radii = [0.1, 0.3, 0.5]  # Reduced for speed
            radius_sensitivity = parameter_sensitivity_analysis(
                cities, "fitness_radius", fitness_radii,
                generations=50
            )
            
            similarity_thresholds = [0.1, 0.3, 0.5]  # Reduced for speed
            threshold_sensitivity = parameter_sensitivity_analysis(
                cities, "similarity_threshold", similarity_thresholds,
                generations=50
            )
        
    except Exception as e:
        print(f"Error processing {tsp_file}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Always go back to original directory
        os.chdir(original_dir)
    
    return tsp_name, "Completed"

def analyze_bin_packing_instance(instance_info, output_dir, runs=3, debug=False):
    """Analyze a single bin packing instance"""
    instance_id, num_items, min_size, max_size, bin_capacity = instance_info
    
    # Reduce problem size in debug mode
    if debug:
        num_items = min(num_items, 30)
    
    print(f"\nAnalyzing bin packing instance {instance_id}...")
    
    # Create instance directory
    instance_dir = os.path.join(output_dir, "bin_packing", f"instance_{instance_id}")
    os.makedirs(instance_dir, exist_ok=True)
    
    # Save current directory
    original_dir = os.getcwd()
    
    try:
        # Change to instance directory
        os.chdir(instance_dir)
        
        # Generate instance
        items, bin_capacity = generate_random_instance(num_items=num_items, min_size=min_size, max_size=max_size, bin_capacity=bin_capacity)
        
        # Run with fewer generations in debug mode
        actual_generations = 20 if debug else 300
        actual_runs = 1 if debug else runs
        actual_population = 50 if debug else 100
        
        # Run basic GA
        print("\nRunning basic genetic algorithm for bin packing...")
        best_individual, best_fitness_history, avg_fitness_history, best_bins_history = bin_packing_ga(
            items, bin_capacity, 
            population_size=actual_population, 
            generations=actual_generations,
            elite_size=int(actual_population*0.1), 
            tournament_size=5, 
            crossover_rate=0.8, 
            mutation_rate=0.05,
            early_stop_generations=10 if debug else 50
        )
        
        # Save results
        with open("basic_results.txt", "w") as f:
            f.write(f"Instance: instance_{instance_id}\n")
            f.write(f"Number of items: {len(items)}\n")
            f.write(f"Bin capacity: {bin_capacity}\n")
            f.write(f"Number of bins: {len(best_individual.bins)}\n")
            f.write(f"Fitness: {best_individual.fitness:.6f}\n")
        
        # Plot results
        plot_bin_packing_results(best_fitness_history, avg_fitness_history, best_bins_history)
        visualize_bin_packing_solution(best_individual, bin_capacity)
        
        # If not in debug mode, run more detailed experiments
        if not debug:
            # Compare mutation policies
            print("\nComparing mutation policies...")
            mutation_policies = {
                "Fixed": ("mutation", "fixed", {}),
                "Adaptive": ("mutation", "adaptive", {}),
                "Hypermutation": ("mutation", "hypermutation", {"threshold": 0.7, "high_rate": 0.3}),
                "Age-based": ("mutation", "age_based", {"max_age": 20, "min_rate": 0.01, "max_rate": 0.2})
            }
            
            compare_bin_packing_policies(
                items, bin_capacity, mutation_policies, 
                generations=100, runs=actual_runs,
                population_size=actual_population,
                elite_size=int(actual_population*0.1),
                early_stop_generations=30
            )
            
            # Compare fitness policies
            print("\nComparing fitness policies...")
            fitness_policies = {
                "Standard": ("fitness", "standard", {}),
                "Novelty": ("fitness", "novelty", {"k_neighbors": 5}),
                "Age-based": ("fitness", "age_based", {"max_age": 20})
            }
            
            compare_bin_packing_policies(
                items, bin_capacity, fitness_policies, 
                generations=100, runs=actual_runs,
                population_size=actual_population,
                elite_size=int(actual_population*0.1),
                early_stop_generations=30
            )
            
            # Compare diversity methods
            print("\nComparing diversity methods...")
            diversity_methods = {
                "None": ("diversity", None, {}),
                "Niching": ("diversity", "niching", {"fitness_radius": 0.2}),
                "Speciation": ("diversity", "speciation", {"similarity_threshold": 0.3, "target_species": 5})
            }
            
            compare_bin_packing_policies(
                items, bin_capacity, diversity_methods, 
                generations=100, runs=actual_runs,
                population_size=actual_population,
                elite_size=int(actual_population*0.1),
                early_stop_generations=30
            )
        
    except Exception as e:
        print(f"Error in bin packing instance {instance_id}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Go back to original directory
        os.chdir(original_dir)
    
    return instance_id, "Completed"

def run_baldwin_effect_experiment_wrapper(output_dir, debug=False):
    """Run Baldwin Effect experiment"""
    print(f"\n{'-'*50}")
    print(f"Running Baldwin Effect Experiment...")
    print(f"{'-'*50}")
    
    # Create Baldwin directory
    baldwin_dir = os.path.join(output_dir, "baldwin_effect")
    os.makedirs(baldwin_dir, exist_ok=True)
    
    # Save current directory
    original_dir = os.getcwd()
    
    try:
        # Change working directory
        os.chdir(baldwin_dir)
        
        # Use fewer parameters in debug mode
        if debug:
            target_length = 10
            population_size = 100
            generations = 10
            learning_attempts = 50
        else:
            target_length = 20
            population_size = 500
            generations = 50
            learning_attempts = 500
        
        # Run the experiment
        results = run_baldwin_effect_experiment(
            target_length=target_length, 
            population_size=population_size, 
            generations=generations, 
            learning_attempts=learning_attempts, 
            crossover_rate=0.8, 
            mutation_rate=0.02
        )
        
        # Save results
        with open("baldwin_results.txt", "w") as f:
            f.write("Baldwin Effect Experiment Results\n")
            f.write("--------------------------------\n")
            f.write("Generation | Mismatches | Correct Positions | Bits Learned\n")
            
            for i in range(len(results['mismatches'])):
                f.write(f"{i:10d} | {results['mismatches'][i]:10.2f} | {results['correct_positions'][i]:17.2f} | {results['learned_bits'][i]:11.2f}\n")
            
            # Check if Baldwin Effect was observed
            initial_mismatches = results['mismatches'][0]
            final_mismatches = results['mismatches'][-1]
            
            initial_correct = results['correct_positions'][0]
            final_correct = results['correct_positions'][-1]
            
            f.write("\nAnalysis:\n")
            f.write(f"Initial mismatches: {initial_mismatches:.2f}\n")
            f.write(f"Final mismatches: {final_mismatches:.2f}\n")
            f.write(f"Change in mismatches: {initial_mismatches - final_mismatches:.2f}\n\n")
            
            f.write(f"Initial correct positions: {initial_correct:.2f}\n")
            f.write(f"Final correct positions: {final_correct:.2f}\n")
            f.write(f"Change in correct positions: {final_correct - initial_correct:.2f}\n\n")
            
            if final_correct > initial_correct and final_mismatches < initial_mismatches:
                f.write("Baldwin Effect was observed: Learning helped evolution find better solutions.\n")
            else:
                f.write("Baldwin Effect was not clearly observed in this experiment.\n")
    
    except Exception as e:
        print(f"Error in Baldwin effect experiment: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Go back to original directory
        os.chdir(original_dir)
    
    return "baldwin", "Completed"

def main():
    parser = argparse.ArgumentParser(description='Run DTSP and Bin Packing Genetic Algorithms')
    parser.add_argument('--dtsp', action='store_true', help='Run DTSP experiments')
    parser.add_argument('--bin-packing', action='store_true', help='Run Bin Packing experiments')
    parser.add_argument('--baldwin', action='store_true', help='Run Baldwin Effect experiment')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs for each experiment')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with fewer generations and runs')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    
    args = parser.parse_args()
    
    # Default to debug mode for faster execution
    if not any([args.dtsp, args.bin_packing, args.baldwin, args.all]):
        args.all = True
        args.debug = True
        print("No specific experiments selected. Running all in debug mode.")
    
    # Create output directory
    output_dir = create_output_dir()
    print(f"Results will be saved in: {output_dir}")
    
    # Define DTSP problems with shorter generations
    tsp_files = [
        ('all/eil51.tsp', 200),
        ('all/st70.tsp', 200),
        ('all/pr76.tsp', 200),
        ('all/kroA100.tsp', 200),
    ]
    
    # Define bin packing problems (smaller instances for speed)
    bin_packing_instances = [
        (1, 30, 10, 50, 100),
        (2, 50, 10, 70, 100),
        (3, 70, 5, 30, 50),
    ]
    
    # Determine whether to use parallel processing
    use_parallel = not args.no_parallel and mp.cpu_count() > 1
    
    if args.all or args.dtsp:
        print("\nRunning DTSP experiments...")
        
        if use_parallel:
            # Parallel execution
            with mp.Pool(processes=min(len(tsp_files), mp.cpu_count())) as pool:
                results = pool.map(
                    partial(solve_dtsp_problem, output_dir=output_dir, runs=args.runs, debug=args.debug),
                    tsp_files
                )
            print(f"DTSP results: {results}")
        else:
            # Sequential execution
            for tsp_file in tsp_files:
                solve_dtsp_problem(tsp_file, output_dir, runs=args.runs, debug=args.debug)
    
    if args.all or args.bin_packing:
        print("\nRunning Bin Packing experiments...")
        
        if use_parallel:
            # Parallel execution
            with mp.Pool(processes=min(len(bin_packing_instances), mp.cpu_count())) as pool:
                results = pool.map(
                    partial(analyze_bin_packing_instance, output_dir=output_dir, runs=args.runs, debug=args.debug),
                    bin_packing_instances
                )
            print(f"Bin packing results: {results}")
        else:
            # Sequential execution
            for instance in bin_packing_instances:
                analyze_bin_packing_instance(instance, output_dir, runs=args.runs, debug=args.debug)
    
    if args.all or args.baldwin:
        run_baldwin_effect_experiment_wrapper(output_dir, debug=args.debug)
    
    print(f"\nAll experiments completed. Results saved in: {output_dir}")

if __name__ == "__main__":
    main()