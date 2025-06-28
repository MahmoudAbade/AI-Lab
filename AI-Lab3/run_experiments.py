from cvrp_solver import *
from download_instances import download_cvrp_instances
import os
import json

def run_full_experiments():
    """Run complete experiments as required by assignment"""
    
    # Download instances first if they don't exist
    print("Checking and downloading required instances...")
    download_cvrp_instances()
    
    # Create directories for saving plots
    os.makedirs("plots", exist_ok=True)
    os.makedirs("plots/instances", exist_ok=True)
    os.makedirs("plots/solutions", exist_ok=True)
    os.makedirs("plots/comparisons", exist_ok=True)
    
    # Load instances
    instance_files = [
        "instances/P-n16-k8.vrp",
        "instances/E-n22-k4.vrp", 
        "instances/A-n32-k5.vrp",
        "instances/B-n45-k6.vrp",
        "instances/A-n80-k10.vrp",
        "instances/X-n101-k25.vrp",
        "instances/M-n200-k17.vrp"
    ]
    
    algorithms = {
        'Greedy Constructive': GreedyConstructiveHeuristic,
        'Tabu Search': TabuSearch,
        'Simulated Annealing': SimulatedAnnealing,
        'Ant Colony Optimization': AntColonyOptimization,
        'Genetic Algorithm': GeneticAlgorithm,
        'ALNS': AdaptiveLargeNeighborhoodSearch,
        'Branch and Bound': BranchAndBound
    }
    
    # Test Ackley function first
    print("Testing algorithms on Ackley function...")
    ackley = AckleyFunction()
    for alg_name in algorithms.keys():
        ackley.test_algorithm(alg_name)
    
    # Run CVRP experiments
    all_results = {}
    
    for instance_file in instance_files:
        if not os.path.exists(instance_file):
            print(f"Instance file {instance_file} not found. Skipping...")
            continue
            
        print(f"\n{'='*60}")
        print(f"Testing instance: {instance_file}")
        print(f"{'='*60}")
        
        instance = CVRPReader.read_instance(instance_file)
        
        # Plot and save initial instance
        print(f"Plotting initial instance...")
        instance_plot_path = f"plots/instances/{instance.name}_instance.png"
        CVRPVisualizer.plot_instance(instance, save_path=instance_plot_path)
        
        # Run algorithms and collect all solutions for each algorithm
        results = PerformanceAnalyzer.compare_algorithms(instance, algorithms, n_runs=5)
        all_results[instance.name] = results
        PerformanceAnalyzer.print_comparison_table(results)
        
        # Find best algorithm based on AVERAGE COST (not best single cost)
        best_algorithm = min(results.keys(), key=lambda x: results[x]['avg_cost'])
        print(f"\nBest algorithm by average cost: {best_algorithm} (avg: {results[best_algorithm]['avg_cost']:.2f})")
        
        # Get the best solution from the best algorithm's runs
        best_solution = PerformanceAnalyzer.get_best_solution_from_algorithm(
            instance, algorithms[best_algorithm], results[best_algorithm]
        )
        
        # Save best solution plot (based on best average algorithm)
        print(f"Plotting best solution ({best_algorithm})...")
        best_solution_path = f"plots/solutions/{instance.name}_{best_algorithm.replace(' ', '_')}_BEST.png"
        CVRPVisualizer.plot_solution(instance, best_solution, 
                                   f"{instance.name} - Best Avg Algorithm ({best_algorithm})",
                                   save_path=best_solution_path)
        
        # Save detailed route plot for best algorithm
        route_details_path = f"plots/solutions/{instance.name}_{best_algorithm.replace(' ', '_')}_details.png"
        CVRPVisualizer.plot_route_details(instance, best_solution, save_path=route_details_path)
        
        # Generate comparison chart
        print(f"Plotting performance comparison...")
        comparison_chart_path = f"plots/comparisons/{instance.name}_comparison.png"
        CVRPVisualizer.plot_comparison_chart(results, instance.name, save_path=comparison_chart_path)
        
        # Generate solutions from all algorithms for comparison
        print(f"Generating solutions from all algorithms...")
        for alg_name, alg_class in algorithms.items():
            try:
                solver = alg_class(instance)
                sol = solver.solve()
                
                # Save individual solution plot
                alg_plot_path = f"plots/solutions/{instance.name}_{alg_name.replace(' ', '_')}.png"
                CVRPVisualizer.plot_solution(instance, sol, 
                                           f"{instance.name} - {alg_name}",
                                           save_path=alg_plot_path)
            except Exception as e:
                print(f"  ✗ Failed to generate solution for {alg_name}: {e}")

    
    # Save results
    with open('experiment_results.json', 'w') as f:
        json_results = {}
        for instance_name, instance_results in all_results.items():
            json_results[instance_name] = {}
            for alg_name, alg_results in instance_results.items():
                json_results[instance_name][alg_name] = {
                    'avg_cost': float(alg_results['avg_cost']),
                    'best_cost': float(alg_results['best_cost']),
                    'std_cost': float(alg_results['std_cost']),
                    'avg_time': float(alg_results['avg_time']),
                    'costs': [float(c) for c in alg_results['costs']],
                    'times': [float(t) for t in alg_results['times']]
                }
        
        json.dump(json_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETED!")
    print(f"{'='*60}")
    print(f"Results saved to: experiment_results.json")
    print(f"Instance plots saved in: plots/instances/")
    print(f"Solution plots saved in: plots/solutions/")
    print(f"Comparison plots saved in: plots/comparisons/")
    print(f"\nPlot Summary:")
    print(f"- Instance plots: {len(all_results)} files")
    print(f"- Best solution plots (by avg cost): {len(all_results)} files")
    print(f"- Detailed route plots: {len(all_results)} files")
    print(f"- Individual algorithm plots: {len(all_results) * len(algorithms)} files")
    print(f"- Comparison charts: {len(all_results)} files")
    
    return all_results

def download_cvrp_instances():
    """Download required CVRP instances"""
    import requests
    
    base_url = "http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/"
    
    instances = {
        # Beginner (n ≤ 30)
        "P-n16-k8.vrp": "P/P-n16-k8.vrp",
        "E-n22-k4.vrp": "E/E-n22-k4.vrp",
        
        # Intermediate (30 < n ≤ 80)  
        "A-n32-k5.vrp": "A/A-n32-k5.vrp",
        "B-n45-k6.vrp": "B/B-n45-k6.vrp", 
        "A-n80-k10.vrp": "A/A-n80-k10.vrp",
        
        # Advanced (n > 80)
        "X-n101-k25.vrp": "X/X-n101-k25.vrp",
        "M-n200-k17.vrp": "M/M-n200-k17.vrp"
    }
    
    os.makedirs("instances", exist_ok=True)
    
    for filename, path in instances.items():
        filepath = f"instances/{filename}"
        
        # Skip if file already exists
        if os.path.exists(filepath):
            print(f"  ✓ {filename} already exists")
            continue
            
        print(f"Downloading {filename}...")
        try:
            url = base_url + path
            response = requests.get(url)
            response.raise_for_status()
            
            with open(filepath, 'w') as f:
                f.write(response.text)
            print(f"  ✓ Downloaded {filename}")
            
        except Exception as e:
            print(f"  ✗ Failed to download {filename}: {e}")

if __name__ == "__main__":
    run_full_experiments()