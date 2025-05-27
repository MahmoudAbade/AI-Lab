# How to Run Genetic Algorithm Code

This document provides instructions for running the genetic algorithm experiments for both the Double Traveling Salesman Problem (DTSP) and Bin Packing Problem, as well as the Baldwin Effect experiment.

## Running Using the Graphical User Interface

### Using the executable (.exe) file:

1. Double-click on `ga_gui.exe` to launch the GUI application
2. In the interface, you'll see the following options:

   - Number of Runs: Set the number of times each experiment will run (default: 3)
   - Debug Mode checkbox: Enable for faster execution with reduced population and generations
   - Disable Parallel Processing checkbox: Enable if you experience issues with parallel execution

3. Choose which experiment to run by clicking one of the buttons:
   - **Run All Experiments**: Runs DTSP, Bin Packing, and Baldwin Effect experiments
   - **DTSP Only**: Runs only the Double Traveling Salesman Problem experiments
   - **Bin Packing Only**: Runs only the Bin Packing Problem experiments
   - **Baldwin Effect Only**: Runs only the Baldwin Effect experiment

4. Results will be saved in a timestamped folder under the `results` directory
5. The results folder path will be displayed in the GUI
6. Use the "Open Folder" button to access the results after execution completes

## Running from Command Line

### Basic Command Line Usage:

1. Open a command prompt/terminal
2. Navigate to the directory containing the scripts
3. Run the main script using Python:

```
python main.py [options]
```

### Command Line Options:

- `--dtsp`: Run only DTSP experiments
- `--bin-packing`: Run only Bin Packing experiments
- `--baldwin`: Run only Baldwin Effect experiment
- `--all`: Run all experiments (default if no options specified)
- `--runs N`: Set number of runs (default: 3)
- `--debug`: Enable debug mode for faster execution
- `--no-parallel`: Disable parallel processing

### Examples:

Run all experiments in debug mode:
```
python main.py --all --debug
```

Run only DTSP with 5 runs:
```
python main.py --dtsp --runs 5
```

Run only Bin Packing with no parallel processing:
```
python main.py --bin-packing --no-parallel
```

Run only Baldwin Effect experiment:
```
python main.py --baldwin
```

## Module-Specific Command Line Usage

You can also run individual modules directly:

### DTSP Module:
```
python dtsp_ga.py
```

### Bin Packing Module:
```
python bin_packing_ga.py
```

## Input Files

- DTSP problems are located in the `all` directory with `.tsp` extension
- The default DTSP problems included are: eil51.tsp, st70.tsp, pr76.tsp, and kroA100.tsp

## Output Files and Visualization

All results will be saved in the `results` directory with a timestamp subfolder. Each experiment creates:

1. Text files with numerical results
2. PNG images with visualizations of:
   - Fitness evolution
   - Solution visualization
   - Comparison of different policies/methods
   - Parameter sensitivity analysis

## System Requirements

- Python 3.7 or later
- Required Python packages: numpy, pandas, matplotlib, seaborn

If running from Python source, install required packages using:
```
pip install numpy pandas matplotlib seaborn
```
