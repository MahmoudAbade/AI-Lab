import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import sys
import os
import threading

# Ensure we're in the right directory
def set_correct_directory():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Change to that directory
    os.chdir(script_dir)
    return script_dir

current_dir = set_correct_directory()

def run_command_thread(command):
    """Run command in a separate thread to keep GUI responsive"""
    # Disable buttons during execution
    for button in buttons:
        button.config(state=tk.DISABLED)
    
    status_bar.config(text="Running experiment...")
    
    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True,
        shell=True
    )
    
    # Real-time output display
    for line in process.stdout:
        output_text.insert(tk.END, line)
        output_text.see(tk.END)
        output_text.update()
    
    process.wait()
    output_text.insert(tk.END, f"\nProcess completed with return code: {process.returncode}\n")
    output_text.insert(tk.END, "=" * 50 + "\n")
    output_text.see(tk.END)
    
    # Re-enable buttons
    for button in buttons:
        button.config(state=tk.NORMAL)
    
    if process.returncode == 0:
        status_bar.config(text="Experiment completed successfully!")
    else:
        status_bar.config(text=f"Experiment completed with errors (code {process.returncode})")
    
    # Extract results directory from output
    output = output_text.get("1.0", tk.END)
    for line in output.split('\n'):
        if "Results will be saved in:" in line:
            result_dir = line.split("Results will be saved in:")[1].strip()
            last_result_var.set(result_dir)
            break

def run_command(command):
    """Wrapper to run command in a thread"""
    # Clear output text
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, f"Running: {command}\n\n")
    output_text.update()
    
    # Start thread
    thread = threading.Thread(target=run_command_thread, args=(command,))
    thread.daemon = True
    thread.start()

def run_all():
    num_runs = runs_var.get()
    debug_mode = debug_var.get()
    parallel = not no_parallel_var.get()
    
    cmd = f"python main.py --all --runs {num_runs}"
    if debug_mode:
        cmd += " --debug"
    if not parallel:
        cmd += " --no-parallel"
        
    run_command(cmd)

def run_dtsp():
    num_runs = runs_var.get()
    debug_mode = debug_var.get()
    parallel = not no_parallel_var.get()
    
    cmd = f"python main.py --dtsp --runs {num_runs}"
    if debug_mode:
        cmd += " --debug"
    if not parallel:
        cmd += " --no-parallel"
        
    run_command(cmd)

def run_bin_packing():
    num_runs = runs_var.get()
    debug_mode = debug_var.get()
    parallel = not no_parallel_var.get()
    
    cmd = f"python main.py --bin-packing --runs {num_runs}"
    if debug_mode:
        cmd += " --debug"
    if not parallel:
        cmd += " --no-parallel"
        
    run_command(cmd)

def run_baldwin():
    num_runs = runs_var.get()
    debug_mode = debug_var.get()
    parallel = not no_parallel_var.get()
    
    cmd = f"python main.py --baldwin --runs {num_runs}"
    if debug_mode:
        cmd += " --debug"
    if not parallel:
        cmd += " --no-parallel"
        
    run_command(cmd)

def open_results_folder():
    result_dir = last_result_var.get()
    if result_dir and os.path.exists(result_dir):
        # Open the folder in file explorer
        os.startfile(result_dir)
    else:
        folder = filedialog.askdirectory(initialdir=os.path.join(current_dir, "results"),
                                         title="Select Results Folder")
        if folder:
            os.startfile(folder)

def clear_output():
    output_text.delete(1.0, tk.END)

def show_about():
    messagebox.showinfo(
        "About Genetic Algorithm Experiments",
        "Genetic Algorithm Experiments\n\n"
        "This application runs various genetic algorithm experiments:\n"
        "- Double Traveling Salesman Problem (DTSP)\n"
        "- Bin Packing Problem\n"
        "- Baldwin Effect Simulation\n\n"
        "Results are saved in the 'results' directory with a timestamp.\n\n"
        "Created for the Artificial Intelligence Laboratory Assignment."
    )

# Create main window
root = tk.Tk()
root.title("Genetic Algorithm Experiments")
root.geometry("900x700")
root.minsize(800, 600)

# Variables
runs_var = tk.IntVar(value=3)
debug_var = tk.BooleanVar(value=False)
no_parallel_var = tk.BooleanVar(value=False)
last_result_var = tk.StringVar(value="")

# Create menu
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Open Results Folder", command=open_results_folder)
file_menu.add_command(label="Clear Output", command=clear_output)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

help_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Help", menu=help_menu)
help_menu.add_command(label="About", command=show_about)

# Main frame
main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)

# Create top frame for configuration
config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
config_frame.pack(fill=tk.X, padx=5, pady=5)

# Configuration options
ttk.Label(config_frame, text="Number of Runs:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
ttk.Spinbox(config_frame, from_=1, to=10, textvariable=runs_var, width=5).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

ttk.Checkbutton(config_frame, text="Debug Mode (Faster Execution)", variable=debug_var).grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
ttk.Checkbutton(config_frame, text="Disable Parallel Processing", variable=no_parallel_var).grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

# Create frame for buttons
button_frame = ttk.LabelFrame(main_frame, text="Experiments", padding="10")
button_frame.pack(fill=tk.X, padx=5, pady=5)

# Create buttons for different experiments
buttons = []
run_all_btn = ttk.Button(button_frame, text="Run All Experiments", command=run_all, width=20)
run_all_btn.grid(row=0, column=0, padx=5, pady=5)
buttons.append(run_all_btn)

run_dtsp_btn = ttk.Button(button_frame, text="DTSP Only", command=run_dtsp, width=20)
run_dtsp_btn.grid(row=0, column=1, padx=5, pady=5)
buttons.append(run_dtsp_btn)

run_bin_btn = ttk.Button(button_frame, text="Bin Packing Only", command=run_bin_packing, width=20)
run_bin_btn.grid(row=0, column=2, padx=5, pady=5)
buttons.append(run_bin_btn)

run_baldwin_btn = ttk.Button(button_frame, text="Baldwin Effect Only", command=run_baldwin, width=20)
run_baldwin_btn.grid(row=0, column=3, padx=5, pady=5)
buttons.append(run_baldwin_btn)

# Add results button
results_frame = ttk.Frame(main_frame)
results_frame.pack(fill=tk.X, padx=5, pady=5)

ttk.Label(results_frame, text="Last Results Folder:").pack(side=tk.LEFT, padx=5)
ttk.Entry(results_frame, textvariable=last_result_var, width=50, state="readonly").pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
results_btn = ttk.Button(results_frame, text="Open Folder", command=open_results_folder)
results_btn.pack(side=tk.RIGHT, padx=5)
buttons.append(results_btn)

# Create output text area
output_frame = ttk.LabelFrame(main_frame, text="Output", padding="10")
output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

output_text = tk.Text(output_frame, wrap=tk.WORD, width=80, height=20)
output_text.pack(fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=output_text.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
output_text.config(yscrollcommand=scrollbar.set)

# Status bar
status_bar = ttk.Label(root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# Initial instructions
output_text.insert(tk.END, "Welcome to the Genetic Algorithm Experiments application!\n\n")
output_text.insert(tk.END, "This application runs the following experiments:\n")
output_text.insert(tk.END, "1. Double Traveling Salesman Problem (DTSP)\n")
output_text.insert(tk.END, "2. Bin Packing Problem\n")
output_text.insert(tk.END, "3. Baldwin Effect Simulation\n\n")
output_text.insert(tk.END, "Select an experiment to run from the buttons above.\n")
output_text.insert(tk.END, "Results will be saved in the 'results' directory with a timestamp.\n\n")
output_text.insert(tk.END, "Current working directory: " + current_dir + "\n")
output_text.insert(tk.END, "=" * 50 + "\n")

# Start the main loop
root.mainloop()