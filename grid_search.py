import os
import itertools
import subprocess

# 1. DEFINE THE HYPERPARAMETER SEARCH SPACE
# Adjust these lists to control the values you want to test.
param_grid = {
    'gof_loss_weight': [0.01],
    'gof_angle': [10.0, 15.0, 20.0], # [10.0, 15.0, 20.0, 25.0]
    'gof_decay': [0.8, 0.9, 0.95]
}

# 2. SET OTHER FIXED PARAMETERS FOR THE TRAINING SCRIPT
# These parameters will remain constant across all runs.
# IMPORTANT: Change 'arch' to the correct path for your environment.
fixed_params = {
    'data': 'WebOfScience',  # Or 'nyt', 'WebOfScience', 'rcv1'
    'arch': '/your model path/', #<-- IMPORTANT: CHANGE THIS
    'batch': 16,
    'epoch': 30,     # Using a slightly lower epoch count for faster grid searching
    'lr': 1e-4,
    'device': 'cuda',
    'update': 1,
    'model': 'prompt',
    'layer': 1,
    'graph': 'GAT',
    'seed': 42,      # Use a fixed seed for reproducibility across runs
    'wandb': False    # Enable wandb logging if you have it set up
}

# --- SCRIPT LOGIC ---

# Generate all possible combinations of hyperparameters from the param_grid
keys, values = zip(*param_grid.items())
hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

total_runs = len(hyperparam_combinations)
print(f"--- Starting Grid Search for {total_runs} combinations ---")

# Iterate over each combination and run the training script
for i, params in enumerate(hyperparam_combinations):
    print(f"\n--- Running combination {i+1}/{total_runs} ---")
    print(f"--- Parameters: {params} ---")

    # Create a unique, descriptive name for this run
    run_name = (
        f"{fixed_params['data']}-"
        f"weight_{params['gof_loss_weight']}-"
        f"angle_{params['gof_angle']}-"
        f"decay_{params['gof_decay']}"
    )

    # Construct the command-line arguments list
    # Using a list is safer than building a single string
    command = ['python', 'train.py']
    command.append(f'--name={run_name}')

    # Add fixed parameters to the command
    for key, value in fixed_params.items():
        if isinstance(value, bool):
            if value:
                command.append(f'--{key}')
        else:
            command.append(f'--{key}={value}')

    # Add the current combination of hyperparameters
    for key, value in params.items():
        command.append(f'--{key}={value}')

    # Print the command that will be executed
    # Using subprocess.list2cmdline to safely show the command string
    print("Executing command:")
    print(subprocess.list2cmdline(command))

    # Execute the training script
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!!! Run failed for combination: {params}. Error: {e}")
    except KeyboardInterrupt:
        print("\n--- Grid search interrupted by user. ---")
        exit()

print(f"\n--- Grid Search Finished: Completed {total_runs} runs. ---")