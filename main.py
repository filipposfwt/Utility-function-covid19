import subprocess
import os
import sys
from datetime import datetime

# ANSI escape code for red text
RED = '\033[91m'
RESET = '\033[0m'

HELP_TEXT = f"""
Usage:
  python main.py                # Starts the interactive pipeline
  python main.py --help | -h    # Show this help message

This pipeline runs COVID data analysis through 4 steps:
  [1] Initial feature analysis
  [2] LSTM-based feature selection
  [3] Utility function curve fitting
  [4] DE + DQN curve optimization

You will be prompted to select a CSV file and choose which steps to run.
"""

def main():
    # Handle --help / -h
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print(HELP_TEXT)
        return

    print("=== COVID Data Pipeline ===")

    # Repeatedly prompt for a valid file path
    while True:
        user_path = input("Enter path to your COVID cases CSV file (e.g. /Users/.../cases_GR.csv): ").strip()
        if os.path.exists(user_path) and user_path.lower().endswith(".csv"):
            break
        print(f"{RED}[ERROR]{RESET} File not found or not a CSV: {user_path}")

    # Create output directory timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step selection
    print("\nSelect which steps you want to run (comma-separated, e.g. 1,3,4). Press Enter to run all:")
    print("  [1] Initial feature analysis")
    print("  [2] LSTM-based feature selection")
    print("  [3] Utility function curve fitting")
    print("  [4] DE + DQN curve optimization")
    selected_steps_input = input("Steps to run: ").strip()

    if selected_steps_input:
        try:
            selected_steps = sorted(set(int(s) for s in selected_steps_input.split(',') if s.strip().isdigit()))
            if not all(1 <= step <= 4 for step in selected_steps):
                raise ValueError
        except ValueError:
            print(f"{RED}[ERROR]{RESET} Invalid step selection. Please enter numbers between 1 and 4.")
            return
    else:
        selected_steps = [1, 2, 3, 4]  # Run all by default

    # Step 1: Initial analysis
    if 1 in selected_steps:
        print("\n[STEP 1] Running initial feature analysis...")
        subprocess.run(["python", "scripts/01_initial_analysis_plots_all_features.py", user_path, timestamp])

        while True:
            user_input = input("\nInitial feature analysis is complete. Have you reviewed the output in 'data/01_initial_analysis'? Proceed to step 2? (Y/N): ").strip().lower()
            if user_input in ['y', 'yes']:
                break
            elif user_input in ['n', 'no']:
                print("Please review the output before proceeding.")
            else:
                print("Invalid input. Please enter Y or N.")
    else:
        print("[STEP 1] Skipped.")

    # Step 2: LSTM feature selection
    if 2 in selected_steps:
        print("\n[STEP 2] Selecting features with LSTM...")
        subprocess.run(["python", "scripts/02_selected_features_LSTM.py", user_path, timestamp])
    else:
        print("[STEP 2] Skipped.")

    # Step 3: Utility function curve fitting
    if 3 in selected_steps:
        print("\n[STEP 3] Running utility function polynomial fit...")
        subprocess.run(["python", "scripts/03_utility_function_calculation.py", user_path, timestamp])
    else:
        print("[STEP 3] Skipped.")

    # Step 4: DE + DQN optimization
    if 4 in selected_steps:
        print("\n[STEP 4] Running DE + DQN curve fitting...")
        subprocess.run(["python", "scripts/04_DEA_DRL-curve_fitness.py", timestamp])
    else:
        print("[STEP 4] Skipped.")

    print("\n✅ Pipeline complete. The results are saved in the 'output' directory.")

if __name__ == "__main__":
    main()
