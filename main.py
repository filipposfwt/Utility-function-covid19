import subprocess
import os

def main():
    user_path = input("Enter path to your COVID cases CSV file (e.g. /Users/.../cases_GR.csv): ").strip()

    if not os.path.exists(user_path):
        print(f"File not found: {user_path}")
        return

    print("Running initial feature analysis...")
    subprocess.run(["python", "scripts/01_initial_analysis_plots_all_features.py", user_path])

    print("Selecting features with LSTM...")
    subprocess.run(["python", "scripts/02_selected_features_LSTM.py", user_path])

    print("Running utility curve polynomial fit...")
    subprocess.run(["python", "scripts/03_utility_function_calculation.py"])

    print("Running DE + DQN curve fitting...")
    subprocess.run(["python", "scripts/04_DEA_DRL-curve_fitness.py"])

    print("Pipeline complete. Check /data/outputs for results.")

if __name__ == "__main__":
    main()
