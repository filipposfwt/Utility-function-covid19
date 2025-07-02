import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# ----------------------------
# Read input file from CLI argument
# ----------------------------
if len(sys.argv) < 3:
    print("Usage: python 01_initial_analysis_plots_all_features.py <path_to_csv> <datetime>")
    sys.exit(1)

file_path = sys.argv[1]
if not os.path.exists(file_path):
    print(f"Error: File does not exist at path {file_path}")
    sys.exit(1)

datetime = sys.argv[2]

# ----------------------------
# Define output directory
# ----------------------------
output_dir = "output/01_initial_analysis/" + datetime
if not os.path.exists(output_dir):
    print(f"Creating output directory: {output_dir}")   
    os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# Load and preprocess data
# ----------------------------
data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data = data.dropna(subset=['date']).copy()
data = data.sort_values(by='date')

# Exclude 'id' and 'date' columns from analysis for feature importance
feature_exclude = ['id', 'date']
data_cleaned = data.drop(columns=feature_exclude).dropna(subset=['new_deaths']).copy()
X = data_cleaned.drop(columns=['new_deaths'])
y = data_cleaned['new_deaths']
X = X.fillna(X.median())

# ----------------------------
# Plot: Total deaths over time
# ----------------------------
plt.figure(figsize=(14, 6))
plt.plot(data['date'], data['total_deaths'], color='b', label='Total Deaths')
plt.xlabel("Date")
plt.ylabel("Total Deaths")
plt.title("Total Deaths Over Time")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "total_deaths_over_time.png"))
plt.close()

# ----------------------------
# Plot: New deaths over time
# ----------------------------
plt.figure(figsize=(14, 6))
plt.plot(data['date'], data['new_deaths'], color='r', label='New Deaths')
plt.xlabel("Date")
plt.ylabel("New Deaths")
plt.title("New Deaths Over Time")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "new_deaths_over_time.png"))
plt.close()

# ----------------------------
# Plot: Total vaccinations over time
# ----------------------------
plt.figure(figsize=(14, 6))
plt.plot(data['date'], data['total_vaccinations'], color='g', label='Total Vaccinations')
plt.xlabel("Date")
plt.ylabel("Total Vaccinations")
plt.title("Total Vaccinations Over Time")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "total_vaccinations_over_time.png"))
plt.close()

# ----------------------------
# Feature importance analysis
# ----------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

feature_importances = rf_model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Random_Forest_Importance': feature_importances
}).sort_values(by="Random_Forest_Importance", ascending=False).reset_index(drop=True)

perm_importance = permutation_importance(rf_model, X, y, n_repeats=30, random_state=42)
perm_importance_df = pd.DataFrame({
    'Feature': features,
    'Permutation_Importance': perm_importance.importances_mean
}).sort_values(by="Permutation_Importance", ascending=False).reset_index(drop=True)

importance_df = importance_df.merge(perm_importance_df, on="Feature")

# ----------------------------
# Plot: Feature importance comparison
# ----------------------------
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(importance_df['Feature'], importance_df['Random_Forest_Importance'], align='center')
plt.xlabel("Random Forest Feature Importance")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()

plt.subplot(1, 2, 2)
plt.barh(importance_df['Feature'], importance_df['Permutation_Importance'], align='center')
plt.xlabel("Permutation Importance")
plt.title("Permutation Feature Importance")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance_comparison.png"))
plt.close()

# ----------------------------
# Save and print importance table
# ----------------------------
importance_df.to_csv(os.path.join(output_dir, "feature_importance_table.csv"), index=False)

print("\nFeature Importance Comparison Table:\n")
print(importance_df)
