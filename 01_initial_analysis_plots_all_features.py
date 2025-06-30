# Import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/Users/panagiotis/Downloads/cases_GR.csv'  # Update the path to your dataset
data = pd.read_csv(file_path)

# Ensure 'date' column is in datetime format
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Drop rows with missing dates
data = data.dropna(subset=['date']).copy()

# Sort by date to ensure time series continuity
data = data.sort_values(by='date')

# Plot the total deaths over time
plt.figure(figsize=(14, 6))
plt.plot(data['date'], data['total_deaths'], color='b', label='Total Deaths')
plt.xlabel("Date")
plt.ylabel("Total Deaths")
plt.title("Total Deaths Over Time")
plt.legend()
plt.grid(False)
plt.show()

# Plot the new deaths over time
plt.figure(figsize=(14, 6))
plt.plot(data['date'], data['new_deaths'], color='r', label='New Deaths')
plt.xlabel("Date")
plt.ylabel("New Deaths")
plt.title("New Deaths Over Time")
plt.legend()
plt.grid(False)
plt.show()

# Plot the cumulative vaccinations over time
plt.figure(figsize=(14, 6))
plt.plot(data['date'], data['total_vaccinations'], color='g', label='Total Vaccinations')
plt.xlabel("Date")
plt.ylabel("Total Vaccinations")
plt.title("Total Vaccinations Over Time")
plt.legend()
plt.grid(False)
plt.show()

# Drop unnecessary columns and handle missing values for feature importance
data_cleaned = data.drop(columns=['id', 'date']).dropna(subset=['new_deaths']).copy()

# Separate features and target variable
X = data_cleaned.drop(columns=['new_deaths'])
y = data_cleaned['new_deaths']

# Handle any remaining missing values in features
X = X.fillna(X.median())

# Initialize and train the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Calculate feature importances from the Random Forest model
feature_importances = rf_model.feature_importances_
features = X.columns

# Organize Random Forest feature importance into a DataFrame
importance_df = pd.DataFrame({
    'Feature': features,
    'Random_Forest_Importance': feature_importances
}).sort_values(by="Random_Forest_Importance", ascending=False).reset_index(drop=True)

# Calculate permutation importance
perm_importance = permutation_importance(rf_model, X, y, n_repeats=30, random_state=42)
perm_importance_df = pd.DataFrame({
    'Feature': features,
    'Permutation_Importance': perm_importance.importances_mean
}).sort_values(by="Permutation_Importance", ascending=False).reset_index(drop=True)

# Merge Random Forest and Permutation Importance into one DataFrame for comparison
importance_df = importance_df.merge(perm_importance_df, on="Feature")

# Plot Random Forest Feature Importance
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(importance_df['Feature'], importance_df['Random_Forest_Importance'], align='center')
plt.xlabel("Random Forest Feature Importance")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()

# Plot Permutation Importance
plt.subplot(1, 2, 2)
plt.barh(importance_df['Feature'], importance_df['Permutation_Importance'], align='center')
plt.xlabel("Permutation Importance")
plt.title("Permutation Feature Importance")
plt.gca().invert_yaxis()

# Show the plots side-by-side
plt.tight_layout()
plt.show()

# Display the comparison table of Random Forest and Permutation Importance
print("\nFeature Importance Comparison Table:\n")
print(importance_df)

