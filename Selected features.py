# Import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Load the dataset
file_path = 'cases_GR.csv'  # Update the path with your username
data = pd.read_csv(file_path)

# Replace N/A values with zero in the dataset
data = data.fillna(0)

# Function to run analysis based on selected target and independent features
def run_analysis(target_column, selected_features):
    """
    Runs feature importance analysis with Random Forest and Permutation Importance
    for the specified target column and selected features.

    Parameters:
    - target_column: The name of the target (dependent) variable.
    - selected_features: List of feature (independent) columns to analyze.
    """

    # Separate features and target variable
    y = data[target_column]
    X = data[selected_features]

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

# Dropdown widget for selecting the target column
target_dropdown = widgets.Dropdown(
    options=data.columns.tolist(),
    description='Target:',
    disabled=False
)

# SelectMultiple widget for selecting multiple features
features_selector = widgets.SelectMultiple(
    options=data.columns.tolist(),
    description='Features:',
    disabled=False
)

# Button to run analysis
run_button = widgets.Button(description="Run Analysis")

# Define function to be called when the button is clicked
def on_run_button_clicked(b):
    target_column = target_dropdown.value
    selected_features = list(features_selector.value)
    run_analysis(target_column, selected_features)

# Attach the function to the button
run_button.on_click(on_run_button_clicked)

# Display widgets
display(target_dropdown, features_selector, run_button)
