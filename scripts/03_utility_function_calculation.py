# Import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from scipy.optimize import differential_evolution

# Load the dataset
file_path = 'cases_GR.csv'  # Update the path with your username
data = pd.read_csv(file_path)

# Replace N/A values with zero in the dataset
data = data.fillna(0)

# Function to run analysis based on selected target and independent features
def run_analysis(target_column, selected_features):
    """
    Runs feature importance analysis with Random Forest and Permutation Importance,
    then performs a polynomial regression on the top 10 features, and plots the results.

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

    # Select the top 10 features based on Random Forest Importance
    top_features = importance_df['Feature'].head(10).tolist()
    print("\nTop 10 Features based on Random Forest Importance:\n", top_features)

    # Prepare data for Polynomial Regression with the top 10 features
    X_top = X[top_features]
   

    # Generate polynomial features (degree = 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_top)

    # Fit a linear regression model on the polynomial features
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)

    # Predict target values
    y_pred = poly_model.predict(X_poly)

    # Calculate R-squared
    r2 = r2_score(y, y_pred)
    print(f"\nR-squared of the Polynomial Regression model: {r2:.4f}")

    # Display the polynomial regression equation
    coefficients = poly_model.coef_
    intercept = poly_model.intercept_
    feature_names = poly.get_feature_names_out(top_features)
    
    equation = f"{target_column} = {intercept:.4f}"
    for coef, name in zip(coefficients, feature_names):
        equation += f" + ({coef:.4f} * {name})"
    
    print("\nPolynomial Regression Equation:")
    print(equation)

     # Plotting the actual vs predicted deaths
    plt.figure(figsize=(10, 5))
    plt.plot(y.values, label="Observed new_deaths", color="blue")
    plt.plot(y_pred, label="2nd order polynomial fitted curve", color="red", linestyle="--")
    plt.xlabel("Days")
    plt.ylabel("Number of Deaths")
    plt.title("Fit curve assessment")
    plt.legend()
    plt.show()

    # --- Additional Plot for Utility Function vs Actual Deaths with Scaling and Smoothing ---
    
    # Define the scaled and smoothed utility function
    def calculate_utility_scaled(deaths, scale=1):
        utility = scale / (deaths + 1)
        # Scale utility to be between 0 and 1
        utility = np.clip(utility, 0, 1)
        return utility

    # Calculate utility values based on actual deaths with scaling to [0,1]
    utility_values = calculate_utility_scaled(y.values)

    # Smooth the utility values if there are consecutive similar values
    # Here we use a simple moving average for smoothing
    window_size = 10
    smoothed_utility_values = np.convolve(utility_values, np.ones(window_size) / window_size, mode='same')
    print(smoothed_utility_values)
    
       # Plot actual deaths with a secondary y-axis for utility function
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot actual deaths on the primary y-axis (left side)
    ax1.plot(y.values, label="Observed New Deaths", color="blue")
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Number of Deaths", color="black")
    ax1.tick_params(axis='y', labelcolor="black")
    
    # Create a secondary y-axis for the utility function (right side)
    ax2 = ax1.twinx()
    
    # Calculate scaled and smoothed utility values
    def calculate_utility_scaled(deaths, scale=1):
        utility = scale / (deaths + 1)
        # Scale utility to be between 0 and 1
        utility = np.clip(utility, 0, 1)
        return utility

    # Apply the utility function and smooth the values
    utility_values = calculate_utility_scaled(y.values)
    window_size = 10
    smoothed_utility_values = np.convolve(utility_values, np.ones(window_size) / window_size, mode='same')

    # Plot the utility curve on the secondary y-axis (right side)
    ax2.plot(smoothed_utility_values, label="Utility Function (Smoothed)", color="green", linestyle="--")
    ax2.set_ylabel("Utility (0 to 1 Scale)", color="black")
    ax2.tick_params(axis='y', labelcolor="black")

    # Add title and legends
    fig.suptitle("Observed Deaths and Utility Function (Smoothed and Scaled)")
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper left")
    
    plt.show()


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