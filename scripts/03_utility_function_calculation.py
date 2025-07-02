# Import libraries
import pandas as pd
import numpy as np
import sys
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

def get_user_input(data):
    """
    Get user input for target column and features through command line interface.
    """
    excluded_columns = ['id', 'date']
    available_columns = [col for col in data.columns if col not in excluded_columns]

    print("\nAvailable columns in the dataset (excluding 'id' and 'date'):")
    for i, col in enumerate(available_columns, 1):
        print(f"{i}. {col}")
    
    while True:
        try:
            target_choice = input("\nEnter the number of the target column: ")
            target_idx = int(target_choice) - 1
            if 0 <= target_idx < len(available_columns):
                target_column = available_columns[target_idx]
                print(f"Selected target: {target_column}")
                break
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Please enter a valid number.")
    
    available_features = [col for col in available_columns if col != target_column]
    print(f"\nAvailable features (excluding target '{target_column}'):")
    for i, col in enumerate(available_features, 1):
        print(f"{i}. {col}")
    
    print("\nEnter feature numbers separated by commas (e.g., 1,3,5) or 'all' for all features:")
    while True:
        try:
            features_input = input("Features: ").strip()
            if features_input.lower() == 'all':
                selected_features = available_features
                break
            else:
                feature_indices = [int(x.strip()) - 1 for x in features_input.split(',')]
                if all(0 <= idx < len(available_features) for idx in feature_indices):
                    selected_features = [available_features[idx] for idx in feature_indices]
                    break
                else:
                    print("Invalid feature numbers. Please try again.")
        except ValueError:
            print("Please enter valid numbers separated by commas, or 'all'.")
    
    print(f"Selected features: {selected_features}")
    return target_column, selected_features

def save_results(output_dir, target_column, importance_df, top_features, r2, equation, 
                 y_values, y_pred, utility_values, smoothed_utility_values):
    """
    Save analysis results to files in the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    with open(os.path.join(output_dir, 'top_features.txt'), 'w') as f:
        f.write("Top 10 Features based on Random Forest Importance:\n")
        for i, feature in enumerate(top_features, 1):
            f.write(f"{i}. {feature}\n")
    
    with open(os.path.join(output_dir, 'model_results.txt'), 'w') as f:
        f.write(f"Target Column: {target_column}\n")
        f.write(f"R-squared of the Polynomial Regression model: {r2:.4f}\n\n")
        f.write("Polynomial Regression Equation:\n")
        f.write(equation)
    
    results_df = pd.DataFrame({
        'actual_values': y_values,
        'predicted_values': y_pred,
        'utility_values': utility_values,
        'smoothed_utility_values': smoothed_utility_values
    })
    results_df.to_csv(os.path.join(output_dir, 'prediction_results.csv'), index=False)
    
    print(f"\nResults saved to: {output_dir}")
    print("Files created:")
    print("- feature_importance.csv")
    print("- top_features.txt")
    print("- model_results.txt")
    print("- prediction_results.csv")
    print("- feature_importance_plot.png")
    print("- fit_curve_plot.png")
    print("- utility_function_plot.png")

def run_analysis(target_column, selected_features, data, output_dir):
    """
    Perform feature importance analysis, polynomial regression, and utility plot generation.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nRunning analysis for target: {target_column}")
    print(f"Using {len(selected_features)} features")

    y = data[target_column]
    X = data[selected_features]

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

    # Plot feature importance
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
    plt.savefig(os.path.join(output_dir, 'feature_importance_plot.png'), dpi=300, bbox_inches='tight')
    # plt.show()  # Disabled for pipeline use

    print("\nFeature Importance Comparison Table:")
    print(importance_df)

    top_features = importance_df['Feature'].head(10).tolist()
    print(f"\nTop 10 Features based on Random Forest Importance: {top_features}")

    X_top = X[top_features]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_top)

    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)
    y_pred = poly_model.predict(X_poly)

    r2 = r2_score(y, y_pred)
    print(f"\nR-squared of the Polynomial Regression model: {r2:.4f}")

    coefficients = poly_model.coef_
    intercept = poly_model.intercept_
    feature_names = poly.get_feature_names_out(top_features)
    
    equation = f"{target_column} = {intercept:.4f}"
    for coef, name in zip(coefficients, feature_names):
        equation += f" + ({coef:.4f} * {name})"
    
    print("\nPolynomial Regression Equation:")
    print(equation)

    plt.figure(figsize=(10, 5))
    plt.plot(y.values, label=f"Observed {target_column}", color="blue")
    plt.plot(y_pred, label="2nd order polynomial fitted curve", color="red", linestyle="--")
    plt.xlabel("Days")
    plt.ylabel(f"Number of {target_column}")
    plt.title("Fit curve assessment")
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'fit_curve_plot.png'), dpi=300, bbox_inches='tight')
    # plt.show()  # Disabled for pipeline use

    def calculate_utility_scaled(values, scale=1):
        utility = scale / (values + 1)
        utility = np.clip(utility, 0, 1)
        return utility

    utility_values = calculate_utility_scaled(y.values)

    window_size = 10
    smoothed_utility_values = np.convolve(utility_values, np.ones(window_size) / window_size, mode='same')
    print(f"\nSmoothed utility values calculated with window size: {window_size}")
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(y.values, label=f"Observed {target_column}", color="blue")
    ax1.set_xlabel("Days")
    ax1.set_ylabel(f"Number of {target_column}", color="black")
    ax1.tick_params(axis='y', labelcolor="black")
    
    ax2 = ax1.twinx()
    ax2.plot(smoothed_utility_values, label="Utility Function (Smoothed)", color="green", linestyle="--")
    ax2.set_ylabel("Utility (0 to 1 Scale)", color="black")
    ax2.tick_params(axis='y', labelcolor="black")

    fig.suptitle(f"Observed {target_column} and Utility Function (Smoothed and Scaled)")
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper left")
    
    plt.savefig(os.path.join(output_dir, 'utility_function_plot.png'), dpi=300, bbox_inches='tight')
    # plt.show()  # Disabled for pipeline use

    save_results(output_dir, target_column, importance_df, top_features, r2, equation, 
                 y.values, y_pred, utility_values, smoothed_utility_values)

def main():
    if len(sys.argv) < 3:
        print("Usage: python 03_utility_function_calculation.py <path_to_csv> <datetime>")
        sys.exit(1)

    file_path = sys.argv[1]
    datetime = sys.argv[2]
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    output_dir = os.path.join("output", "03_utility_function", datetime)

    print(f"Loading data from: {file_path}")
    data = pd.read_csv(file_path)
    data = data.fillna(0)
    
    print(f"Dataset loaded successfully. Shape: {data.shape}")
    print(f"Output will be saved to: {output_dir}")

    target_column, selected_features = get_user_input(data)
    run_analysis(target_column, selected_features, data, output_dir)

if __name__ == "__main__":
    main()
