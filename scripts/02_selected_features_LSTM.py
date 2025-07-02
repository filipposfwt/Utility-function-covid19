import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Verify and load file path from command line
if len(sys.argv) < 3:
    print("Usage: python 02_selected_features_LSTM.py <path_to_csv> <datetime>")
    sys.exit(1)

file_path = sys.argv[1]
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    sys.exit(1)

datetime = sys.argv[2]

# Define output directory
OUTPUT_DIR = "output/02_selected_features_LSTM/" + datetime
if not os.path.exists(OUTPUT_DIR):
    print(f"Creating output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and preprocess data
data = pd.read_csv(file_path)
data = data.fillna(0)

# Exclude 'id' and 'date' columns from analysis
excluded_columns = ['id', 'date']
available_columns = [col for col in data.columns if col not in excluded_columns]

# Function to determine top 10 features from both Random Forest and Permutation Importance
def get_top_features(target_column, selected_features):
    X = data[selected_features]
    y = data[target_column]

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    rf_importances = rf_model.feature_importances_
    features = X.columns

    rf_importance_df = pd.DataFrame({
        'Feature': features,
        'Random_Forest_Importance': rf_importances
    }).sort_values(by="Random_Forest_Importance", ascending=False)

    perm_importance = permutation_importance(rf_model, X, y, n_repeats=30, random_state=42)
    perm_importance_df = pd.DataFrame({
        'Feature': features,
        'Permutation_Importance': perm_importance.importances_mean
    }).sort_values(by="Permutation_Importance", ascending=False)

    top_rf_features = rf_importance_df['Feature'].head(10).tolist()
    top_perm_features = perm_importance_df['Feature'].head(10).tolist()

    return {'rf': top_rf_features, 'perm': top_perm_features}

# Function to create sequences for LSTM
def create_sequences(X, y, sequence_length=30):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    return np.array(X_seq), np.array(y_seq)

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, 128, num_layers=3, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        h = h[:, -1, :]
        out = self.fc(h)
        return out

# Train model
def train_lstm_model(X, y, feature_set_name):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    X_seq, y_seq = create_sequences(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.1, random_state=42, shuffle=False)

    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

    model = LSTMModel(input_size=X_train.shape[2])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_rmse = float("inf")
    best_epoch = -1
    best_model_state = None

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.flatten(), y_train)
        rmse = torch.sqrt(loss)
        rmse.backward()
        optimizer.step()

        if rmse < best_rmse:
            best_rmse = rmse
            best_epoch = epoch
            best_model_state = model.state_dict()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/100, Training RMSE: {rmse.item():.4f}')

    print(f'\nBest epoch: {best_epoch+1} with Training RMSE: {best_rmse:.4f}')

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).flatten()

    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(range(len(y_test)), y_test, label='True Values', color='blue')
    plt.plot(range(len(y_pred)), y_pred, label='Predicted Values', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Deaths')
    plt.title(f'True vs. Predicted Deaths ({feature_set_name}) - LSTM')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(OUTPUT_DIR, f"plot_{feature_set_name.replace(' ', '_')}.png")
    plt.savefig(plot_path)
    plt.close()

    forecast_results = pd.DataFrame({
        'Time': range(1, len(y_test)+1),
        'True Values': y_test,
        'Predicted Values': y_pred
    })

    csv_path = os.path.join(OUTPUT_DIR, f"forecast_{feature_set_name.replace(' ', '_')}.csv")
    forecast_results.to_csv(csv_path, index=False)

    print(f"Saved plot to: {plot_path}")
    print(f"Saved forecast to: {csv_path}")

    return forecast_results

# Run analysis
def run_analysis_and_train_lstm(target_column, selected_features):
    top_features = get_top_features(target_column, selected_features)

    X_rf = data[top_features['rf']].values
    y = data[target_column].values
    print("\nTraining LSTM with top features from Random Forest Importance...")
    forecast_results_rf = train_lstm_model(X_rf, y, "Random Forest Importance")

    X_perm = data[top_features['perm']].values
    print("\nTraining LSTM with top features from Permutation Importance...")
    forecast_results_perm = train_lstm_model(X_perm, y, "Permutation Importance")

    print("\nForecast Results (Random Forest Importance):")
    print(forecast_results_rf.head())

    print("\nForecast Results (Permutation Importance):")
    print(forecast_results_perm.head())

# Main function
def main():
    print("Available columns in the dataset (excluding 'id' and 'date'):\n")
    for idx, col in enumerate(available_columns):
        print(f"{idx}: {col}")

    target_index = int(input("\nEnter the number corresponding to the target column (e.g. Deaths): "))
    target_column = available_columns[target_index]

    numeric_columns = data[available_columns].select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != target_column]

    print("\nAvailable numeric features:")
    for idx, col in enumerate(numeric_columns):
        print(f"{idx}: {col}")

    feature_indices = input("\nEnter indices of features to use (comma-separated): ")
    selected_features = [numeric_columns[int(idx.strip())] for idx in feature_indices.split(',')]

    print(f"\nSelected target: {target_column}")
    print(f"Selected features: {selected_features}")

    run_analysis_and_train_lstm(target_column, selected_features)

if __name__ == "__main__":
    main()
