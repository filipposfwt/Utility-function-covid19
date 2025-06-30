# Import libraries
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
import ipywidgets as widgets
from IPython.display import display

# Load the dataset
file_path = 'cases_GR.csv'  # Update the path to your dataset
data = pd.read_csv(file_path)
data = data.fillna(0)

# Function to determine top 10 features from both Random Forest and Permutation Importance
def get_top_features(target_column, selected_features):
    X = data[selected_features]
    y = data[target_column]

    # Initialize and train the Random Forest Regressor model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    # Calculate feature importances from the Random Forest model
    rf_importances = rf_model.feature_importances_
    features = X.columns

    # Random Forest Feature Importance DataFrame
    rf_importance_df = pd.DataFrame({
        'Feature': features,
        'Random_Forest_Importance': rf_importances
    }).sort_values(by="Random_Forest_Importance", ascending=False).reset_index(drop=True)

    # Calculate permutation importance
    perm_importance = permutation_importance(rf_model, X, y, n_repeats=30, random_state=42)
    perm_importance_df = pd.DataFrame({
        'Feature': features,
        'Permutation_Importance': perm_importance.importances_mean
    }).sort_values(by="Permutation_Importance", ascending=False).reset_index(drop=True)

    # Select top 10 features for each approach
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

# Define LSTM model in PyTorch
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

# Function to train the LSTM model using PyTorch and keep the best epoch based on training RMSE
def train_lstm_model(X, y, feature_set_name):
    # Normalize data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # Create sequences
    X_seq, y_seq = create_sequences(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.1, random_state=42, shuffle=False)
    
    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

    model = LSTMModel(input_size=X_train.shape[2])
    criterion = nn.MSELoss()  # MSE criterion for RMSE calculation
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_rmse = float("inf")
    best_epoch = -1
    best_model_state = None

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.flatten(), y_train)
        rmse = torch.sqrt(loss)  # Calculate RMSE from MSE
        rmse.backward()
        optimizer.step()

        # Check if this is the best epoch based on training RMSE
        if rmse < best_rmse:
            best_rmse = rmse
            best_epoch = epoch
            best_model_state = model.state_dict()  # Save best model state

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Training RMSE: {rmse.item():.4f}')

    print(f'\nBest epoch: {best_epoch+1} with Training RMSE: {best_rmse:.4f}')
    
    # Load the best model state based on training RMSE
    model.load_state_dict(best_model_state)

    # Forecasting on the test set using the best model state
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).flatten()

    # Inverse scaling for the test set predictions and true values
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Plot the true values vs. the predicted values for the test set
    plt.figure(figsize=(14, 6))
    plt.plot(range(len(y_test)), y_test, label='True Values', color='blue')
    plt.plot(range(len(y_pred)), y_pred, label='Predicted Values', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Deaths')
    plt.title(f'True vs. Predicted Deaths ({feature_set_name}) - LSTM')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Store point forecasts in a DataFrame for further analysis
    forecast_results = pd.DataFrame({
        'Date': range(1, len(y_test) + 1),
        'True Values': y_test,
        'Predicted Values': y_pred
    })
    print("\nPoint Forecast Results:")
    print(forecast_results)
    return forecast_results

# Main function to run analysis and train LSTM models for both feature importance methods
def run_analysis_and_train_lstm(target_column, selected_features):
    top_features = get_top_features(target_column, selected_features)
    
    # Training and evaluating the model with features selected by Random Forest Importance
    X_rf = data[top_features['rf']].values
    y = data[target_column].values
    print("\nTraining LSTM with top features from Random Forest Importance...")
    forecast_results_rf = train_lstm_model(X_rf, y, feature_set_name="Random Forest Importance")
    
    # Training and evaluating the model with features selected by Permutation Importance
    X_perm = data[top_features['perm']].values
    print("\nTraining LSTM with top features from Permutation Importance...")
    forecast_results_perm = train_lstm_model(X_perm, y, feature_set_name="Permutation Importance")
    
    # Display forecast results for comparison
    print("\nDetailed Point Forecast Results (Random Forest Importance):")
    print(forecast_results_rf)
    
    print("\nDetailed Point Forecast Results (Permutation Importance):")
    print(forecast_results_perm)

# Dropdown and selection widgets for target and features
target_dropdown = widgets.Dropdown(
    options=data.columns.tolist(),
    description='Target:',
    disabled=False
)

features_selector = widgets.SelectMultiple(
    options=data.columns.tolist(),
    description='Features:',
    disabled=False
)

run_button = widgets.Button(description="Run Analysis")

def on_run_button_clicked(b):
    target_column = target_dropdown.value
    selected_features = list(features_selector.value)
    run_analysis_and_train_lstm(target_column, selected_features)

run_button.on_click(on_run_button_clicked)

display(target_dropdown, features_selector, run_button)
