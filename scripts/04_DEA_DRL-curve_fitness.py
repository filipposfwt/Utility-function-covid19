import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from collections import deque
import random
import os
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# --------------------- User Selects CSV with Format Check ---------------------
def select_csv_file():
    root_dir = 'output/03_utility_function'
    expected_columns = {'smoothed_utility_values', 'utility_value'}

    while True:
        print("\nAvailable directories in 'output/03_utility_function':")
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        for idx, subdir in enumerate(subdirs):
            print(f"  [{idx+1}] {subdir}")

        try:
            dir_choice = int(input("Select a subdirectory by number: "))
            if not 1 <= dir_choice <= len(subdirs):
                raise ValueError
            selected_dir = os.path.join(root_dir, subdirs[dir_choice - 1])
        except ValueError:
            print("[ERROR] Invalid directory selection.")
            continue

        csv_files = [f for f in os.listdir(selected_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"[ERROR] No CSV files found in {selected_dir}")
            continue

        print(f"\nCSV files in '{selected_dir}':")
        for idx, file in enumerate(csv_files):
            print(f"  [{idx+1}] {file}")

        try:
            file_choice = int(input("Select a CSV file by number: "))
            if not 1 <= file_choice <= len(csv_files):
                raise ValueError
            selected_file = os.path.join(selected_dir, csv_files[file_choice - 1])
            data = pd.read_csv(selected_file)

            if 'utility_value' in data.columns:
                x_data = data['Index'].values if 'Index' in data.columns else np.arange(len(data))
                y_true = data['utility_value'].values
                return x_data, np.clip(y_true, 0, 1), selected_file

            elif 'smoothed_utility_values' in data.columns:
                x_data = np.arange(len(data))
                y_true = data['smoothed_utility_values'].values
                return x_data, np.clip(y_true, 0, 1), selected_file

            else:
                print(f"[ERROR] CSV does not contain expected columns: {expected_columns}")
                continue

        except ValueError:
            print("[ERROR] Invalid file selection.")

# --------------------- Output Directory Handling ---------------------
if len(sys.argv) < 2:
    print("Usage: python 04_DEA_DRL-curve_fitness.py <datetime>")
    sys.exit(1)

datetime = sys.argv[1]
output_dir = os.path.join("output", "04_DEA_DRL-curve-fitness", datetime)
os.makedirs(output_dir, exist_ok=True)
print(f"[INFO] Output will be saved to: {output_dir}")

# --------------------- Load Selected CSV ---------------------
x_data, y_true, selected_file = select_csv_file()
print(f"[INFO] Loaded data from: {selected_file}")

# --------------------- Plot Utility Curve ---------------------
plt.plot(x_data, y_true, label="Utility Function Curve", color="grey")
plt.title("Utility Function")
plt.xlabel("Days")
plt.ylabel("Utility Value")
plt.legend()
plt.savefig(os.path.join(output_dir, "utility_function_curve.png"))
plt.close()

# --------------------- Fitness and DE Model ---------------------
def ui(xi, params):
    return 1 / (1 + np.exp(- (params[0] * xi + params[1])))

def fitness_function(variables, x_segment, y_segment):
    k = variables[0]
    ai_vars = variables[1:21]
    ui_params = variables[21:].reshape(20, -1)

    y_pred = np.ones_like(x_segment, dtype=np.float64) * k
    for i in range(20):
        ui_xi = ui(x_segment, ui_params[i])
        term = 1 + ai_vars[i] * ui_xi
        term = np.maximum(term, 0.01)
        y_pred *= term

    y_pred = y_pred / (k + y_pred)
    mse = np.mean(((y_segment - y_pred) / (y_segment + 0.1)) ** 2)
    return mse

# --------------------- DQN Agent ---------------------
state_size = 3
action_size = 5
batch_size = 32

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size * self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size * self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

agent = DQNAgent(state_size, action_size)
mutation_values = [0.1, 0.2, 0.3, 0.4, 0.5]
crossover_values = [0.1, 0.3, 0.5, 0.7, 0.9]
bounds = [(0.1, 2)] + [(-0.3, 1.0) for _ in range(20)] + [(0.0, 1.0), (-5.0, 5.0)] * 20

# --------------------- DEA Segment Fitting ---------------------
N = 20
segment_length = len(x_data) // N
y_pred_total = np.zeros_like(y_true, dtype=np.float64)
ai_values_segments = []

for seg in range(N):
    start_idx = seg * segment_length
    end_idx = len(x_data) if seg == N - 1 else (seg + 1) * segment_length

    x_segment = x_data[start_idx:end_idx]
    y_segment = y_true[start_idx:end_idx]
    if len(x_segment) == 0:
        continue

    result = differential_evolution(
        fitness_function, bounds, args=(x_segment, y_segment),
        strategy='best1bin', maxiter=1, popsize=15, mutation=0.5,
        recombination=0.7, polish=False, init='random',
        disp=False, updating='deferred', workers=1
    )
    population = result.population

    n_generations = 50
    for gen in range(n_generations):
        fitness = np.array([fitness_function(ind, x_segment, y_segment) for ind in population])
        best_fitness = np.min(fitness)
        mean_fitness = np.mean(fitness)
        std_fitness = np.std(fitness)
        state = np.array([best_fitness, mean_fitness, std_fitness]).reshape(1, -1)

        action_index = agent.act(state)
        mutation = mutation_values[action_index // agent.action_size]
        crossover = crossover_values[action_index % agent.action_size]

        result = differential_evolution(
            fitness_function, bounds, args=(x_segment, y_segment),
            strategy='best1bin', maxiter=1, popsize=15, mutation=mutation,
            recombination=crossover, polish=False, init=population,
            disp=False, updating='deferred', workers=1
        )
        population = result.population

    best_idx = np.argmin([fitness_function(ind, x_segment, y_segment) for ind in population])
    best_solution = population[best_idx]

    k = best_solution[0]
    ai_vars = best_solution[1:21]
    ui_params = best_solution[21:].reshape(20, -1)
    ai_values_segments.append(ai_vars)

    y_pred_segment = np.ones_like(x_segment, dtype=np.float64) * k
    for i in range(20):
        term = 1 + ai_vars[i] * ui(x_segment, ui_params[i])
        y_pred_segment *= np.maximum(term, 0.01)
    y_pred_segment = y_pred_segment / (k + y_pred_segment)

    y_pred_total[start_idx:end_idx] = y_pred_segment

    plt.figure()
    plt.plot(x_segment, y_segment, label="Utility values", color="blue")
    plt.plot(x_segment, y_pred_segment, label="Fitted values", color="orange", linestyle="--")
    plt.title(f"Segment {seg+1} Fit")
    plt.xlabel("Days")
    plt.ylabel("Utility Value")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"segment_{seg+1}_fit.png"))
    plt.close()

# --------------------- Overall Fitted Curve Plot ---------------------
plt.figure()
plt.plot(x_data, y_true, label="Utility Function", color="blue")
plt.plot(x_data, y_pred_total, label="Fitted curve", color="orange", linestyle="--")
plt.title("Overall fitted curve")
plt.xlabel("Days")
plt.ylabel("Utility Value")
plt.legend()
plt.savefig(os.path.join(output_dir, "overall_fitted_curve.png"))
plt.close()

# --------------------- Print ai Weights ---------------------
print("\nWeights (a_i) for each segment:")
for seg_idx, (ai_vars, x_segment) in enumerate(zip(ai_values_segments, np.array_split(x_data, N))):
    print(f"\nSegment {seg_idx+1}")
    print("  ai values:")
    for i, ai_value in enumerate(ai_vars):
        print(f"    a{i+1}: {ai_value:.4f}")
