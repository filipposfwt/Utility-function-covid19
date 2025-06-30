import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from collections import deque
import random

# Για το νευρωνικό δίκτυο
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Φορτώνουμε τα δεδομένα
data = pd.read_csv('utility_function_data_indexed.csv')
x_data = data['Index'].values
y_true = data['utility_value'].values

# Σχεδίαση της καμπύλης από το utility function
plt.plot(x_data, y_true, label="Utility Function Curve", color="grey")
plt.title("Utility Function")
plt.xlabel("Days")
plt.ylabel("Utility Value")
plt.legend()
plt.show()

# Εξασφαλίζουμε ότι το y_true βρίσκεται στο διάστημα [0,1]
y_true = np.clip(y_true, 0, 1)

# Ορισμός της συνάρτησης u_i(x_i)
def ui(xi, params):
    return 1 / (1 + np.exp(- (params[0] * xi + params[1])))

# Τροποποιημένη συνάρτηση fitness με προσαρμογή στις χαμηλές τιμές του y_segment
def fitness_function(variables, x_segment, y_segment):
    k = variables[0]  # Δυναμική σταθερά k
    ai_vars = variables[1:21]  # 20 παράμετροι a_i
    ui_params = variables[21:].reshape(20, -1)  # 20 σύνολα παραμέτρων για u_i(x_i)

    # Υπολογισμός της προσαρμοσμένης τιμής y για το segment
    y_pred = np.ones_like(x_segment, dtype=np.float64) * k
    for i in range(20):
        ui_xi = ui(x_segment, ui_params[i])
        term = 1 + ai_vars[i] * ui_xi
        # Βεβαιωθείτε ότι το term δεν είναι αρνητικό
        term = np.maximum(term, 0.01)
        y_pred *= term

    # Κανονικοποίηση του y_pred στο διάστημα (0,1)
    y_pred = y_pred / (k + y_pred)

    # Υπολογισμός του MSE
    mse = np.mean(((y_segment - y_pred) / (y_segment + 0.1)) ** 2)
    return mse

# Παράμετροι για τον RL agent
state_size = 3  # [best_fitness, mean_fitness, std_fitness]
action_size = 5  # Διακριτές ενέργειες για mutation και crossover
batch_size = 32

# Καθορίζουμε τον RL Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # Μέγεθος κατάστασης
        self.action_size = action_size  # Μέγεθος δράσης (επιλογές για mutation και crossover)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Συντελεστής έκπτωσης
        self.epsilon = 1.0  # Αρχική πιθανότητα εξερεύνησης
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Νευρωνικό δίκτυο για την προσέγγιση της συνάρτησης Q
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size * self.action_size, activation='linear'))  # Συνδυασμός ενεργειών
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # Αποθήκευση εμπειριών
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Επιλογή δράσης βάσει της πολιτικής ε-απληστίας
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size * self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        # Εκπαίδευση του νευρωνικού δικτύου
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # Μείωση του epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Αρχικοποίηση του agent
agent = DQNAgent(state_size, action_size)

# Τιμές για mutation και crossover
mutation_values = [0.1, 0.2, 0.3, 0.4, 0.5]
crossover_values = [0.1, 0.3, 0.5, 0.7, 0.9]

# Τροποποιημένα όρια για τα ai_vars[i]
bounds = [(0.1, 2)] + [(-0.3, 1.0) for _ in range(20)] + [(0.0, 1.0), (-5.0, 5.0)] * 20

# Διαχωρισμός των δεδομένων σε segments
N = 20  # Αριθμός segments
segment_length = len(x_data) // N

# Προετοιμασία για αποθήκευση των προβλέψεων και των βαρών ai
y_pred_total = np.zeros_like(y_true, dtype=np.float64)
ai_values_segments = []  # Λίστα για αποθήκευση των ai για κάθε segment

# Επεξεργασία κάθε segment
for seg in range(N):
    start_idx = seg * segment_length
    if seg == N - 1:
        end_idx = len(x_data)
    else:
        end_idx = (seg + 1) * segment_length

    x_segment = x_data[start_idx:end_idx]
    y_segment = y_true[start_idx:end_idx]

    # Ελέγχουμε αν το segment έχει δεδομένα
    if len(x_segment) == 0:
        continue

    # Αρχικοποίηση πληθυσμού για το DE
    result = differential_evolution(
        fitness_function,
        bounds,
        args=(x_segment, y_segment),
        strategy='best1bin',
        maxiter=1,  # Θα ελέγχουμε εμείς τον αριθμό των γενεών
        popsize=15,
        mutation=0.5,
        recombination=0.7,
        polish=False,
        init='random',
        disp=False,
        updating='deferred',
        workers=1
    )
    population = result.population

    n_generations = 50  # Αριθμός γενεών ανά segment
    for gen in range(n_generations):
        fitness = np.array([fitness_function(ind, x_segment, y_segment) for ind in population])
        best_fitness = np.min(fitness)
        mean_fitness = np.mean(fitness)
        std_fitness = np.std(fitness)
        state = np.array([best_fitness, mean_fitness, std_fitness]).reshape(1, -1)

        # Επιλογή δράσης από τον agent
        action_index = agent.act(state)
        mutation = mutation_values[action_index // agent.action_size]
        crossover = crossover_values[action_index % agent.action_size]

        # Εκτέλεση DE για μία γενιά με τις επιλεγμένες παραμέτρους
        result = differential_evolution(
            fitness_function,
            bounds,
            args=(x_segment, y_segment),
            strategy='best1bin',
            maxiter=1,
            popsize=15,
            mutation=mutation,
            recombination=crossover,
            polish=False,
            init=population,
            disp=False,
            updating='deferred',
            workers=1
        )

        # Ενημέρωση του πληθυσμού
        population = result.population

    # Μετά το πέρας των γενεών, λαμβάνουμε το καλύτερο άτομο
    fitness = np.array([fitness_function(ind, x_segment, y_segment) for ind in population])
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx]

    # Υπολογισμός της προσαρμοσμένης καμπύλης για το segment
    k = best_solution[0]
    ai_vars = best_solution[1:21]
    ui_params = best_solution[21:].reshape(20, -1)
    ai_values_segments.append(ai_vars)

    y_pred_segment = np.ones_like(x_segment, dtype=np.float64) * k
    for i in range(20):
        ui_xi = ui(x_segment, ui_params[i])
        term = 1 + ai_vars[i] * ui_xi
        # Βεβαιωθείτε ότι το term δεν είναι αρνητικό
        term = np.maximum(term, 0.01)
        y_pred_segment *= term

    # Κανονικοποίηση του y_pred_segment στο διάστημα (0,1)
    y_pred_segment = y_pred_segment / (k + y_pred_segment)
    y_pred_total[start_idx:end_idx] = y_pred_segment

    # Σχεδίαση του fit σε αντιδιαστολή με τις πραγματικές τιμές για το segment
    plt.figure()
    plt.plot(x_segment, y_segment, label="Utility values", color="blue")
    plt.plot(x_segment, y_pred_segment, label="Fitted values", color="orange", linestyle="--")
    plt.title(f"Segment {seg+1} Fit")
    plt.xlabel("Days")
    plt.ylabel("Utility Value")
    plt.legend()
    plt.show()

# Σχεδίαση της συνολικής καμπύλης προσαρμογής
plt.figure()
plt.plot(x_data, y_true, label="Utility Function", color="blue")
plt.plot(x_data, y_pred_total, label="Fitted curve", color="orange", linestyle="--")
plt.title("Overall fitted curve")
plt.xlabel("Days")
plt.ylabel("Utility Value")
plt.legend()
plt.show()

# Εκτύπωση των βαρών ai για κάθε segment, καθώς και των τιμών xi και ui(xi)
print("\nWeights (a_i) for each segment:")

for seg_idx, (ai_vars, x_segment) in enumerate(zip(ai_values_segments, np.array_split(x_data, N))):
    print(f"\nSegment {seg_idx+1}")
    print("  ai values:")
    for i, ai_value in enumerate(ai_vars):
        print(f"    a{i+1}: {ai_value}")
    
