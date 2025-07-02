# EURO24 COVID-19 Forecasting & Utility Analysis

This project analyzes COVID-19 death data in Greece using feature selection, LSTM prediction, utility function modeling, and DRL-assisted curve fitting.

## Setup

### Clone the github repository:
```bash
git clone https://github.com/filipposfwt/Utility-function-covid19
cd Utility-function-covid19
```
### Install Python 3.11 (if not already installed):

TensorFlow usually lags behind the latest Python versions because it has many native components and dependencies that must be compiled and tested across platforms. As of now (mid-2025), TensorFlow supports up to Python 3.11.


### Create a new virtual environment using Python 3.11:

```bash
/opt/homebrew/bin/python3.11 -m venv euro24-venv-py311
```

### Activate the new environment:
```bash
source euro24-venv-py311/bin/activate
```

### Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Pipeline

To execute the full analysis pipeline, run:

```bash
python main.py
```

You will be prompted to enter the path to your COVID cases CSV file (e.g. /Users/.../cases_GR.csv). The script will then allow you to interactively select which analysis steps to run, including:

- 01_feature_selection.py — Recursive Feature Elimination
- 02_lstm_prediction.py — LSTM model forecasting
- 03_utility_function_calculation.py — Utility modeling and polynomial regression
- 04_curve_fitting_drl.py — Deep Reinforcement Learning-based curve fitting

Each step is optional, and you can run only the ones you’re interested in.

All results are saved in `output/<step>/<timestamp>/`, where `<timestamp>` is the date and time when you launched `main.py`.

## Running scripts separately

Each script in the pipeline can also be run independently, in order to perform a specific analysis stage.

### Example Usage

#### Step 1 - Feature Selection
```bash
python 01_feature_selection.py /path/to/cases_GR.csv <timestamp>
```
#### Step 2 - Feature Selection LSTM
```bash
python 02_lstm_prediction.py /path/to/cases_GR.csv <timestamp>
```
#### Step 3 — Utility Function Calculation
```bash
python 03_utility_function_calculation.py /path/to/cases_GR.csv <timestamp>
```
#### Step 4 — DRL Curve Fitting
```bash
python 04_curve_fitting_drl.py <timestamp>
```
Replace <timestamp> with the desired folder name (e.g., 20250702_1530) to keep outputs organized.