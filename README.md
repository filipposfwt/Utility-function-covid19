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

You will be prompted to enter the path to your COVID cases CSV file (e.g. `/Users/.../cases_GR.csv`). The script will sequentially execute all analysis steps and output results to `/data/outputs` or as specified in the scripts.