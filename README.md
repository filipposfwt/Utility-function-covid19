# EURO24 COVID-19 Forecasting & Utility Analysis

This project analyzes COVID-19 death data in Greece using feature selection, LSTM prediction, utility function modeling, and DRL-assisted curve fitting.

## Setup

### 1. **Clone the github repository:
```bash
git clone https://github.com/filipposfwt/Utility-function-covid19
```
### 2. **Install Python 3.11** (if not already installed):

TensorFlow usually lags behind the latest Python versions because it has many native components and dependencies that must be compiled and tested across platforms. As of now (mid-2025), TensorFlow supports up to Python 3.11.

Install python with the following commands:

#### Linux 

#### MacOS

```bash
brew install python@3.11
```

### 3. **Create a new virtual environment using Python 3.11:

```bash
/opt/homebrew/bin/python3.11 -m venv euro24-venv-py311
```

### 4. Activate the new environment:
```bash
source euro24-venv-py311/bin/activate
```

### 5. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the pipeline