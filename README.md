
# AutoPilotPrediction (FlightDeck)

> FlightDeck continuously polls a SQLite “betdata” table, feeds the last N “Brad” values into one or more PyTorch LSTM models, and writes each prediction to a separate SQLite database. On shutdown, it exports all predictions—including “WON/LOST” flags and bank changes—to a timestamped CSV.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
   1. [Recording Wins/Losses & Bank Updates](#recording-winslosses--bank-updates)  
   2. [Back-testing vs. Live Betting](#back-testing-vs-live-betting)  
4. [Model Recommendations](#model-recommendations)  
5. [Getting Started](#getting-started)  
   1. [Prerequisites](#prerequisites)  
   2. [Clone & Install](#clone--install)  
   3. [Setup Configuration](#setup-configuration)  
   4. [Run the Script](#run-the-script)  
6. [Directory Structure](#directory-structure)  
7. [Usage Details](#usage-details)  
   1. [Command-Line Arguments](#command-line-arguments)  
   2. [Interactive Commands](#interactive-commands)  
8. [Examples](#examples)  
   1. [Creating a Dummy Database](#creating-a-dummy-database)  
   2. [Testing & Interpreting Outputs](#testing--interpreting-outputs)  
   1. [Running Smoke Tests](#running-smoke-tests)  
   2. [Adding New Models](#adding-new-models)  
10. [Troubleshooting](#troubleshooting)  
12. [Acknowledgments](#acknowledgments)  

---

## Project Overview

FlightDeck monitors a SQLite database table (by default `SummaryData` in your `betdata.db`), keeps a sliding window of the last N “Brad” values (0 or 1), runs one or more PyTorch LSTM models on that window to predict the next Brad, and logs predictions into a separate SQLite database (`data/predictiondata.db`). When stopped (Ctrl+C or `exit`), it exports all predictions—including whether each prediction “WON” or “LOST” and how your bank balance changed—to a timestamped CSV under `src/records/`.

You can use FlightDeck to:

- **Back-test** any combination of pretrained models against historical data (insert rows manually or via a script, then inspect the CSV to see how each model would have performed).  
- **Live-bet**, by having FlightDeck run on a “live” `betdata.db` that is fed by your betting platform. Whenever the model(s) say “BET,” you place a wager in the next round and record the actual outcome.

---

## Features

- **Real-time polling** of a SQLite table (`SummaryData` by default).  
- **Sliding-window LSTM inference** (configurable `seq_length`, default 4).  
- **Multiple model support**: by default loads exactly `Harmonia_6.pt` and `Hydra_2_2.pt` (full model names) from `models/`, or specify a subset via `--pt_files`.  
- **Dynamic threshold**: type `set_threshold <value>` at runtime to adjust the multiplier that decides “WON” vs “LOST.”  
- **Automatic database setup**: creates `data/predictiondata.db` (and its table) if missing, enables WAL mode, and auto-creates the `data/` folder.  
- **Detailed outputs**:  
  - Writes raw predictions to `data/predictiondata.db`  
  - On shutdown, exports:  
    - `predictions_<models>_TM<threshold>_BA<bet>_<timestamp>.csv` (detailed, with “WON/LOST” and bank values)  
    - `console_output_<models>_TM<threshold>_SL<seq_length>_BA<bet>_<timestamp>.txt` (full log)  
- **Model-agnostic design**: simply drop any additional LSTM‐based `.pt` into `models/`, and FlightDeck will include it when specified.

---


## Model Recommendations

By default (no `--pt_files` passed), FlightDeck loads **Prometheus 6 & Prometheus 7** as an ensemble. Prometheus 6 & 7 work best when you use a **2× threshold multiplier**, offering fewer, high-confidence “BET” signals.

Below are alternative single-model recommendations for different thresholds:

- **Harmonia 6** (`Harmonia_6.pt`):  
  - Ideal when using a **1.01× threshold multiplier** (very low multiplier bets).  
  - Suited for fast, mean-reversion strategies where “Brad” flips frequently.  
  - Example:  
    ```bash
    python src/FlightDeck.py --pt_files models/Harmonia_6.pt --threshold_multiplier 1.01
    ```

- **Hydra 2_2** (`Hydra_2_2.pt`):  
  - Ideal when using a **1.2× threshold multiplier** (moderate multiplier bets).  
  - An ensemble LSTM trained on short- and medium-term trends.  
  - Example:  
    ```bash
    python src/FlightDeck.py --pt_files models/Hydra_2_2.pt --threshold_multiplier 1.2
    ```

- **Prometheus 6 & Prometheus 7** (`Prometheus_6.pt`, `Prometheus_7.pt`):  
  - Default if you run without `--pt_files`.  
  - Best when using a **2× threshold multiplier** (higher multiplier, conservative bets).  
  - Example (default):  
    ```bash
    python src/FlightDeck.py --threshold_multiplier 2.0
    ```
  - Or explicitly:  
    ```bash
    python src/FlightDeck.py --pt_files models/Prometheus_6.pt models/Prometheus_7.pt --threshold_multiplier 2.0
    ```

To add other models, place their `.pt` in `models/` and specify via `--pt_files`. FlightDeck will only place a bet when **all loaded models agree** (predict Brad = 0).

## Getting Started

### Prerequisites

- **Python 3.10+** installed.  
- **Git** (for cloning).  
- *(Optional)* [SQLite CLI](https://www.sqlite.org/download.html) to create or inspect databases from the command line.

### Clone & Install

```bash
# 1. Clone the GitHub repository
git clone https://github.com/Batoille/AutoPilotPrediction.git
cd AutoPilotPrediction

# 2. Create & activate a Python virtual environment
python -m venv venv
# On PowerShell:
.\venv\Scripts\Activate.ps1
# On CMD:
venv\Scripts\activate.bat
# On macOS/Linux:
# source venv/bin/activate

# 3. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Setup Configuration

- On first run, if you don’t pass `--db_path`, FlightDeck prompts:
  ```
  No valid db_path provided and none found in config.
  Enter full path to your betdata.db:
  ```
  Paste the absolute path (e.g. `C:\path\to\betdata.db`) or type it. That path is saved to `src/config.json` for future runs.

- To skip the prompt, manually create `src/config.json`:
  ```json
  {
    "db_path": "C:\\full\\path\\to\\betdata.db"
  }
  ```

### Run the Script

```bash
# With explicit db_path (overrides config.json):
python src/FlightDeck.py --db_path "/full/path/to/betdata.db"

# Or let it prompt you once:
python src/FlightDeck.py
```

- **Output**:  
  - `data/predictiondata.db` is created automatically and populated.  
  - On shutdown (Ctrl+C or typing `exit`), FlightDeck writes:  
    - `src/records/predictions_<models>_TM<threshold>_BA<bet>_<timestamp>.csv`  
    - `src/records/console_output_<models>_TM<threshold>_SL<seq_length>_BA<bet>_<timestamp>.txt`  

---

## Directory Structure

```
AutoPilotPrediction/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   └── .gitkeep             # Placeholder so Git tracks data/
├── models/                  # Place Harmonia_6.pt, Hydra_2_2.pt (and any other .pt models) here
├── src/
│   ├── FlightDeck.py        # Main monitoring + inference loop
│   ├── config.json          # (auto-generated on first run; stores betdata.db path)
│   ├── records/             # Created on shutdown; contains CSV + console logs
│   │   ├── *.csv
│   │   └── *.txt
│   └── (any helper modules)
└── .github/                 # (Optional) GitHub Actions workflows, issue templates, etc.
    └── workflows/
        └── ci.yml
```

---

## Usage Details

### Command-Line Arguments

```text
--db_path <path>             Full path to your betdata.db (optional if in config.json)
--table_name <string>        Table to monitor (default: SummaryData)
--target_column_index <int>  Zero-based index of “Brad” (default: 5)
--pt_files <file1> <file2>   List of .pt model files (default: models/Harmonia_6.pt, models/Hydra_2_2.pt)
--poll_interval <float>      Seconds between polls (default: 1.0)
--seq_length <int>           Sliding window length (default: 4)
--threshold_multiplier <float> Initial multiplier for WON/LOST (default: 1.5)
--bet_amount <float>         Wager per prediction (default: 1.0)
--prediction_db <path>       Where to write predictions DB (default: data/predictiondata.db)
--prediction_table <string>  Base table name (default: PredictionResults)
--verbose                    Enable DEBUG-level logging
```

### Interactive Commands

- **`set_threshold <new_value>`**  
  Type, for example, `set_threshold 1.8` to adjust the threshold in real time.

- **`exit`** or **`Ctrl+C`**  
  Gracefully stops polling, flushes all predictions to CSV/SQLite, and exits.

---

## Examples

### Creating a Dummy Database

If you don’t have a `betdata.db`, create one with just the `SummaryData` schema:

```bash
# Using sqlite3 CLI (Windows/macOS/Linux)
sqlite3 ~/Desktop/sample_betdata.db
```

At the `sqlite>` prompt, run:

```sql
CREATE TABLE SummaryData (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  RoundMultiplier REAL,
  TotalBetAmount  REAL,
  TotalCashOut    REAL,
  ProfitLoss      REAL,
  Brad            INTEGER,
  Bank            REAL
);
.exit
```

You now have `~/Desktop/sample_betdata.db` ready for testing.

### Testing & Interpreting Outputs

1. **Run FlightDeck** (in one terminal, with venv active):
   ```bash
   python src/FlightDeck.py --db_path "~/Desktop/sample_betdata.db"
   ```
   You should see it enable WAL mode, list tables, and begin polling lines.

2. **Insert test rows** (in a second terminal):
   ```bash
   sqlite3 ~/Desktop/sample_betdata.db \
     "INSERT INTO SummaryData (RoundMultiplier, TotalBetAmount, TotalCashOut, ProfitLoss, Brad, Bank) VALUES (2.0, 1.0, 0.0, -1.0, 1, 100);"
   sqlite3 ~/Desktop/sample_betdata.db \
     "INSERT INTO SummaryData (RoundMultiplier, TotalBetAmount, TotalCashOut, ProfitLoss, Brad, Bank) VALUES (1.5, 1.0, 0.0, -0.5, 0, 100);"
   sqlite3 ~/Desktop/sample_betdata.db \
     "INSERT INTO SummaryData (RoundMultiplier, TotalBetAmount, TotalCashOut, ProfitLoss, Brad, Bank) VALUES (1.2, 1.0, 0.0, -0.8, 1, 100);"
   sqlite3 ~/Desktop/sample_betdata.db \
     "INSERT INTO SummaryData (RoundMultiplier, TotalBetAmount, TotalCashOut, ProfitLoss, Brad, Bank) VALUES (0.8, 1.0, 0.0, -0.2, 0, 100);"
   ```

3. **Observe** FlightDeck’s logs:
   - It will note when the deque reaches length 4.  
   - It will run both `Harmonia_6.pt` and `Hydra_2_2.pt` inference, show “BET” or “SKIP,” and queue a pending prediction.  
   - Two inserts later, it will mark “WON”/“LOST” for the original prediction and update the bank in `SummaryData`.  

4. **Stop FlightDeck** (`Ctrl+C`):
   - It writes `src/records/predictions_Harmonia_6_Hydra_2_2_TM<threshold>_BA<bet>_<timestamp>.csv`.  
   - It writes `src/records/console_output_Harmonia_6_Hydra_2_2_TM<threshold>_SL<seq_length>_BA<bet>_<timestamp>.txt`.  

5. **Inspect** `data/predictiondata.db`:
   ```bash
   sqlite3 data/predictiondata.db ".tables"
   # → Should show something like: PredictionResults_Harmonia_6_Hydra_2_2
   sqlite3 data/predictiondata.db "SELECT * FROM PredictionResults_Harmonia_6_Hydra_2_2;"
   ```

---


## Troubleshooting

- **Missing `data/` folder or `predictiondata.db`**  
  - Ensure `data/.gitkeep` exists. FlightDeck auto-creates `data/` and `predictiondata.db`.  
- **“No matching distribution for torch”**  
  - Verify `requirements.txt` has:  
    ```
    --extra-index-url https://download.pytorch.org/whl/cpu
    torch==2.6.0+cpu
    torchvision==0.16.0+cpu
    torchaudio==2.6.0+cpu
    ```
- **Permission denied** on Windows when activating venv:  
  ```powershell
  Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
- **Models fail to load**:  
  - Confirm each `.pt` uses the same `seq_length` and matches `LSTMModelWithPositionalEncoding`.  
  - Look for `logger.error("Error loading .pt file …")` messages.

---


## Acknowledgments

- Lawrence Ayim, my partner on this project.
- Built on top of PyTorch and Python’s SQLite3 library.
- Special thanks to the open-source community.
