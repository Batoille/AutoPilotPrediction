# -*- coding: utf-8 -*-
"""
FlightDeck.py

Part of the AutoPilot project. Monitors a SQLite database table’s “Brad” column,
maintains a sliding window (deque) of the last N Brad values, feeds that window into
one or more PyTorch LSTM models to predict the next Brad value, and prints status updates.

All predictions (including “Standing Here” markers) are written continuously to a separate
SQLite database (specified by --prediction_db) in a table named “PredictionResults_<model_names>”.
When a prediction row has 'H' in the Standing Here column, then two rows later (i.e. where
PredictionMadeAtID equals that row’s PredictionTargetID), its Action is updated to 'B'
and its Result is set to “WON” or “LOST” based on the multiplier vs. threshold.

When the script is stopped (e.g., via Ctrl+C), the entire contents of the prediction table
are automatically exported to a timestamped CSV file. The CSV’s filename includes
the loaded model names, the threshold multiplier, the bet amount, and the current date/time
for easy reference.

Supports dynamic updating of the threshold multiplier at runtime by typing:
    set_threshold <new_float_value>
in the console. Typing “exit” or pressing Ctrl+C will gracefully shut down the monitor loop
and trigger the CSV export.

On first run, if the specified betdata database path is not known, the script will prompt
the user to enter its full path and will save that in a local `config.json` for future runs.

Usage example (loads Prometheus_6.pt and Prometheus_7.pt under models/ by default):
    python FlightDeck.py

Usage example (explicitly specifying one or more .pt files):
    python FlightDeck.py \
        --pt_files      models\modelA.pt models\modelB.pt \
        --seq_length    4 \
        --poll_interval 1.0 \
        --threshold     1.5 \
        --bet_amount    1.0 \
        --verbose
"""

import os
import sys
import io
import time
import threading
import sqlite3
import argparse
import warnings
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import datetime
import glob
import re

import torch
import torch.nn as nn
import pandas as pd  # Required for CSV export

# ===============================
# Suppress FutureWarning (Optional)
# ===============================
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# ===============================
# ANSI Colors (Defined Globally)
# ===============================
GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"

# ===============================
# Setup Logging
# ===============================
import logging
logger = logging.getLogger("flightdeck")
logger.setLevel(logging.INFO)  # Default, can switch to DEBUG if --verbose

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt="%(asctime)s  %(levelname)-7s  %(message)s",
                              datefmt="%H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


# ===============================
# Custom Logger to Capture Console Output (for final writeout)
# ===============================
class CapturingLogger:
    """
    Duplicates writes to stdout and captures them in an internal buffer.
    """
    def __init__(self):
        self.terminal = sys.stdout
        self.buffer = io.StringIO()

    def write(self, message):
        self.terminal.write(message)
        self.buffer.write(message)

    def flush(self):
        self.terminal.flush()

    def get_contents(self):
        return self.buffer.getvalue()


# ===============================
# Model Definition (LSTM with Positional Encoding)
# ===============================
class LSTMModelWithPositionalEncoding(nn.Module):
    def __init__(self, vocab_size=2, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2, max_seq_len=100):
        super(LSTMModelWithPositionalEncoding, self).__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, vocab_size * 2)  # Predict 2 bits

    def forward(self, src):
        batch_size, seq_len = src.size()
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum supported {self.max_seq_len}")
        embedded = self.embedding(src)
        positions = torch.arange(0, seq_len, device=src.device).unsqueeze(0).expand(batch_size, -1)
        positional_embedded = self.positional_embedding(positions)
        encoded_input = embedded + positional_embedded
        lstm_out, _ = self.lstm(encoded_input)
        last_output = lstm_out[:, -1, :]
        logits = self.fc(last_output)
        return logits.view(batch_size, 2, -1)


# ===============================
# Helper Functions
# ===============================
def sanitize_filename_component(component):
    component = component.replace(' ', '_')
    component = re.sub(r'[^A-Za-z0-9_\-]', '', component)
    return component


def enable_wal_mode(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        mode = cursor.fetchone()
        conn.close()
        logger.info(f"WAL mode enabled. Current journal mode: {mode[0]}")
    except sqlite3.Error as e:
        logger.error(f"Error enabling WAL mode: {e}")


def list_tables_and_info(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"Found {len(tables)} user table(s) in '{db_path}':")
        for table in tables:
            logger.info(f"  ┏━ Table: {table} ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            cursor.execute(f"PRAGMA table_info('{table}')")
            columns = cursor.fetchall()
            if columns:
                for col in columns:
                    col_id, col_name, col_type, col_notnull, col_default, col_pk = col
                    logger.info(f"    • Column ID={col_id}, Name={col_name}, Type={col_type}, "
                                f"NotNull={bool(col_notnull)}, Default={col_default}, PK={bool(col_pk)}")
            else:
                logger.info("    • No columns found for this table.")
            logger.info(f"  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        conn.close()
    except sqlite3.Error as e:
        logger.error(f"Error reading tables from {db_path}: {e}")


def get_last_row(db_path, table_name):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        return row
    except sqlite3.Error as e:
        logger.error(f"SQLite error while fetching last row from '{table_name}': {e}")
        return None


def get_max_id(db_path, table_name):
    """Return the maximum id value in the specified table, or 0 if table is empty or on error."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT MAX(id) FROM {table_name}")
        result = cursor.fetchone()
        conn.close()
        if result is None:
            return 0
        max_id = result[0]
        return max_id if max_id is not None else 0
    except sqlite3.Error as e:
        logger.error(f"SQLite error while fetching MAX(id) from '{table_name}': {e}")
        return 0


def get_column_value(row, column_index):
    if row and len(row) > column_index:
        return row[column_index]
    logger.warning(f"Column index {column_index} is out of range for the row.")
    return None


def update_bank_value(db_path, table_name, row_id, new_bank_value):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"UPDATE {table_name} SET Bank = ? WHERE id = ?", (new_bank_value, row_id))
        conn.commit()
        conn.close()
        logger.info(f"Bank value updated for row ID {row_id} to {new_bank_value}")
    except sqlite3.Error as e:
        logger.error(f"Error updating bank value: {e}")


# Global variable to hold model_names (ensure this is set after models are loaded)
global_model_names = []

predictions_table_dropped = False
def upsert_prediction(pred_db_path, prediction, table_name="PredictionResults"):
    """
    Upserts a prediction row into the prediction database.
    The table is created with fixed columns and extra output tensor columns (one per model).
    """
    global predictions_table_dropped
    global global_model_names
    try:
        conn = sqlite3.connect(pred_db_path)
        cursor = conn.cursor()
        if not predictions_table_dropped:
            logger.info("Creating predictions table with extended columns for each model...")
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            extra_cols = ""
            for model_name in global_model_names:
                col_name = f"OutputTensor_{sanitize_filename_component(model_name)}"
                extra_cols += f", {col_name} TEXT"
            create_table_sql = f"""
                CREATE TABLE {table_name} (
                    id INTEGER,
                    RoundMultiplier REAL,
                    TotalBetAmount REAL,
                    TotalCashOut REAL,
                    ProfitLoss REAL,
                    Brad INTEGER,
                    Bank REAL,
                    PredictionMadeAtID INTEGER PRIMARY KEY,
                    PredictionTargetID INTEGER,
                    Prediction TEXT,
                    [Standing Here] TEXT,
                    Action TEXT,
                    Result TEXT,
                    InputTensor TEXT,
                    OutputPredictions TEXT
                    {extra_cols}
                )
            """
            cursor.execute(create_table_sql)
            predictions_table_dropped = True

        extra_keys = []
        extra_placeholders = []
        extra_values = []
        for model_name in global_model_names:
            key = f"OutputTensor_{sanitize_filename_component(model_name)}"
            extra_keys.append(key)
            extra_placeholders.append("?")
            extra_values.append(prediction.get(key, ""))

        cursor.execute(f"SELECT 1 FROM {table_name} WHERE PredictionMadeAtID = ?", (prediction['PredictionMadeAtID'],))
        exists = cursor.fetchone() is not None

        if exists:
            update_set = (
                "id = ?, RoundMultiplier = ?, TotalBetAmount = ?, TotalCashOut = ?, ProfitLoss = ?, "
                "Brad = ?, Bank = ?, PredictionTargetID = ?, Prediction = ?, [Standing Here] = ?, "
                "Action = ?, Result = ?, InputTensor = ?, OutputPredictions = ?"
            )
            for key in extra_keys:
                update_set += f", {key} = ?"
            update_sql = f"UPDATE {table_name} SET {update_set} WHERE PredictionMadeAtID = ?"
            fixed_vals = [
                prediction['id'],
                prediction['RoundMultiplier'],
                prediction['TotalBetAmount'],
                prediction['TotalCashOut'],
                prediction['ProfitLoss'],
                prediction['Brad'],
                prediction['Bank'],
                prediction['PredictionTargetID'],
                str(prediction['Prediction']),
                prediction.get('Standing Here', ""),
                prediction.get('Action', ""),
                prediction.get('Result', ""),
                prediction.get('InputTensor', ""),
                prediction.get('OutputPredictions', "")
            ]
            vals = fixed_vals + extra_values + [prediction['PredictionMadeAtID']]
            cursor.execute(update_sql, tuple(vals))
            logger.info(f"Updated existing prediction ID {prediction['PredictionMadeAtID']}")
        else:
            fixed_cols = (
                "id, RoundMultiplier, TotalBetAmount, TotalCashOut, ProfitLoss, Brad, Bank, "
                "PredictionMadeAtID, PredictionTargetID, Prediction, [Standing Here], Action, Result, InputTensor, OutputPredictions"
            )
            if extra_keys:
                fixed_cols += ", " + ", ".join(extra_keys)
            placeholders = ", ".join(["?"] * 15)
            if extra_placeholders:
                placeholders += ", " + ", ".join(extra_placeholders)
            insert_sql = f"INSERT INTO {table_name} ({fixed_cols}) VALUES ({placeholders})"
            fixed_vals = [
                prediction['id'],
                prediction['RoundMultiplier'],
                prediction['TotalBetAmount'],
                prediction['TotalCashOut'],
                prediction['ProfitLoss'],
                prediction['Brad'],
                prediction['Bank'],
                prediction['PredictionMadeAtID'],
                prediction['PredictionTargetID'],
                str(prediction['Prediction']),
                prediction.get('Standing Here', ""),
                prediction.get('Action', ""),
                prediction.get('Result', ""),
                prediction.get('InputTensor', ""),
                prediction.get('OutputPredictions', "")
            ]
            vals = fixed_vals + extra_values
            cursor.execute(insert_sql, tuple(vals))
            logger.info(f"Inserted new prediction ID {prediction['PredictionMadeAtID']}")

        conn.commit()
    except Exception as e:
        logger.error(f"Error upserting prediction to SQL: {e}")
    finally:
        conn.close()


# ===============================
# CONFIGURATION‐RELATED HELPERS
# ===============================
def load_config(config_file):
    """
    Load JSON config if it exists. Returns a dict, or {} if no file or invalid JSON.
    """
    if os.path.isfile(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Unable to read config file '{config_file}': {e}")
    return {}


def save_config(config_file, data):
    """
    Atomically write JSON data to config_file.
    """
    try:
        tmp_file = config_file + ".tmp"
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_file, config_file)
        logger.debug(f"Configuration saved to '{config_file}'")
    except Exception as e:
        logger.error(f"Error saving config to '{config_file}': {e}")


def resolve_db_path(provided_path, config_file):
    """
    Determine the actual path to betdata.db:
    1. If provided_path is not None and file exists, use it and save to config.
    2. Else, load config_file; if it contains a valid 'db_path' that still exists, use that.
    3. Otherwise, prompt user to input path until valid, then save to config and return it.
    """
    # 1) If the user provided --db_path and it exists, use it
    if provided_path:
        if os.path.isfile(provided_path):
            logger.info(f"Using command‐line db_path: {provided_path}")
            cfg = load_config(config_file)
            if cfg.get('db_path') != provided_path:
                cfg['db_path'] = provided_path
                save_config(config_file, cfg)
            return provided_path
        else:
            logger.warning(f"Provided db_path '{provided_path}' does not exist. Ignoring it.")

    # 2) Attempt to load from config
    cfg = load_config(config_file)
    saved_path = cfg.get('db_path')
    if saved_path and os.path.isfile(saved_path):
        logger.info(f"Loaded betdata.db path from config: {saved_path}")
        return saved_path

    # 3) Prompt user interactively via console
    logger.warning(f"No valid db_path provided and none found in config. You must supply the path to betdata.db.")
    while True:
        user_input = input("Enter full path to your betdata.db: ").strip()
        if os.path.isfile(user_input):
            logger.info(f"Found database at: {user_input}")
            cfg['db_path'] = user_input
            save_config(config_file, cfg)
            return user_input
        else:
            logger.warning(f"Path '{user_input}' does not exist or is not a file. Please try again.")


# ===============================
# Command Listener (Dynamic Threshold)
# ===============================
def listen_for_commands(threshold_multiplier, lock, stop_event):
    logger.info("Command Listener started. Type 'set_threshold <value>' or 'exit'.")
    while not stop_event.is_set():
        try:
            user_input = input().strip()
        except UnicodeDecodeError as e:
            logger.debug(f"Received invalid input: {e}")
            continue
        except EOFError:
            logger.debug("EOF detected. Exiting command listener.")
            stop_event.set()
            break
        except Exception as e:
            logger.debug(f"Error in command listener: {e}")
            continue

        if not user_input:
            continue

        cmd = user_input.lower().split()
        if cmd[0] == 'exit':
            logger.info("Command listener received 'exit'; shutting down listener.")
            stop_event.set()
            break
        elif cmd[0] == "set_threshold":
            if len(cmd) == 2:
                try:
                    new_val = float(cmd[1])
                    with lock:
                        threshold_multiplier['value'] = new_val
                    logger.info(f"Threshold multiplier updated to {new_val}")
                except ValueError:
                    logger.warning("Invalid value for threshold multiplier.")
            else:
                logger.warning("Invalid command format. Use: set_threshold <new_value>")
        else:
            logger.warning("Unknown command. Available commands: set_threshold <new_value>, exit")


# ===============================
# Main Monitoring Function
# ===============================
def monitor_with_capture(
    db_path, monitored_table, poll_interval, stop_event, target_column_index, brad_deque, models,
    device, all_predictions, pending_preds, seq_length, threshold_multiplier, bet_amount,
    model_names, prediction_db_path, prediction_table, lock, initial_last_id
):
    """
    Polls the monitored_table in the betdata database, but ignores any rows with id <= initial_last_id.
    Only processes rows inserted after the script started.
    """
    if not os.path.exists(db_path):
        logger.error(f"Error: The file '{db_path}' does not exist.")
        return

    logger.info(f"Starting to poll table '{monitored_table}' every {poll_interval} second(s)...")
    previous_last_id = initial_last_id
    bank_value_found = False
    initial_bank_value = 0

    while not stop_event.is_set():
        # Fetch the most recent row
        last_row = get_last_row(db_path, monitored_table)
        if not last_row:
            # Table might be empty; just wait and retry
            time.sleep(poll_interval)
            continue

        current_last_id = get_column_value(last_row, 0)  # Assuming 'id' is at index 0

        # Only react if the row is strictly newer than initial_last_id AND not already processed
        if current_last_id > initial_last_id and current_last_id != previous_last_id:
            logger.debug("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            # Extract columns from the new row
            row_id = current_last_id
            round_multiplier_value = get_column_value(last_row, 1)
            total_bet_amount = get_column_value(last_row, 2)
            total_cash_out = get_column_value(last_row, 3)
            profit_loss = get_column_value(last_row, 4)
            column_value = get_column_value(last_row, target_column_index)     # Brad
            bank_value = get_column_value(last_row, target_column_index + 1)    # Bank

            # First row with a bank value (only among new rows)
            if not bank_value_found:
                initial_bank_value = bank_value
                if initial_bank_value is not None:
                    logger.info(f"Initial Bank value detected (post‐startup): {initial_bank_value}GH₵")
                    bank_value_found = True
                else:
                    logger.debug("Bank value not found in this new row. Skipping until we find a valid bank.")
                    previous_last_id = current_last_id
                    time.sleep(poll_interval)
                    continue

            # Validate the Brad column
            if column_value not in [0, 1]:
                logger.warning(f"Invalid Brad value detected at ID {row_id}: {column_value}. Expected 0 or 1.")
                previous_last_id = current_last_id
                time.sleep(poll_interval)
                continue

            logger.debug(f"New data detected: ID={row_id}, RoundMultiplier={round_multiplier_value}, Brad={column_value}")

            # Process pending predictions whose target is this new row
            if pending_preds:
                for prediction in list(pending_preds):
                    if row_id == prediction['PredictionTargetID']:
                        logger.info(f"Processing pending prediction ID={prediction['PredictionMadeAtID']}")
                        if round_multiplier_value is not None:
                            try:
                                target_round_multiplier_float = float(round_multiplier_value)
                                with lock:
                                    current_threshold = threshold_multiplier['value']
                                logger.debug(f"Current threshold: {current_threshold}")
                                if target_round_multiplier_float > current_threshold:
                                    logger.info(GREEN + "Result: WON" + RESET)
                                    prediction['Result'] = "WON"
                                    expected_bank = initial_bank_value + (bet_amount * (current_threshold - 1))
                                    actual_bank = expected_bank
                                    logger.info(f"Previous Bank Value: {initial_bank_value}GH₵")
                                    initial_bank_value = actual_bank
                                    update_bank_value(db_path, monitored_table, row_id, initial_bank_value)
                                    logger.info(f"Expected Bank Value: {expected_bank}GH₵")
                                    logger.info(f"Actual Bank Value: {actual_bank}GH₵")
                                else:
                                    logger.info(RED + "Result: LOST" + RESET)
                                    prediction['Result'] = "LOST"
                                    expected_bank = initial_bank_value + (bet_amount * (current_threshold - 1))
                                    actual_bank = initial_bank_value - bet_amount
                                    logger.info(f"Previous Bank Value: {initial_bank_value}GH₵")
                                    initial_bank_value = actual_bank
                                    update_bank_value(db_path, monitored_table, row_id, initial_bank_value)
                                    logger.info(f"Expected Bank Value: {expected_bank}GH₵")
                                    logger.info(f"Actual Bank Value: {actual_bank}GH₵")
                            except ValueError:
                                logger.warning("ValueError during conversion. Setting result as UNKNOWN")
                                prediction['Result'] = "UNKNOWN"
                        else:
                            logger.warning("Round multiplier is None. Setting result as UNKNOWN")
                            prediction['Result'] = "UNKNOWN"

                        # Update the prediction record in all_predictions list & database
                        for row_pred in all_predictions:
                            if row_pred['PredictionMadeAtID'] == prediction['PredictionMadeAtID']:
                                logger.debug(f"Updating record for ID {prediction['PredictionMadeAtID']} with Action=B and Result={prediction['Result']}")
                                row_pred['Action'] = 'B'
                                row_pred['Result'] = prediction['Result']
                                upsert_prediction(prediction_db_path, row_pred, table_name=prediction_table)
                                break
                        pending_preds.remove(prediction)
                        logger.debug(f"Removed pending prediction ID={prediction['PredictionMadeAtID']}")

            # Push this new Brad value into the deque
            brad_deque.append(column_value)
            logger.debug(f"Brad Deque now has {len(brad_deque)} entries: {list(brad_deque)}")

            # Once the deque is full, perform inference
            if len(brad_deque) == brad_deque.maxlen and models:
                logger.debug("Deque is full; performing model inference…")
                input_tensor = torch.tensor(list(brad_deque), dtype=torch.long).unsqueeze(0).to(device)
                logger.debug(f"Input tensor: {input_tensor}")

                try:
                    # Run each model's inference concurrently.
                    def run_inference_model(model, tensor):
                        with torch.no_grad():
                            return model(tensor)

                    with ThreadPoolExecutor(max_workers=len(models)) as executor:
                        futures = [executor.submit(run_inference_model, m, input_tensor) for m in models]
                        all_logits = [f.result() for f in futures]

                    # Summarize logits & predicted bits
                    logits_info = []
                    predicted_bits_list = []
                    for i, logits in enumerate(all_logits):
                        pred_bits = torch.argmax(logits, dim=2).squeeze(0).tolist()
                        predicted_bits_list.append(pred_bits)
                        logits_info.append(f"  • {model_names[i]} → logits={logits.squeeze().tolist()}, bits={pred_bits}")
                    logger.debug("Inference results:\n" + "\n".join(logits_info))

                    # Extract the “second bit” from each model
                    second_bits = [bits[1] for bits in predicted_bits_list]
                    logger.debug(f"Model predictions for next Brad values: {second_bits}")

                    with lock:
                        current_threshold = threshold_multiplier['value']

                    if all(bit == 0 for bit in second_bits):
                        logger.info(GREEN + "All models agree to BET" + RESET)
                        action = ""
                        standing_here = "H"
                    else:
                        logger.info(RED + "Not all models agree to BET" + RESET)
                        action = ""
                        standing_here = ""

                    prediction_target_id = row_id + 2
                    prediction_made_at_id = row_id
                    input_tensor_str = str(input_tensor.cpu().tolist())

                    # Build the prediction dictionary
                    new_prediction = {
                        'id': row_id,
                        'RoundMultiplier': round_multiplier_value,
                        'TotalBetAmount': total_bet_amount,
                        'TotalCashOut': total_cash_out,
                        'ProfitLoss': profit_loss,
                        'Brad': column_value,
                        'Bank': bank_value,
                        'PredictionMadeAtID': prediction_made_at_id,
                        'PredictionTargetID': prediction_target_id,
                        'Prediction': second_bits,
                        'Standing Here': standing_here,
                        'Action': action if standing_here == "" else "",
                        'Result': None,
                        'InputTensor': input_tensor_str,
                        'OutputPredictions': str(second_bits)
                    }

                    # For each model, add the second‐row logits string
                    for i, model_name in enumerate(model_names):
                        second_row_tensor = all_logits[i][0, 1]
                        second_row_str = str(second_row_tensor.cpu().tolist())
                        key_name = f"OutputTensor_{sanitize_filename_component(model_name)}"
                        new_prediction[key_name] = second_row_str

                    logger.info(f"New prediction added: ID={prediction_made_at_id}, Targets row {prediction_target_id}")
                    all_predictions.append(new_prediction)
                    upsert_prediction(prediction_db_path, new_prediction, table_name=prediction_table)

                    if standing_here == "H":
                        pending_preds.append({
                            'PredictionMadeAtID': prediction_made_at_id,
                            'PredictionTargetID': prediction_target_id,
                            'Prediction': second_bits,
                            'Action': "B"
                        })
                        logger.debug(f"Pending prediction queued: ID={prediction_made_at_id}")
                    else:
                        all_predictions[-1]['Result'] = ""
                        upsert_prediction(prediction_db_path, all_predictions[-1], table_name=prediction_table)
                        logger.debug(f"Prediction result set to empty for ID={prediction_made_at_id}")

                except Exception as e:
                    logger.error(f"Error during inference: {e}")

        # Always update previous_last_id, so we do not process the same row twice
        previous_last_id = current_last_id
        time.sleep(poll_interval)


# ===============================
# Main Function
# ===============================
def main():
    # Capture console output for a final write
    capturing_logger = CapturingLogger()
    sys.stdout = capturing_logger

    # ──────────────────────────────────────────────────────────────────────────────
    # 1) Compute the path to the “data/predictiondata.db” file, one level up from “src/”
    # ──────────────────────────────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # script_dir will be: ".../AutoPilotPrediction/src"
    default_pred_db = os.path.abspath(os.path.join(script_dir, "../data/predictiondata.db"))
    # default_pred_db → ".../AutoPilotPrediction/data/predictiondata.db"

    parser = argparse.ArgumentParser(description="FlightDeck: Real‐time SQLite Monitor + LSTM Predictions")

    # 1) Remove any hardcoded default for db_path. Allow user to pass --db_path if they prefer.
    parser.add_argument(
        '--db_path', type=str, default=None,
        help='(Optional) Full path to betdata.db. If omitted, the script will prompt on first run.'
    )
    parser.add_argument(
        '--table_name', type=str,
        default="SummaryData",
        help='Name of the table to monitor (default: SummaryData).'
    )
    parser.add_argument(
        '--target_column_index', type=int,
        default=5,
        help='Zero-based index of the column to extract (6th column for Brad, default: 5).'
    )

    # 2) Auto-discover only Prometheus_6.pt and Prometheus_7.pt
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    default_pt_list = []
    for name in ["Prometheus_6.pt", "Prometheus_7.pt"]:
        path = os.path.join(models_dir, name)
        if os.path.isfile(path):
            default_pt_list.append(path)
        else:
            logger.warning(f"Default model '{name}' not found in '{models_dir}'.")

    if not default_pt_list:
        logger.warning(f"No 'Prometheus_6.pt' or 'Prometheus_7.pt' found under '{models_dir}'. You can still pass --pt_files manually.")
    parser.add_argument(
        '--pt_files', type=str, nargs="*",
        default=default_pt_list,
        help=(
            "Paths to .pt model file(s). "
            f"If omitted, Prometheus_6.pt and Prometheus_7.pt under '{models_dir}' will be loaded (if present)."
        )
    )

    # 3) Poll interval, output, seq_len, threshold, bet amount, prediction DB/table
    parser.add_argument(
        '--poll_interval', type=float,
        default=1.0,
        help='Time interval in seconds between polls (default: 1.0).'
    )
    parser.add_argument(
        '--output_predictions', type=str,
        default="predictions.txt",
        help='File to save all predictions (default: predictions.txt).'
    )
    parser.add_argument(
        '--seq_length', type=int,
        default=4,
        help='Sequence length for the model and deque (default: 4).'
    )
    parser.add_argument(
        '--threshold_multiplier', type=float,
        default=1.5,
        help='Initial threshold multiplier to determine WON or LOST (default: 1.5).'
    )
    parser.add_argument(
        '--bet_amount', type=float,
        default=1.0,
        help='Amount to bet for each prediction (default: 1.0).'
    )

    # ──────────────────────────────────────────────────────────────────────────────
    # 2) Override the prediction_db default so it always points to data/predictiondata.db
    # ──────────────────────────────────────────────────────────────────────────────
    parser.add_argument(
        '--prediction_db', type=str,
        default=default_pred_db,
        help=(
            'Path to the predictions SQLite database file '
            f'(default: {default_pred_db})'
        )
    )
    parser.add_argument(
        '--prediction_table', type=str,
        default="PredictionResults",
        help='Name of the prediction table in the predictions database (default: PredictionResults).'
    )

    # 4) Verbosity flag
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='If set, show debug-level logging in the console.'
    )

    args = parser.parse_args()

    # Enable DEBUG logging if requested
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled (DEBUG level).")

    # Config file for storing db_path
    config_file = os.path.join(os.getcwd(), "config.json")

    # Resolve betdata.db path using the new logic (no hardcoded default)
    db_path = resolve_db_path(args.db_path, config_file)
    monitored_table = args.table_name
    target_column_index = args.target_column_index
    pt_file_paths = args.pt_files
    poll_interval = args.poll_interval
    output_predictions_file = args.output_predictions
    seq_length = args.seq_length
    initial_threshold_multiplier = args.threshold_multiplier
    bet_amount = args.bet_amount

    # Use the absolute path we set above
    prediction_db_path = args.prediction_db
    prediction_table = args.prediction_table

    threshold_multiplier = {'value': initial_threshold_multiplier}
    lock = threading.Lock()

    logger.info("┏━ FlightDeck (AutoPilotPrediction) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"┃  Database:        {db_path}")
    logger.info(f"┃  Table:           {monitored_table}")
    logger.info(f"┃  Sequence Length: {seq_length}")
    logger.info(f"┃  Threshold:       {threshold_multiplier['value']}")
    logger.info(f"┃  Bet Amount:      {bet_amount}")
    logger.info(f"┃  Device:          {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enable_wal_mode(db_path)
    list_tables_and_info(db_path)

    models = []
    model_names = []
    for file_path in pt_file_paths:
        model = LSTMModelWithPositionalEncoding(max_seq_len=seq_length).to(device)
        try:
            state_dict = torch.load(file_path, map_location=device)
            model_dict = model.state_dict()
            pretrained_dict = {
                k: v for k, v in state_dict.items()
                if k in model_dict and v.size() == model_dict[k].size()
            }
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            model.eval()
            models.append(model)
            model_names.append(os.path.basename(file_path))
            logger.info(f"Successfully loaded model from '{file_path}'")
        except Exception as e:
            logger.error(f"Error loading .pt file '{file_path}': {e}")

    if not models:
        logger.error("No models were loaded successfully. Exiting.")
        sys.exit(1)

    # Update global list for upsert
    global global_model_names
    global_model_names = model_names

    # If default prediction table name is used, append model names
    if args.prediction_table == "PredictionResults":
        sanitized = "_".join(sanitize_filename_component(name) for name in model_names)
        prediction_table = f"PredictionResults_{sanitized}"
        logger.info(f"Prediction table name set to {prediction_table}")

    # ──────────────────────────────────────────────────────────────────────────────
    # 4) Compute the baseline ID so that we ignore any existing rows in betdata.db
    # ──────────────────────────────────────────────────────────────────────────────
    initial_last_id = get_max_id(db_path, monitored_table)
    logger.info(f"Ignoring all existing rows with id ≤ {initial_last_id}. Only new insertions afterward will be processed.")

    brad_deque = deque(maxlen=seq_length)
    all_predictions = []
    pending_predictions = []

    stop_event = threading.Event()
    command_listener_stop_event = threading.Event()

    # Start command listener thread
    command_listener_thread = threading.Thread(
        target=listen_for_commands,
        args=(threshold_multiplier, lock, command_listener_stop_event),
        daemon=True
    )
    command_listener_thread.start()

    # Define monitoring function
    def monitoring_thread_function():
        monitor_with_capture(
            db_path=db_path,
            monitored_table=monitored_table,
            poll_interval=poll_interval,
            stop_event=stop_event,
            target_column_index=target_column_index,
            brad_deque=brad_deque,
            models=models,
            device=device,
            all_predictions=all_predictions,
            pending_preds=pending_predictions,
            seq_length=seq_length,
            threshold_multiplier=threshold_multiplier,
            bet_amount=bet_amount,
            model_names=model_names,
            prediction_db_path=prediction_db_path,
            prediction_table=prediction_table,
            lock=lock,
            initial_last_id=initial_last_id
        )

    monitoring_thread = threading.Thread(
        target=monitoring_thread_function,
        daemon=True
    )
    monitoring_thread.start()

    logger.info("Table polling started. Press Ctrl+C to stop.")
    logger.info("Type 'exit' in the console listener to stop command listener.")

    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("⏹ Shutting down… please wait.")
        stop_event.set()
        monitoring_thread.join()
        command_listener_stop_event.set()
        command_listener_thread.join()
        logger.info("✔ All threads terminated.")

        # Write plain-text predictions
        if all_predictions:
            with open(output_predictions_file, 'w', encoding='utf-8') as f:
                for pred in all_predictions:
                    f.write(
                        f"Prediction made at ID {pred['PredictionMadeAtID']} for target ID {pred.get('PredictionTargetID')}: "
                        f"{pred['Prediction']} | Standing Here: {pred.get('Standing Here', '')} | Action: {pred.get('Action', '')} | Result: {pred.get('Result', '')}\n"
                    )
            logger.info(f"All predictions saved to {output_predictions_file}")
        else:
            logger.info("No predictions were made during this session.")

        # Export to CSV
        try:
            conn = sqlite3.connect(prediction_db_path)
            extra_cols_query = ""
            for model_name in model_names:
                col_name = f"OutputTensor_{sanitize_filename_component(model_name)}"
                extra_cols_query += f", {col_name}"

            query = f"""
            SELECT
                id,
                RoundMultiplier,
                TotalBetAmount,
                TotalCashOut,
                ProfitLoss,
                Brad,
                Bank,
                PredictionMadeAtID,
                PredictionTargetID,
                Prediction,
                [Standing Here],
                LAG(Action, 2, '') OVER (ORDER BY PredictionMadeAtID) AS Action,
                LAG(Result, 2, '') OVER (ORDER BY PredictionMadeAtID) AS Result,
                InputTensor,
                OutputPredictions
                {extra_cols_query}
            FROM {prediction_table};
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_str = sanitize_filename_component("_".join(model_names))
            csv_name = f"predictions_{model_str}_TM{threshold_multiplier['value']}_BA{bet_amount}_{ts}.csv"

            records_dir = os.path.join(os.getcwd(), "records")
            os.makedirs(records_dir, exist_ok=True)

            csv_path = os.path.join(records_dir, csv_name)
            df.to_csv(csv_path, index=False)
            logger.info(f"Predictions exported to CSV file: {csv_path}")
        except Exception as e:
            logger.error(f"Error exporting predictions to CSV: {e}")

        # Save console log
        try:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_str = sanitize_filename_component("_".join(model_names))
            console_name = (
                f"console_output_{model_str}_TM{sanitize_filename_component(str(threshold_multiplier['value']))}_"
                f"SL{sanitize_filename_component(str(seq_length))}_BA{sanitize_filename_component(str(bet_amount))}_{ts}.txt"
            )
            records_dir = os.path.join(os.getcwd(), "records")
            os.makedirs(records_dir, exist_ok=True)
            console_path = os.path.join(records_dir, console_name)
            with open(console_path, 'w', encoding='utf-8') as f:
                f.write(capturing_logger.get_contents())
            logger.info(f"Console output saved to {console_path}")
        except Exception as e:
           	logger.error(f"Error exporting console output to text file: {e}")
    finally:
        sys.stdout = capturing_logger.terminal


if __name__ == "__main__":
    main()
