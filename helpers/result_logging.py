import logging
import pandas as pd
import os
from datetime import datetime

from helpers import config
from .helpers import *

columns = config['return_files']['overall_return_file_columns']
individual_results_dir = config['return_files']['individual_results_dir']

def check_progress(log_file='experiment_progress.csv', experiment_name=None):
    """
    Check the progress of experiments.

    :param log_file: Path to the log file storing experiment progress.
    :param experiment_name: Name of the experiment to check progress for.
    :return: A DataFrame containing the progress data for the specified experiment.
    """
    cols = config['return_files']['overall_return_file_columns']
    if not os.path.exists(log_file):
        return pd.DataFrame(columns=cols)

    progress = pd.read_csv(log_file)
    if experiment_name:
        progress = progress[progress['Experiment'] == experiment_name]
    return progress.tail(1) if not progress.empty else pd.DataFrame(columns=['Experiment', 'Run', 'Model', 'Prompt', 'Text', 'Fine_Tuned', 'Prediction', 'Ground_Truth', 'Timestamp', 'Detection_Type'])


def initialize_csv(log_file):
    """
    Initialize the CSV file if it does not exist.
    Creates the CSV file with the appropriate columns if needed.

    :param log_file: Path to the log file.
    :return: DataFrame with the log data or empty DataFrame with columns.
    """
    if not os.path.exists(log_file):
        df = pd.DataFrame(columns=columns)
        df.to_csv(log_file, index=False)
        return df
    else:
        return pd.read_csv(log_file)


def log_overall_results(entry_dict):
    overall_results_file = 'overall_results.csv'
    if not os.path.exists(overall_results_file):
        pd.DataFrame(columns=entry_dict.keys()).to_csv(overall_results_file, index=False)

    pd.DataFrame([entry_dict]).to_csv(overall_results_file, mode='a', header=False, index=False)


def log_individual_results(entries):
    """
    Log individual results to CSV files.

    :param entries: List of dictionaries containing values for each entry.
    """
    individual_results_columns = config['return_files']['individual_return_file_columns']
    for entry_dict in entries:
        # Determine the path to save the result
        fine_tuned_folder = 'fine-tuned' if entry_dict['FineTuned'] else 'non-fine-tuned'
        model_folder = entry_dict['Model'].replace('-', '_').replace(':', '_')
        prompt_type_folder = entry_dict['Prompt_Type']
        classification_type_folder = entry_dict['Classification_Type']
        experiment_name = entry_dict['Experiment']



        dir_path = os.path.join(individual_results_dir, fine_tuned_folder, model_folder,
                                prompt_type_folder, classification_type_folder)
        os.makedirs(dir_path, exist_ok=True)

        # Create the CSV file path
        csv_file_path = os.path.join(dir_path, f'{experiment_name}.csv')

        # Create a DataFrame for the entry
        individual_df = pd.DataFrame([[
            entry_dict['Experiment'], entry_dict['Run'], entry_dict['Model'],
            entry_dict['Prompt_Type'], entry_dict['Prompt_Role'], entry_dict['Prompt'],
            entry_dict['Text'], entry_dict['FineTuned'], entry_dict['Prediction'],
            entry_dict['Ground_Truth'], datetime.now(), entry_dict['Detection_Type'],
            entry_dict['Classification_Type']
        ]], columns=individual_results_columns)

        # If the CSV file exists, append the entry
        if os.path.exists(csv_file_path):
            individual_df.to_csv(csv_file_path, mode='a', header=False, index=False)
        else:
            individual_df.to_csv(csv_file_path, index=False)

def save_results(results, model_name, prompt_type, prompt_role, response_type, fine_tuned):
    base_dir = 'individual_results'
    fine_tuned_dir = os.path.join(base_dir, 'fine-tuned' if fine_tuned else 'non-fine-tuned')
    model_dir = os.path.join(fine_tuned_dir, model_name)
    prompt_type_dir = os.path.join(model_dir, prompt_type)
    response_type_dir = os.path.join(prompt_type_dir, response_type)

    experiment_name = generate_experiment_name(prompt_type, 'article', prompt_role, response_type)

    # Ensure directories exist
    os.makedirs(response_type_dir, exist_ok=True)

    # Define CSV file path
    csv_file = os.path.join(response_type_dir, f'{experiment_name}.csv')

    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    if os.path.exists(csv_file):
        results_df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(csv_file, index=False)
