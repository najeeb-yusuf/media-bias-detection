import logging
import pandas as pd
import os
import numpy as np
from datetime import datetime
import re

columns = [
    'Experiment', 'Model', 'Prompt_Type', 'Prompt_Role', 'Prompt',
    'Fine_Tuned', 'Classification_Type', 'Detection_Type', 'Accuracy',
    'Precision', 'Recall', 'F1-Score', 'No_Runs'
]


def check_progress(log_file='experiment_progress.xlsx', experiment_name=None):
    """
    Check the progress of experiments.

    :param log_file: Path to the log file storing experiment progress.
    :param experiment_name: Name of the experiment to check progress for.
    :return: A DataFrame containing the progress data for the specified experiment.
    """
    if not os.path.exists(log_file):
        # Return an empty DataFrame with the expected columns if the log file doesn't exist
        return pd.DataFrame(
            columns=['Experiment', 'Run', 'Model', 'Prompt', 'Text', 'Fine_Tuned', 'Prediction', 'Timestamp',
                     'Detection_Type'])

    # Load the Excel file
    with pd.ExcelFile(log_file) as xls:
        if experiment_name in xls.sheet_names:
            # Load the data from the specific sheet
            progress = pd.read_excel(xls, sheet_name=experiment_name)
            if not progress.empty:
                # Return the last entry of the sheet
                return progress.tail(1)
            else:
                # Return an empty DataFrame with the expected columns if the sheet is empty
                return pd.DataFrame(
                    columns=['Experiment', 'Run', 'Model', 'Prompt', 'Text', 'Fine_Tuned', 'Prediction', 'Ground_Truth',
                             'Timestamp', 'Detection_Type', ])
        else:
            # Return an empty DataFrame with the expected columns if the sheet doesn't exist
            return pd.DataFrame(
                columns=['Experiment', 'Run', 'Model', 'Prompt', 'Text', 'Fine_Tuned', 'Prediction', 'Ground_Truth',
                         'Timestamp', 'Detection_Type', 'Provider'])


def initialize_excel(log_file):
    """
    Initialize the Excel file with the 'Overall Results' sheet if it does not exist.
    Creates the Excel file and the 'Overall Results' sheet if needed.

    :param log_file: Path to the log file.
    :return: DataFrame with the log data or empty DataFrame with columns.
    """
    if os.path.exists(log_file):
        # Try to read the 'Overall Results' sheet
        try:
            return pd.read_excel(log_file, sheet_name='Overall Results')
        except ValueError:
            # If 'Overall Results' sheet does not exist, create it
            create_excel_with_sheet(log_file)
            return pd.DataFrame(columns=columns)
    else:
        # If the log file does not exist, create it with the 'Overall Results' sheet
        create_excel_with_sheet(log_file)
        return pd.DataFrame(columns=columns)


def create_excel_with_sheet(log_file):
    """
    Create an Excel file with the 'Overall Results' sheet and the defined columns.

    :param log_file: Path to the log file.
    """
    df = pd.DataFrame(columns=columns)
    with pd.ExcelWriter(log_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Overall Results', index=False)


def log_overall_results(entry_dict, log_file='experiment_results.xlsx'):
    """
    Log overall results to the progress log file.

    :param entry_dict: Dictionary containing values for the entry.
    :param log_file: Path to the log file.
    """
    progress = initialize_excel(log_file)

    # Create a DataFrame for the new entry
    new_entry = pd.DataFrame([entry_dict])

    # Append new entry to the existing progress DataFrame
    progress = pd.concat([progress, new_entry], ignore_index=True)

    # Write the updated DataFrame back to the Excel file
    with pd.ExcelWriter(log_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        progress.to_excel(writer, sheet_name='Overall Results', index=False)


def log_individual_results(entries, log_file):
    """
    Log individual results to the progress log file.

    :param entries: List of dictionaries containing values for each entry.
    :param log_file: Path to the log file.
    """
    # Collect all entries into a DataFrame
    all_entries = []
    for entry_dict in entries:
        individual_df = pd.DataFrame([[
            entry_dict['Experiment'], entry_dict['Run'], entry_dict['Model'],
            entry_dict['Prompt_Type'], entry_dict['Prompt_Role'], entry_dict['Prompt'],
            entry_dict['Text'], entry_dict['FineTuned'], entry_dict['Prediction'],
            entry_dict['Ground_Truth'], datetime.now(), entry_dict['Detection_Type'],
            entry_dict['Classification_Type']
        ]], columns=[
            'Experiment', 'Run', 'Model', 'Prompt_Type', 'Prompt_Role', 'Prompt',
            'Text', 'FineTuned', 'Prediction', 'Ground_Truth', 'Timestamp',
            'Detection_Type', 'Classification_Type'
        ])
        all_entries.append(individual_df)

    # Concatenate all DataFrames into one
    combined_df = pd.concat(all_entries, ignore_index=True)

    # Open the Excel file once and write all entries
    with pd.ExcelWriter(log_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        for entry_dict in entries:
            model = entry_dict['Model'].replace('-', '_').replace(':', '_') if len(entry_dict['Model']) < 10 else \
            entry_dict['Model'].replace('-', '_').replace(':', '_')[:10]
            individual_sheet_name = f"{entry_dict['Experiment']}_{model}"
            if individual_sheet_name in writer.book.sheetnames:
                startrow = writer.book[individual_sheet_name].max_row
                combined_df[combined_df['Experiment'] == entry_dict['Experiment']].to_excel(writer,
                                                                                            sheet_name=individual_sheet_name,
                                                                                            index=False, header=False,
                                                                                            startrow=startrow)
            else:
                combined_df[combined_df['Experiment'] == entry_dict['Experiment']].to_excel(writer,
                                                                                            sheet_name=individual_sheet_name,
                                                                                            index=False)

def extract_single_word(response):
    # Combine valid responses into a single regex pattern
    valid_responses = ["conservative", "liberal", "Non-biased", "NonBiased", "Non Biased", "left", "right", "biased",
                       "biased.", "Non Biased.", "NonBiased.", "center"]
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in valid_responses) + r')\b'

    # Search for the pattern in the response
    match = re.search(pattern, response, re.IGNORECASE)

    # If a match is found, process it
    if match:
        word = match.group(0).lower()
        if word == "conservative":
            return "right"
        elif word == "liberal":
            return "left"
        elif word in ["center", "center."]:
            return "center"
        elif word in ["non-biased", "nonbiased", "non biased.", "nonbiased."]:
            return "nonbiased"
        elif word == "left":
            return "left"
        elif word == "right":
            return "right"
        elif word in ["biased", "biased."]:
            return "biased"
        else:
            return word
    # Return an empty string if no match is found
    return "none"


def generate_experiment_name(prompt_type, text_type, role, response_type):
    """
    - prompt_type is 'simple' or 'role' or 'detailed'
    - text_type is 'article' or 'sentence'
    - role is 'none',Media bias researcher", "Media bias data annotator", "News article writer"
    - reponse_type is 'binary' or 'multiclass'

    NOTE: use yaml specification
    """
    conv_prompt_type = {
        "simple": "s",
        "role": 'r',
        "complex": "c",
    }

    conv_text_type = {
        "article": 'a',
        "sentence": "s"
    }

    conv_role = {
        "none": 'n',
        None: 'n',
        "News article writer": 'w',
        "Media bias researcher": 'r',
        "Media bias data annotator": 'a'
    }

    conv_response_type = {
        "binary": 'b',
        "multiclass": 'm',
    }
    returned_name = f"{conv_prompt_type[prompt_type]}_{conv_text_type[text_type]}_{conv_role[role]}_{conv_response_type[response_type]}"  # conversions are necessary to keep the length of the sheet names below 31 characters
    return returned_name
