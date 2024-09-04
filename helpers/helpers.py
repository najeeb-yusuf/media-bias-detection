import logging
import pandas as pd
import os
import numpy as np
from datetime import datetime
import re

from helpers import config

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
    """
    conv_prompt_type = config['experiment_setup']['conversions']['prompt_type']
    conv_text_type = config['experiment_setup']['conversions']['text_type']
    conv_response_type = config['experiment_setup']['conversions']['response_type']
    conv_role = config['experiment_setup']['conversions']['role']


    returned_name = f"{conv_prompt_type[prompt_type]}_{conv_text_type[text_type]}_{conv_role[role]}_{conv_response_type[response_type]}"  # conversions are necessary to keep the length of the sheet names below 31 characters
    return returned_name

def prepare_prompt(prompt_type, role, bias_level='binary'):
    prompts = config['variables']['prompt']['prompt_templates']
    bias_detail = config['variables']['prompt']['bias_detail']
    responses = config['variables']['prompt']['responses']
    return prompts[prompt_type].replace('_insert role_', role).replace('_insert bias detail_', bias_detail[bias_level]).replace('_insert response type_', responses[bias_level])


