import time

import yaml
import os
import logging
import pandas as pd
from datetime import datetime

from models.openai import ChatGPTPrompt
from models.gemini import Gemini
from models.octai import OctAI
from models.anthropic import ClaudeAI

from helpers.helpers import *
from helpers.result_logging import *
from data_preparation.data_loading import *

from config_handler import ConfigHandler

EXPERIMENTS_YAML = "experiments.yml"
experiments = ConfigHandler.load_config(EXPERIMENTS_YAML)

# Initialize models
models = {
    'chatgpt': ChatGPTPrompt(model_name=experiments['models']['chatgpt'], api_key=os.environ['OPENAI_API_KEY']),
    'gemini': Gemini(model_name=experiments['models']['gemini'], api_key=os.environ['GEMINI_API_KEY']),
    'claude': ClaudeAI(api_key=os.environ['ANTHROPIC_API_KEY'], model_name=experiments['models']['claudeai']),
    'llama': OctAI(api_key=os.environ['OCTAI_API_KEY'], model_name=experiments['models']['llama']),
}

rate_limit_timeout = {
    'chatgpt': 0.5,
    'gemini': 0.5,
    'octoai': 1,
    'claudeai': 1.2
}

def load_experiments_config(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


def run_experiment(params,model):
    # Placeholder function: implement the actual experiment run here
    # Call the model's `run_experiment` method with the provided parameters
    sleep_time = rate_limit_timeout[model.product_name]
    results = model.run_experiment(params,sleep_time=sleep_time)
    return results


import concurrent.futures
from datetime import datetime


def run_batch_experiments(articles, anns, models, batch_config):

    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = []
        for prompt_type in batch_config['prompt_types']:
            for prompt_role in batch_config['prompt_roles']:
                for response_type in batch_config['response_types']:
                    for model_name, model in models.items():
                        for fine_tuned in [False]:
                            if model.product_name != 'octoai': continue
                            # Create params for the experiment
                            experiment_name = generate_experiment_name(prompt_type, 'article', prompt_role, response_type)
                            prompt = prepare_prompt(prompt_type,prompt_role,response_type)
                            params = {
                                'experiment_name': experiment_name,
                                'prompt': prompt,
                                'prompt_type': prompt_type,
                                'prompt_role': prompt_role,
                                'texts': articles,
                                'ground_truths': anns,
                                'detection_type': 'article',
                                'num_runs': batch_config.get('num_runs', 3),
                                'fine_tuned': fine_tuned,
                                'classification_type': response_type
                            }

                            # Submit the experiment to be run in a separate thread
                            task = executor.submit(run_experiment, params, model)
                            tasks.append(task)

        # Process the results as they complete
        for task in concurrent.futures.as_completed(tasks):
            try:
                results = task.result()
                log_overall_results( {
                                        'Experiment': results['Experiment'],
                                        'Run': results['Run'],
                                        'Model': results['Model'],
                                        'Prompt_Type': results['Prompt_Type'],
                                        'Prompt_Role': results['Prompt_Role'],
                                        'Prompt': results['Prompt'],
                                        'Fine_Tuned': results['Fine_Tuned'],
                                        'Classification_Type': results['Classification_Type'],
                                        'Detection_Type': results['Detection_Type'],
                                        'Accuracy': results['Accuracy'],
                                        'Precision': results['Precision'],
                                        'Recall': results['Recall'],
                                        'F1-Score': results['F1-Score'],
                                        'No_Runs': results['No_Runs'],
                                    })
            except Exception as e:
                logging.error(f"Error running batch experiment: {e}")


def run_specific_experiments(articles, anns, models, specific_config):
    tasks = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for experiment in specific_config:
            experiment_name = generate_experiment_name(
                experiment['prompt_types'],
                'article',
                experiment['prompt_role'],
                experiment['response_type']
            )
            prompt_type = experiment['prompt_type']
            prompt_role = experiment['prompt_role']
            response_type = experiment['response_type']
            fine_tuned = experiment['fine_tuned']

            prompt = prepare_prompt(prompt_type, prompt_role, response_type)
            model_name = experiment['model']
            model = models[model_name]

            params = {
                'experiment_name': experiment_name,
                'prompt': prompt,
                'prompt_type': prompt_type,
                'prompt_role': prompt_role,
                'texts': articles,
                'ground_truths': anns,
                'detection_type': 'article',
                'num_runs': 3,
                'fine_tuned': fine_tuned,
                'classification_type': response_type
            }

            # Submit the experiment to be run in a separate thread
            task = executor.submit(run_experiment, params, model)
            tasks.append((task, model_name, prompt_type, prompt_role, response_type, fine_tuned))

        # Process the results as they complete
        for task, model_name, prompt_type, prompt_role, response_type, fine_tuned in concurrent.futures.as_completed(
                tasks):
            try:
                results = task.result()
                save_results(results, model_name, prompt_type, prompt_role, response_type, fine_tuned)
                log_overall_results({
                    'Experiment': experiment_name,
                    'Model': model_name,
                    'Prompt_Type': prompt_type,
                    'Prompt_Role': prompt_role,
                    'Response_Type': response_type,
                    'Fine_Tuned': fine_tuned,
                    'Timestamp': datetime.now()
                })
            except Exception as e:
                logging.error(f"Error running specific experiment: {e}")


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    config = load_experiments_config(EXPERIMENTS_YAML)
    articles_path = '../BASIL/articles'
    annotations_path = '../BASIL/annotations'
    articles, anns = get_data(articles_path, annotations_path)
    if 'batch' in config:
        logging.info('Running batch experiments...')
        run_batch_experiments(articles, anns,models, config['batch'])
    # TODO: test this function
    if 'specific' in config:
        pass
    #     run_specific_experiments(articles, anns, models, config['specific'])


if __name__ == "__main__":
    main()
