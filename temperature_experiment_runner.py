import random
import os
import csv
import concurrent.futures
import sys
import time

from sklearn.metrics import accuracy_score, precision_score

from config_handler import ConfigHandler

from models.openai import ChatGPTPrompt
from models.octai import OctAI
from models.gemini import Gemini

from helpers.helpers import *
from helpers.result_logging import *

from data_preparation.data_loading import *
# Binary mapping
binary2num = {
    'biased': 1,
    'nonbiased': 0,
    'center': 0,
    'left': 1,
    'right': 1,
    'none': 0
}

# Configuration
models = {
    'ChatGPTPrompt': lambda: ChatGPTPrompt(model_name='gpt-4o-mini', api_key=os.environ['OPENAI_API_KEY']),
    'Llama_Qwen': lambda: OctAI(model_name='qwen2-7b-instruct', api_key=os.environ['OCTAI_API_KEY']),
    'Llama_Meta': lambda: OctAI(model_name='meta-llama-3.1-70b-instruct', api_key=os.environ['OCTAI_API_KEY']),
    'Gemini': lambda: Gemini(api_key=os.environ['GEMINI_API_KEY'], model_name='gemini-1.5-flash')
}

rate_limit_timeout = {
    'ChatGPTPrompt': 2,
    'Llama_Qwen': 2,
    'Llama_Meta': 2,
    'Gemini': 2
}
temperatures = [0.0, 0.25, 0.5, 0.75, 1.0]

# Output directory setup
base_dir = "temperature_experiments"
os.makedirs(base_dir, exist_ok=True)


# Execution function
def execute_model(model_name, temperature, run_idx, texts, annotations, prompt):
    model = models[model_name]()
    sampled_indices = random.sample(range(len(texts)), 100)
    sampled_texts = [texts[i] for i in sampled_indices]
    sampled_annotations = [annotations[i] for i in sampled_indices]

    model_dir = os.path.join(base_dir, model_name)
    output_file_name = f"{model_name}_temp_{temperature}_run_{run_idx}.csv"
    output_file = os.path.join(model_dir, output_file_name)

    predictions = []
    progress = 1
    run_completed = False
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            total_rows = len(rows)
            if total_rows == 100:
                logging.info(f'{output_file_name} already exists, fetching predictions')
                run_completed = True
                for row in rows:
                    pred = int(row['Prediction'])
                    predictions.append(pred)
            else:
                run_completed = False

    if not run_completed:
        print(f"Model:{output_file_name} being executed")
        for text in sampled_texts:
            # if we need to slow down to prevent api hit limits then sleep
            time.sleep(rate_limit_timeout[model_name])

            result = model.predict_single(text, prompt=prompt, temperature=temperature)
            word = extract_single_word(result)
            predictions.append(int(binary2num.get(word, 0)))
            logging.info(f'{model_name} Run {run_idx}: {progress}/100 ')
            progress += 1

    try:
        accuracy = accuracy_score(sampled_annotations, predictions)
        precision = precision_score(sampled_annotations, predictions)
    except ValueError:
        logging.error(f'{model_name} Run {run_idx}: Accuracy score is incorrect')
        logging.info(f"Preds: {predictions}, anns:{sampled_annotations}")
        sys.exit(1)
    # Save individual results

    os.makedirs(model_dir, exist_ok=True)
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Text', 'Annotation', 'Prediction'])
        writer.writerows(zip(sampled_texts, sampled_annotations, predictions))

    return model_name, temperature, run_idx, accuracy, precision

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'media-bias-llm/config.yml')
        config = ConfigHandler.load_config(config_path)
    except FileNotFoundError:
        print("'config.yml' not found")
        sys.exit(1)
    except Exception as e:
        print(e)

    # Main execution loop
    articles_path = '../BASIL/articles'
    annotations_path = '../BASIL/annotations'
    articles, annotations = get_data(articles_path, annotations_path)
    performance_metrics = []
    role = config['variables']['prompt']['roles'][0]
    prompt = prepare_prompt('role', role, 'binary')

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for temperature in temperatures:
            for run_idx in range(1, 4):  # Run 3 times per temperature
                for model_name in models:
                    futures.append(executor.submit(execute_model, model_name, temperature, run_idx, articles, annotations, prompt))

        for future in concurrent.futures.as_completed(futures):
            performance_metrics.append(future.result())

    # Save performance metrics
    metrics_file = os.path.join(base_dir, "performance_metrics.csv")
    with open(metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model Name', 'Temperature', 'Run Index', 'Accuracy', 'Precision'])
        writer.writerows(performance_metrics)
