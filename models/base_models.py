from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import logging
import random
import time

from collections import Counter

from helpers.helpers import *
from helpers.result_logging import *

binary2num = config['experiment_setup']['conversions']['binary2num']
class BaseModel:
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name

    def predict_single(self, text, prompt, fine_tune=False, no_labels=2):
        pass

    def predict(self, texts, prompt, fine_tuned=False, no_labels=2):
        predictions = []
        for text in texts:
            prediction = self.predict_single(text, prompt, fine_tuned, no_labels)
            predictions.append(prediction)
        return predictions

    def run_experiment(self, params, sleep_time=0.4):
        # Extract parameters from the dictionary
        experiment_name = params.get('experiment_name')
        prompt = params.get('prompt')
        prompt_type = params.get('prompt_type')
        prompt_role = params.get('prompt_role')
        texts = params.get('texts')
        ground_truths = params.get('ground_truths')
        detection_type = params.get('detection_type', 'article')
        num_runs = params.get('num_runs', 1)
        fine_tuned = params.get('fine_tuned', False)
        classification_type = params.get('classification_type', 'binary')

        all_predictions = []
        all_individual_results = []
        logging.info("Prompt: %s", prompt)

        for run in range(1, num_runs + 1):
            preds = []
            size = len(texts)
            progress_counter = 0

            for text, ground_truth in zip(texts, ground_truths):
                # Get the result (Mock prediction as placeholder)
                result = self.predict_single(text, prompt, fine_tuned)

                time.sleep(sleep_time)
                result = extract_single_word(result)
                try:
                    result = binary2num[result.lower()]
                    preds.append(result)
                except Exception as e:
                    result = 0
                    preds.append(result)
                    logging.warning("Invalid result from %s in prompt %s for experiment %s", e, prompt, experiment_name)

                # Convert ground truth if needed
                if isinstance(ground_truth, str):
                    ground_truth = binary2num[ground_truth]

                entry_dict_individual = {
                    'Experiment': experiment_name,
                    'Run': run,
                    'Model': self.model_name,
                    'Prompt_Type': prompt_type,
                    'Prompt_Role': prompt_role,
                    'Prompt': prompt,
                    'Text': text,
                    'FineTuned': fine_tuned,
                    'Prediction': result,
                    'Ground_Truth': ground_truth,
                    'Detection_Type': detection_type,
                    'Classification_Type': classification_type
                }
                all_individual_results.append(entry_dict_individual)

                logging.info(f"Progression: {progress_counter} / {size}")
                progress_counter += 1

            all_predictions.append(preds)

        # Transpose the list of lists to group predictions for each test case across all runs
        transposed_predictions = list(zip(*all_predictions))

        # Get the most common prediction for each test case
        most_common_predictions = [Counter(preds).most_common(1)[0][0] for preds in transposed_predictions]

        predictions = most_common_predictions

        try:
            accuracy = accuracy_score(ground_truths, predictions)
            precision = precision_score(ground_truths, predictions, average='binary')
            recall = recall_score(ground_truths, predictions, average='binary')
            f1 = f1_score(ground_truths, predictions, average='binary')
        except Exception as e:
            logging.error(e)
            accuracy = accuracy_score(ground_truths, predictions)
            precision, recall, f1 = 0, 0, 0

        overall_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        entry_dict_overall = {
            'Experiment': experiment_name,
            'Run': num_runs,
            'Model': self.model_name,
            'Prompt_Type': prompt_type,
            'Prompt_Role': prompt_role,
            'Prompt': prompt,
            'Fine_Tuned': fine_tuned,
            'Classification_Type': classification_type,
            'Detection_Type': detection_type,
            'Accuracy': overall_metrics['accuracy'],
            'Precision': overall_metrics['precision'],
            'Recall': overall_metrics['recall'],
            'F1-Score': overall_metrics['f1'],
            'No_Runs': num_runs
        }

        # Save overall and individual results in CSV format
        logging.info(
            f'Logged overall results for experiment {experiment_name}, model: {self.model_name}, prompt: {prompt}')
        save_results(all_individual_results, self.model_name, prompt_type, prompt_role, classification_type, fine_tuned)
        return entry_dict_overall

# Fine-Tuned Model Class
class FineTunedModel(BaseModel):
    def __init__(self, model_name, api_key, train_texts, train_labels, test_texts, test_labels, num_labels=2):
        super().__init__(api_key)
        self.model_name = model_name
        self.num_labels = num_labels

        self.train_texts = train_texts
        self.train_labels = train_labels
        self.test_texts = test_texts
        self.test_labels = test_labels

    def fine_tune(self, output_dir):
        # To be implemented
        pass

    def predict(self, texts, model_dir=None):
        # To be implemented
        pass

    def run_experiment(self, params):
        pass