from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import random
from collections import counter
from helpers.helpers import check_progress, initialize_excel, extract_single_word, log_individual_results, log_overall_results
from dictionaries import multiclass2num, binary2num

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

        # Load progress
        progress = check_progress()
        initialize_excel(EXCEL_FILE)

        all_predictions = []
        all_individual_results = []
        logging.info("Prompt:", prompt)
        for run in range(1, num_runs + 1):
            preds = []
            size = len(texts)
            progress_counter = 0
            for text, ground_truth in zip(texts, ground_truths):
                if not progress[(progress['Experiment'] == experiment_name) &
                                (progress['Fine_Tuned'] == fine_tuned) &
                                (progress['Run'] == run) &
                                (progress['Model'] == self.model_name) &
                                (progress['Prompt'] == prompt) &
                                (progress['Text'] == text)].empty:
                    logging.info(
                        f'Skipping completed experiment for run {run}, model: {self.model_name}, prompt: {prompt}, text: {text}')
                    continue

                # Get the result
                # result = self.predict([text], prompt, fine_tuned=fine_tuned)[0].lower()
                result = random.choice(['biased', 'nonbiased'])

                time.sleep(sleep_time)
                result = extract_single_word(result)
                try:
                    if classification_type == 'binary':
                        result = binary2num[result]
                        preds.append(result)
                    else:
                        result = multiclass2num[result]
                        preds.append(result)
                    metric = 'binary'
                except Exception as e:
                    result = 0
                    preds.append(result)
                    logging.warning("Invalid result from ", e, prompt, experiment_name)
                # Save predictions and ground truths for overall metrics calculation

                if type(ground_truth) is str:
                    ground_truth = binary2num[ground_truth] if classification_type == 'binary' else multiclass2num[
                        ground_truth]
                entry_dict_individual = {
                    'Experiment': experiment_name,
                    'Run': run,
                    'Model': self.model_name,
                    'Prompt_Type': prompt_type,
                    'Prompt_Role': prompt_role,  # Replace with actual role if available
                    'Prompt': prompt,
                    'Text': text,
                    'FineTuned': fine_tuned,
                    'Prediction': result,
                    'Ground_Truth': ground_truth,
                    'Detection_Type': detection_type,
                    'Classification_Type': classification_type
                }
                all_individual_results.append(entry_dict_individual)
                print(f"Progression: {progress_counter} / {size}")
                progress_counter += 1

            all_predictions.append(preds)

        # Transpose the list of lists to group predictions for each test case across all runs
        transposed_predictions = list(zip(*all_predictions))

        # Get the most common prediction for each test case
        most_common_predictions = [Counter(preds).most_common(1)[0][0] for preds in transposed_predictions]

        predictions = most_common_predictions

        try:
            accuracy = accuracy_score(ground_truths, predictions)
            precision = precision_score(ground_truths, predictions, average=metric)
            recall = recall_score(ground_truths, predictions, average=metric)
            f1 = f1_score(ground_truths, predictions, average=metric)
        except Exception as e:
            logging.error(e)
            accuracy = accuracy_score(ground_truths, predictions)
            precision = 0
            recall = 0
            f1 = 0

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
        log_overall_results(entry_dict_overall, EXCEL_FILE)
        logging.info(
            f'Logged overall results for experiment {experiment_name}, model: {self.model_name}, prompt: {prompt}')
        log_individual_results(all_individual_results, EXCEL_FILE)