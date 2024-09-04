# Large Language Model Experimentation Framework

This project provides a framework for running and analyzing experiments to assess the performance of various large language models (LLMs) across different tasks, this is specifically designed for media bias detection. The framework includes tools for setting up experiments, running them in batches or individually, and logging the results. It is particularly designed to test models on tasks related to media bias detection using different temperature settings.

## Project Structure

```plaintext
.
├── README.md
├── config.yml                 # Configuration file for general settings
├── experiments.yml            # Configuration file for experiment definitions
├── config_handler.py          # Handles the loading of configuration files
├── data_preparation           # Module for data loading and preparation
│   ├── __init__.py
│   ├── data_loading.py
│   └── data_splits.py
├── helpers                    # Utility functions for logging and result processing
│   ├── __init__.py
│   ├── helpers.py
│   └── result_logging.py
├── models                     # Model wrappers for different LLMs
│   ├── __init__.py
│   ├── anthropic.py
│   ├── base_models.py
│   ├── gemini.py
│   ├── octai.py
│   └── openai.py
├── results                    # Directory to store results (generated during runtime)
├── runner.py                  # Main script to run experiments
└── temperature_runner.py      # Script to run experiments with varying temperature settings
```

## Setup

### Prerequisites

- Python 3.10 or higher
- API keys for the models you wish to test (e.g., OpenAI, Gemini, Anthropic, OctAI)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/llm-experimentation.git
   cd llm-experimentation
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables with the appropriate API keys:

   ```bash
   export OPENAI_API_KEY='your-openai-api-key'
   export GEMINI_API_KEY='your-gemini-api-key'
   export ANTHROPIC_API_KEY='your-anthropic-api-key'
   export OCTAI_API_KEY='your-octai-api-key'
   ```

### Configuration

- **`config.yml`**: This file contains general configurations, such as the default roles for prompts and the number of runs for each experiment.
- **`experiments.yml`**: This file defines the specific experiments to be run. It includes details like the models to be tested, the types of prompts, roles, and response types.

## Running Experiments

### Batch Experiments

To run a batch of experiments defined in the `experiments.yml` file, use the `runner.py` script. The script allows you to execute multiple experiments concurrently, logging the results for later analysis.

```bash
python runner.py
```

This script will:

1. Load the experiment configurations.
2. Initialize the specified models.
3. Run the experiments with various combinations of prompts and models.
4. Log the results, including accuracy, precision, recall, and F1-score.

### Temperature Experiments

To assess the impact of different temperature settings on model performance, use the `temperature_runner.py` script. This script runs each experiment with varying temperatures, typically used to control the randomness of the model's output.

```bash
python temperature_runner.py
```

This script will:

1. Run each model with different temperature settings (`0.0`, `0.25`, `0.5`, `0.75`, `1.0`).
2. Sample 100 texts per experiment and compute predictions.
3. Log performance metrics such as accuracy and precision for each run.

## Results and Logging

Results from experiments are saved in the `results` directory. Each experiment generates logs with detailed performance metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

Temperature experiment results are stored in the `temperature_experiments` directory, with metrics saved in a CSV file.

## Extending the Framework

The framework is designed to be easily extendable. You can add new models by implementing a wrapper in the `models` directory and updating the `experiments.yml` and `temperature_runner.py` configurations.

## License

This project is licensed under the MIT License.

## Contact

For any questions or support, please reach out to [Najeeb Yusuf] at [yusufnajlawal@gmail.com].
