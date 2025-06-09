import argparse
import copy
import logging
import os
import re
import subprocess
from datetime import datetime
from pprint import pformat
from typing import Mapping, Optional, Union

import torch
import yaml

from src.utils.model import download_model
from src.utils.paths import MODEL_DIR, CONFIG_DIR, DATA_DIR


def update_config(original: dict, updates: dict) -> dict:
    """
    Updates the original config with the updates.

    Parameters:
        original: Original config dictionary.
        updates: Updates to be applied.

    Returns:
        dict: The updated config dictionary.
    """
    original = copy.deepcopy(original)

    for key, value in updates.items():
        if isinstance(value, Mapping) and key in original:
            original[key] = update_config(original.get(key, {}) or {}, value)
        else:
            original[key] = value

    return original


class ModelTuner:
    """A class to fine-tune LLMs using the torchtune framework."""
    _logger = None

    def __init__(self, model: str, hf_token: Optional[str] = None) -> None:
        """
        Initializes the ModelTuner class.

        Parameters:
            model: Model to fine-tune.
            hf_token: Hugging Face token for authentication. Searches environment variables if not provided.
        """
        self._initialize_class_attributes()
        self._model = model
        self._hf_token = hf_token

    @property
    def model(self) -> str:
        """Name or path of the model to fine-tune."""
        return self._model

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """
        Sets up and returns a logger with the specified name and level.

        Returns:
            logging.Logger: A logger instance.
        """
        logger = logging.getLogger("fine-tune")

        # Set handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    @staticmethod
    def get_epochs_trained(output_dir: str) -> int:
        """
        Checks the maximum number of epochs the model has been trained before.

        Parameters:
            output_dir: Directory where the model weights are saved.

        Returns:
            int: The maximum number of epochs trained, or 0 if no epochs are found.
        """
        regex_epoch = re.compile(r'^epoch_(\d+)$')
        epochs = [0]

        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if match := regex_epoch.match(file):
                    epochs.append(int(match.group(1)) + 1)

        return max(epochs)

    @classmethod
    def _initialize_class_attributes(cls) -> None:
        """Initializes class-level attributes."""
        # Set up logger
        if cls._logger is None:
            cls._logger = cls._setup_logger()
            cls.set_logger_level(logging.INFO)

        # Get device setup
        cls._gpu_count = torch.cuda.device_count()

    @classmethod
    def set_logger_level(cls, level: Union[str, int] = logging.INFO) -> None:
        """
        Sets logging level for the class logger.

        Parameters:
            level: Logging level.
        """
        cls._logger.setLevel(level)
        for handler in cls._logger.handlers:
            handler.setLevel(level)

    def create_config(self, model: str, output_dir: str, train_data: str, epochs: int = 1, batch_size: int = 1,
                      use_dev_data: bool = False, resume: bool = False) -> str:
        """
        Creates a new config file for the fine-tuning process.

        Parameters:
            model: Model to fine-tune.
            output_dir: Directory to save the output and config file.
            train_data: Path to the training data file.
            epochs: Number of epochs to train the model.
            batch_size: Batch size for training.
            use_dev_data: Whether to use dev data for training.
            resume: Whether to resume training from a previous checkpoint.

        Returns:
            str: Path to the new config file.
        """
        # Load the default config file
        config_file = "{}{}.yaml"
        default_config_file = config_file.format(model, "")
        default_config_path = os.path.join(CONFIG_DIR, "torchtune_org", default_config_file)
        with open(default_config_path, 'r') as f:
            default_config = yaml.safe_load(f)

        # Todo: add support for validation data

        # Create changes to the config
        modifications = {
            'checkpoint_dir': os.path.join(MODEL_DIR, self._model.split("/")[-1]),
            'output_dir': output_dir,
            'dataset': {
                'source': "json",
                'data_files': train_data,
                'column_map': {
                    'input': 'user_prompt',
                    'output': 'json_output'
                },
                'split': 'train' if not use_dev_data else 'train+validation',
            },
            'resume_from_checkpoint': resume,
            'should_load_recipe_state': resume,
            'epochs': epochs,
            'batch_size': batch_size,
        }

        # Update the default config with the modifications
        new_config = update_config(default_config, modifications)

        self._logger.debug(f"\n##### Default Config #####\n"
                           f"{pformat(default_config)}\n\n")
        self._logger.debug(f"\n##### Changes #####\n"
                           f"{pformat(modifications)}\n\n")
        self._logger.debug(f"\n##### New Config #####\n"
                           f"{pformat(new_config)}\n\n")

        # Save the new config to a file
        new_config_dir = os.path.join(CONFIG_DIR, "torchtune_run")
        new_config_file = output_dir.rsplit('/', maxsplit=1)[-1] + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        new_config_path = os.path.join(new_config_dir, new_config_file)
        os.makedirs(new_config_dir, exist_ok=True)
        with open(new_config_path, 'w') as file:
            yaml.dump(new_config, file, default_flow_style=False)

        return new_config_path

    @classmethod
    def _run_torchtune(cls, config_path: str) -> None:
        """ Runs the torchtune command with the specified config file."""
        cmd = (["tune", "run"] +
               ["lora_finetune_single_device"] +
               ["--config", config_path])

        cls._logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    def finetune(self, epochs: int = 1, batch_size: int = -1, train_data: str = None, use_dev: bool = False) -> None:
        """
        Fine-tunes the model with the specified parameters.

        Parameters:
            epochs: Number of epochs to train the model.
            batch_size: Batch size for training.
            train_data: Path to the training data file. If None, uses the default training data.
            use_dev: Whether to use dev data for training.
        """
        download_model(self._model, self._hf_token)

        # Output directory
        model = self._model.split('/')[-1].lower().replace("-", "_")
        output_dir = os.path.join(MODEL_DIR, f"finetuned_{model}" + ("_with_dev" if use_dev else ""))

        # Data file
        if train_data is None:
            train_data = os.path.join(DATA_DIR, "finetune", "finetune_train.json")

        # Get trained epochs
        epochs_trained = self.get_epochs_trained(output_dir)

        # Create a new config for fine-tuning
        new_config_path = self.create_config(model, output_dir, train_data, epochs, batch_size, use_dev,
                                             resume=bool(epochs_trained))

        # Todo: Invoke torchtune directly instead of using subprocess
        # Fine-tune the model
        self._run_torchtune(new_config_path)


def main():
    """Main function to parse arguments and run ModelTuner."""
    parser = argparse.ArgumentParser(description="Script for fine-tuning Llama 3 Instruct models.")

    # Dataset arguments
    parser.add_argument('--train-data', type=str, default=None,
                        help='Path to the train data file (default: None)')

    # Training arguments
    parser.add_argument('--model', type=str, default='3.2-1B',
                        choices=['3.1-8B', '3.2-1B', '3.2-3B'],
                        help='Name of the model to train (default: 3.2-1B)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Epochs to train the model (default: 1)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for training [1,] (default: 1)')
    parser.add_argument('--use-dev', action='store_true',
                        help='Use dev data for training (default: False)')

    # Logging arguments
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set logging level (default: INFO)')

    args = parser.parse_args()

    # Initialize and run the ModelTuner
    model = "meta-llama/Llama-{}-Instruct".format(args.model)
    tuner = ModelTuner(model)
    tuner.set_logger_level(args.log_level)
    tuner.finetune(args.epochs, args.batch_size, args.train_data, args.use_dev)


if __name__ == "__main__":
    main()
