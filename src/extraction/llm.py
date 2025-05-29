import json
import logging
import os
import re
from typing import Optional, Union

import jsonschema
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from src.data.enums import Event
from src.utils.paths import DATA_DIR

SCHEMA = {
    "type": "object",
    "properties": {
        "url": {
            "type": ["string", "null"],
            "description": "URL containing additional information."
        },
        "expires": {
            "type": ["string", "null"],
            "description": "The expiration date and time of the information of the alert message."
        },
        "sender": {
            "type": ["string", "null"],
            "description": "The originator of the alert message."
        },
        "location": {
            "type": ["string", "null"],
            "description": "All affected areas of the alert message."
        },
        "event": {
            "type": ["string", "null"],
            "description": "The type of event the alert message is about.",
            "enum": list(Event.__members__.keys())
        }
    },
    "required": ["url", "expires", "sender", "location", "event"]
}

SYSTEM_PROMPT = """You are a helpful and scrupulous assistant that extracts relevant information from the given data and generates a JSON object with the following schema:
{schema}"""

# Todo: Convert the time format to ISO 8601
USER_PROMPT = """Generate a JSON object from the provided alert message.
Use the JSON literal null if the information for a field cannot be found in the message.
The event type MUST be one of the following. Use 'Other' if there is no match:
{events}

Following is an example of the expected input:
headline: 
Flood Advisory issued November 19 at 2:49PM CST expiring November 23 at 9:00AM CST by NWS Chicago IL 
description: 
Illinois River at Ottawa affecting La Salle County
The National Weather Service in Chicago has issued a
* Flood Advisory for
The Illinois River at Ottawa.
* until Thursday morning.
* At  230 PM Sunday the stage was 461.0 feet.
* Action stage is 461.0 feet.
* Flood stage is 463.0 feet.
* Forecast...The river will rise to near 462.0 feet by Monday
morning.
* Impact...At 462.4 feet...Allen Park entrance threatened and west
boat ramp is submerged. 
instruction: 
PRECAUTIONARY/PREPAREDNESS ACTIONS...

Safety message...If you encounter a flooded roadway...turn around and
find an alternate route.

Additional information can be found at weather.gov/chicago.

This is the expected JSON output you should generate from the above alert message:
{{{{
    "event": "FlashFloodWarning",
    "expires": "November 23 at 9:00AM CST",
    "location": "La Salle",
    "sender": "NWS Chicago IL",
    "url": "weather.gov/chicago"
}}}}

The actual alert message that you need to process is as follows:
headline: 
{{headline}}
description: 
{{description}}
instruction: 
{{instruction}}

Return the JSON object, and only the JSON object, using the information above with no extra text or markdown.
""".format(
    events='\n'.join(list(Event.__members__.keys()))
)


def get_logger(name: str, level: Optional[Union[int, str]] = logging.INFO) -> logging.Logger:
    """
    Sets up and returns a logger with the specified name and level.

    Parameters:
        name: Name of the logger.
        level: Logging level.

    Returns:
        logging.Logger: A logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Set handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


class Extractor:
    """
    Extractor for extracting relevant information from alert messages using a predefined schema.
    """
    REGEX_JSON = re.compile(r'\{[\s\S]+\}')
    _logger = None

    def __init__(self, model: str, schema: str = SCHEMA, prompt: str = USER_PROMPT, retries: int = 3) -> None:
        self._initialize_class_attributes()

        self._model_name = model
        self._schema = schema
        self._system_prompt = SYSTEM_PROMPT.format(
            schema=json.dumps({key: value['description'] for key, value in SCHEMA['properties'].items()}, indent=4)
        )
        self._user_prompt = prompt
        self._retries = retries
        self._device = self._get_accelerator()

        # Load model and tokenizer
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            device_map=self._device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model,
            padding_side="left",
            use_fast=True,
        )
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._generation_config = GenerationConfig(
            bos_token_id=self._tokenizer.bos_token_id,
            eos_token_id=self._model.config.eos_token_id,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        self._chat_template_config = {
            "tokenize": False,
            "add_generation_prompt": False,  # Llama does not support generation prompts
            "continue_final_message": True,
        }

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def schema(self) -> dict:
        return self._schema

    @property
    def prompt(self) -> str:
        return self._user_prompt

    @property
    def retries(self) -> int:
        """Number of retries for generating output."""
        return self._retries

    @retries.setter
    def retries(self, value: int) -> None:
        if value < 0:
            raise ValueError("Retries must be a non-negative integer.")
        self._retries = value

    @property
    def device(self) -> torch.device:
        return self._device

    @classmethod
    def _initialize_class_attributes(cls) -> None:
        if cls._logger is None:
            cls._logger = get_logger(cls.__name__, level=logging.INFO)

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

    # Todo: Check URL format using regex
    # Todo: Check if values exist in the input
    @classmethod
    def _validate_json(cls, json_data: dict) -> bool:
        """
        Validates the JSON object against the predefined schema.

        Parameters:
            json_data: The JSON object to validate.

        Returns:
            bool: True if the JSON object is valid, False otherwise.
        """
        try:
            jsonschema.validate(instance=json_data, schema=SCHEMA)
            return True
        except jsonschema.exceptions.ValidationError as e:
            cls._logger.debug(f"Invalid JSON format: {e.message}")
            return False

    @classmethod
    def _get_accelerator(cls) -> torch.device:
        # Check for NVIDIA GPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
        # Check for TPU
        elif 'COLAB_TPU_ADDR' in os.environ:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
        # Check for Apple Silicon GPU
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        # Default to CPU
        else:
            device = torch.device('cpu')

        cls._logger.info(f"Using device: {device}")
        return device

    def _apply_cml(self, user_prompt: str) -> list:
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "The JSON:\n"},
        ]

    # Todo: Add support for batching
    def extract(self, **source) -> dict:
        """
        Extracts relevant fields from the alert message.

        Parameters:
           source: Keyword arguments containing the alert message fields.

        Returns:
            A dictionary containing the extracted fields.
        """
        # Format prompt
        prompts = [
            self._tokenizer.apply_chat_template(
                self._apply_cml(self._user_prompt.format(**source)),
                **self._chat_template_config,
            )
        ]

        model_inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            # truncation=True,
            # max_length=2048
        ).to(self._model.device)
        input_length = model_inputs["input_ids"].shape[1]

        # Generate output
        for _ in range(1 + self._retries):
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=128,
                # do_sample=False,
                # temperature=0.1
            )
            output = self._tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0].strip()

            # Extract JSON from output
            if match := self.REGEX_JSON.search(output):
                output = match.group(0)

            # Parse and validate JSON output
            try:
                json_data = json.loads(output)
                if self._validate_json(json_data):
                    return json_data
            except json.JSONDecodeError:
                self._logger.debug(f"Invalid JSON output: {output}")

        self._logger.error(f"Failed to generate a valid JSON after {self._retries + 1} tries.")
        raise RuntimeError(f"Failed to generate a valid JSON after {self._retries + 1} tries.")


if __name__ == "__main__":
    def read_csv_in_batches(file_path: str, batch_size: int = 10):
        """
        Reads a CSV file in batches.
        """
        for chunk in pd.read_csv(file_path, chunksize=batch_size):
            yield chunk


    data_path = DATA_DIR / "extracted_data.csv"
    extractor = Extractor(model="unsloth/Llama-3.2-3B-Instruct")
    extractor.set_logger_level(logging.DEBUG)

    for batch in read_csv_in_batches(data_path):
        for index, row in batch.iterrows():
            uuid = row.get("uuid", "")
            headline = row.get("headline", "")
            description = row.get("description", "")
            instruction = row.get("instruction", "")

            try:
                extracted_data = extractor.extract(headline=headline, description=description, instruction=instruction)
                print(f"Extracted data for {uuid}: {extracted_data}")
            except RuntimeError as e:
                pass
        break
