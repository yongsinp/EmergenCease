import json
import logging
import os
import re
from copy import deepcopy
from typing import Optional, Union, Any

import jsonschema
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.data.enums import Event
from src.extraction.ner import REGEX_URL
from src.utils.file import read_csv_in_batches
from src.utils.model import download_model, check_model
from src.utils.paths import DATA_DIR, MODEL_DIR

SCHEMA = {
    "type": "object",
    "properties": {
        "url": {
            "type": ["string"],
            "description": "URL leading to additional information, exactly as written in the alert message, separated by semicolons. Leave empty if no URL is found.",
        },
        "expires": {
            "type": ["string"],
            "description": "The expiration date and time of the information of the alert message. Leave empty if no expiration time is found."
        },
        "sender": {
            "type": ["string"],
            "description": "The organization that sent the alert message. Leave empty if no sender is found."
        },
        "location": {
            "type": ["string"],
            "description": "All affected areas (i.e., the longest substring as written in the alert message), separated by semicolons. Leave empty if no location is found."
        },
        "event": {
            "type": ["string"],
            "description": "The type of event the alert message is about.",
            "enum": list(Event.__members__.keys())
        }
    },
    "required": ["url", "expires", "sender", "location", "event"]
}

# Todo: Move prompts to a config file
SYSTEM_PROMPT = """You are a helpful and scrupulous assistant that extracts relevant information from the given data and generates a JSON object with the following schema:
{schema}"""

# Todo: Convert the time format to ISO 8601
USER_PROMPT = """Generate a JSON object from the provided alert message.
Use empty string "" if the information for a field cannot be found in the message.
The event type MUST be one of the following. Use "Other" if there is no match:
{events}

Following is an example of the expected input:
headline: 
Flash Flood Warning issued November 19 at 2:49PM CST expiring November 23 at 9:00AM CST by NWS Chicago IL 
description: 
Illinois River at Ottawa affecting La Salle County
The National Weather Service in Chicago has issued a
* Flash Flood Warning for
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
    "location": "Illinois River at Ottawa; La Salle County; Allen Park",
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
    Extractor for extracting relevant information from alert messages using an LLM and a predefined schema.
    """
    REGEX_JSON = re.compile(r'\{[\s\S]+\}')
    _logger = None

    def __init__(self, model: str, schema: str = SCHEMA, prompt: str = USER_PROMPT, adapter: str = None,
                 retries: int = 4, hf_token: Optional[str] = None) -> None:
        """
        Initializes the Extractor with a specified LLM.

        Parameters:
            model: Hugging Face model identifier for the model used in extraction.
            schema: A JSON schema defining the expected output format.
            prompt: A user prompt to guide the LLM in generating the output.
            adapter: Optional path to a model adapter.
            retries: Number of retries for generating output if the first attempt fails.
            hf_token: Optional Hugging Face token for authentication. Searches environment variables if not provided.
        """
        self._initialize_class_attributes()

        self._model_name = model.split('/')[-1]
        self._adapter_path = os.path.join(MODEL_DIR, adapter)
        self._schema = schema
        self._system_prompt = SYSTEM_PROMPT.format(
            schema=json.dumps({key: value['description'] for key, value in SCHEMA['properties'].items()}, indent=4)
        )
        self._user_prompt = prompt
        self._retries = retries
        self._device = self._get_accelerator()

        if not check_model(model):
            download_model(model, hf_token)

        # Load model and tokenizer
        model_path = os.path.join(MODEL_DIR, self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self._device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
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

        # Load adapter if provided
        if self._adapter_path:
            self._model = PeftModel.from_pretrained(self._model, self._adapter_path)

    @property
    def model(self) -> str:
        """Model name or path."""
        return self._model_name

    @property
    def schema(self) -> dict:
        """JSON schema defining the expected output format."""
        return self._schema

    @property
    def prompt(self) -> str:
        """User prompt to guide the LLM in generating the output."""
        return self._user_prompt

    @property
    def retries(self) -> int:
        """Number of retries for generating output."""
        return self._retries

    @retries.setter
    def retries(self, value: int) -> None:
        """
        Number of retries for generating output.

        Parameters:
            value: A non-negative integer representing the number of retries.
        """
        if value < 0:
            raise ValueError("Retries must be a non-negative integer.")
        self._retries = value

    @property
    def device(self) -> torch.device:
        """The accelerator on which the model is loaded."""
        return self._device

    @staticmethod
    def _replace(target: Union[dict, list], original: Any, replacement: Any) -> None:
        """Recursively replaces original value with the replacement value."""
        if isinstance(target, dict):
            for key, value in target.items():
                if isinstance(value, (dict, list)):
                    Extractor._replace(value, original, replacement)
                else:
                    if value == original:
                        target[key] = replacement
        elif isinstance(target, list):
            for i, value in enumerate(target):
                if isinstance(value, (dict, list)):
                    Extractor._replace(value, original, replacement)
                else:
                    if value == original:
                        target[i] = replacement

    @staticmethod
    def _postprocess(source: dict, json_data: dict) -> dict:
        """
        Post-processes the extracted JSON data.

        Parameters:
            source: A dictionary containing the source fields to check against.
            json_data: A dictionary to post-process.

        Returns:
            A dictionary with post-processed values.
        """
        new_data = deepcopy(json_data)
        Extractor._replace(new_data, None, "")

        # Replace unrecognized event types with "Other"
        if new_data.get("event", "") not in Event.__members__.keys():
            new_data["event"] = Event.Other.value

        # Remove URL if not present in the source fields
        if all(not REGEX_URL.search(value) for value in source.values()):
            new_data["url"] = ""

        # Remove expires and sender if not present in the source fields
        for key in ["expires", "sender"]:
            value = new_data.get(key, "").strip().lower()
            if all(not value in str(src_value).strip().lower() for src_value in source.values()):
                new_data[key] = ""

        return new_data

    @classmethod
    def _initialize_class_attributes(cls) -> None:
        """Initializes class-level attributes."""
        if cls._logger is None:
            cls._logger = get_logger(cls.__name__, level=logging.INFO)

    @classmethod
    def set_logger_level(cls, level: Union[str, int] = logging.INFO) -> None:
        """
        Sets logging level for the class logger.

        Parameters:
            level: Logging level. [DEBUG, INFO, WARNING, ERROR, CRITICAL]
        """
        cls._logger.setLevel(level)
        for handler in cls._logger.handlers:
            handler.setLevel(level)

    # Todo: Check if values exist in the input
    @classmethod
    def _validate_json(cls, source: dict, json_data: dict) -> bool:
        """
        Validates the JSON object against the predefined schema and rules.

        Parameters:
            source: A dictionary containing the source fields to check against.
            json_data: The JSON object to validate.

        Returns:
            bool: True if the JSON object is valid, False otherwise.
        """
        # Check URL validity
        urls = json_data.get("url", "").split(";")
        for url in filter(None, urls):
            if not isinstance(url, str) or not REGEX_URL.match(url):
                raise ValueError(f"Malformed URL: {url}")
            elif all(url.lower().strip() not in value.lower().strip() for value in source.values()):
                raise ValueError(f"URL '{url}' not found in the source fields.")

        try:
            jsonschema.validate(instance=json_data, schema=SCHEMA)
            return True
        except jsonschema.exceptions.ValidationError as e:
            cls._logger.debug(f"Invalid JSON format: {e.message}")
            return False

    @classmethod
    def _get_accelerator(cls) -> torch.device:
        """Returns the available accelerator."""
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
        """Returns model input in the chat markup language format."""
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
           source: Keyword arguments containing the alert message fields (headline, description, and instruction).

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

        # Create model input
        model_inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)
        input_length = model_inputs["input_ids"].shape[1]  # For extracting the generated output

        # Generate output
        for _ in range(1 + self._retries):
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=128,
            )
            output = self._tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0].strip()

            # Extract JSON from output
            if match := self.REGEX_JSON.search(output):
                output = match.group(0)

            # Parse and validate JSON output
            try:
                json_data = json.loads(output)
                json_data = self._postprocess(source, json_data)
                if self._validate_json(source, json_data):
                    return json_data
            except (json.JSONDecodeError, ValueError) as e:
                self._logger.debug(f"Invalid JSON output: {e}")

        raise RuntimeError(f"Failed to generate a valid JSON after {self._retries + 1} tries.")


def main():
    """Example code for extracting information from alert texts using the Extractor class."""
    data_path = DATA_DIR / "finetune" / "finetune_test.csv"
    extractor = Extractor(model="meta-llama/Llama-3.2-1B-Instruct")
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


if __name__ == "__main__":
    main()
