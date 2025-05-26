import json
import os

import jsonschema
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from src.data.enums import Event

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
            "description": "The affected area of the alert message."
        },
        "event": {
            "type": ["string", "null"],
            "description": "The type of event the alert message is about.",
            "enum": list(Event.__members__.keys()) + [None]
        }
    },
    "required": ["url", "expires", "sender", "location", "event"]
}

# Todo: Convert the time format to ISO 8601
PROMPT = """
Generate a JSON object from the provided alert message.
The JSON object should have the following structure:
{schema}
Categorize the alert into one of the following events:
{events}
Use null for any fields that are not present in the text, or for events that are not listed in the allowed events.

###### Example ######
### Alert Message ###
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
### Alert Message End ###

Expected JSON output:
{{{{
    "event": "FlashFloodWarning",
    "expires": "November 23 at 9:00AM CST",
    "location": "La Salle",
    "sender": "NWS Chicago IL",
    "url": "weather.gov/chicago"
}}}}

###### Actual Task ######
### Alert Message ###
headline: 
{{headline}}
description: 
{{description}}
instruction: 
{{instruction}}
### Alert Message End ###

Expected JSON output:
""".format(
    schema="{" + json.dumps({key: value['description'] for key, value in SCHEMA['properties'].items()}, indent=4) + "}",
    events=', '.join(list(Event.__members__.keys()))
)


class Extractor:
    """
    Extractor for extracting relevant information from alert messages using a predefined schema.
    """

    def __init__(self, model: str, schema: str = SCHEMA, prompt: str = PROMPT, retries: int = 3) -> None:
        self._model_name = model
        self._schema = schema
        self._prompt = prompt
        self._retries = retries
        self._device = self._get_accelerator()

        # Load model and tokenizer
        # self._quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            # quantization_config=self._quantization_config,
            device_map=self._device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model,
            padding_side="left",
            # use_fast=True,
        )
        self._generation_config = GenerationConfig(
            eos_token_id=self._model.config.eos_token_id,
        )

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def schema(self) -> dict:
        return self._schema

    @property
    def prompt(self) -> str:
        return self._prompt

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

    # Todo: Check if values exist in the input
    @staticmethod
    def _validate_json(json_data: dict) -> bool:
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
            print(f"Invalid JSON format: {e.message}")
            return False

    @staticmethod
    def _get_accelerator() -> torch.device:
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
        # Check for AMD GPU
        elif torch.backends.hip.is_available():
            device = torch.device('hip')
        # Default to CPU
        else:
            device = torch.device('cpu')

        print(f"Using device: {device}")
        return device

    def extract(self, headline: str = "", description: str = "", instruction: str = "") -> dict:
        """
        Extracts relevant fields from the alert message.

        Parameters:
            headline: The headline of the alert message.
            description: The description of the alert message.
            instruction: The instruction of the alert message.

        Returns:
            A dictionary containing the extracted fields.
        """
        # Format prompt
        prompt = self._prompt.format(
            headline=headline,
            description=description,
            instruction=instruction
        )

        model_inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            # padding=True,
            # truncation=True,
            # max_length=2048
        ).to(self._model.device)
        input_length = model_inputs["input_ids"].shape[1]

        # Generate output
        for _ in range(self._retries):
            generated_ids = self._model.generate(
                **model_inputs,
                # max_new_tokens=512,
                # do_sample=False,
                # temperature=0.1
            )
            output = self._tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0].strip()

            # Todo: Check for ###### Actual Task End ######
            try:
                json_data = json.loads(output)
                if self._validate_json(json_data):
                    return json_data
            except json.JSONDecodeError:
                print(f"Invalid JSON output: {output}")

        raise RuntimeError(f"Failed to generate valid JSON after {self._retries + 1} tries.")


if __name__ == "__main__":
    extractor = Extractor(model="unsloth/Llama-3.2-3B")
    result = extractor.extract(
        headline="Tornado Warning issued August 5 at 3:59PM MDT until August 5 at 4:30PM MDT by NWS Pueblo CO",
        description="""The National Weather Service in Pueblo has issued a

* Tornado Warning for...
Southeastern El Paso County in east central Colorado...
Northwestern Crowley County in southeastern Colorado...
Northeastern Pueblo County in southeastern Colorado...

* Until 430 PM MDT.

* At 359 PM MDT, a severe thunderstorm capable of producing a tornado
was located 11 miles south of Truckton, or 26 miles northeast of
Pueblo Airport, moving southeast at 25 mph.

HAZARD...Tornado.

SOURCE...Radar indicated rotation.

IMPACT...Flying debris will be dangerous to those caught without
shelter. Mobile homes will be damaged or destroyed.
Damage to roofs, windows, and vehicles will occur.  Tree
damage is likely.

* This tornadic thunderstorm will remain over mainly rural areas of
southeastern El Paso, northwestern Crowley and northeastern Pueblo
Counties.""",
        instruction="""TAKE COVER NOW! Move to a basement or an interior room on the lowest
floor of a sturdy building. Avoid windows. If you are outdoors, in a
mobile home, or in a vehicle, move to the closest substantial shelter
and protect yourself from flying debris.""",
    )
    print(result)
