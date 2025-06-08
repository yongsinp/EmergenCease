import copy
import os.path
from logging import DEBUG
from typing import Union, Optional

import jsonschema.exceptions

from src.data.cap import Cap
from src.data.enums import Language, Event
from src.extraction.llm import Extractor, SCHEMA
from src.preprocess.preprocess import EVENT_MAP
from src.utils.file import read_jsonl_in_batches, read_json
from src.utils.paths import DATA_DIR, MODEL_DIR


class Translator:
    # Todo: Move path to a config file
    TEMPLATES = read_json("./cap_templates.json")
    LANGUAGES = {language for language in next(iter(TEMPLATES.values())).keys()}

    def __init__(self, model: str, adapter: Optional[str] = None) -> None:
        """
        Initializes the Translator with a specified LLM.

        :param model:
        """
        self.model = model
        self.extractor = Extractor(model, adapter=adapter)
        self.extractor.set_logger_level(DEBUG)

    @staticmethod
    def _get_language(info_block: dict) -> str:
        """
        Extracts the language from the info block.

        Parameters:
            info_block: A dictionary containing the info block.

        Returns:
            The language code, defaulting to 'en' if not specified.
        """
        return info_block.get("language", Language.en.name).split("-")[0]

    def _extract_info(self, en_info: list) -> dict:
        """
        Extracts information from the provided list of info items.

        Parameters:
            en_info: A list of dictionaries containing information items.

        Returns:
            A dictionary with extracted information.
        """
        if not en_info:
            return {}

        headline = en_info.get("headline", "")
        description = en_info.get("description", "")
        instruction = en_info.get("instruction", "")

        return self.extractor.extract(headline=headline, description=description, instruction=instruction)

    def translate(self, source: Union[dict, Cap]) -> Cap:
        """
        Translates the source data into multilingual CAP (Common Alerting Protocol) data.

        Parameters:
            source: A dictionary or JSON string representing the source data.

        Returns:
            A Cap object containing the translated data.
        """
        if isinstance(source, dict):
            try:
                source = Cap.from_dict(source)
            except jsonschema.exceptions.ValidationError as e:
                print(f"Validation error: {e.message}")
                return source

        info = source.content.get("info", [])
        en_info = next((item for item in info if Translator._get_language(item) == Language.en.name), None)

        # If no English info block is found, return the source as is
        if en_info is None:
            return source

        required_info = {key: None for key in SCHEMA['properties'].keys()}

        if event := EVENT_MAP.get(en_info.get("event", None), None):  # Default to Event.Other if not found
            if event == Event.Other:
                return source  # If the event is 'Other', return the source as is
            required_info["event"] = event
        if expires := en_info.get("expires", None):
            required_info["expires"] = expires
        if sender := en_info.get("senderName", None):
            required_info["sender"] = sender
        if url := en_info.get("web", None):
            required_info["url"] = url
        if location := [areaDesc for area in en_info.get("area", []) if (areaDesc := area.get("areaDesc", None))]:
            required_info["location"] = "; ".join(location)

        if any(missing_info := [k for k, v in required_info.items() if v is None]):
            try:
                extracted_info = self._extract_info(en_info)
            except RuntimeError as e:
                print(f"Error extracting information: {e}")
                return source

            if extracted_info["event"] == Event.Other:
                return source

            for key in missing_info:
                required_info[key] = extracted_info[key]

        languages = set(Translator._get_language(item) for item in info)
        missing_languages = set(Translator.LANGUAGES) - languages
        for language in missing_languages:
            if template := self.TEMPLATES.get(required_info["event"], {}).get(language, None):
                template = template.format(
                    SENDING_AGENCY=required_info["sender"],
                    LOCATION=required_info["location"],
                    TIME=required_info["expires"],
                    URL=required_info["url"],
                )

                new_info_block = copy.deepcopy(en_info)
                new_info_block["language"] = language
                new_info_block["headline"] = template
                new_info_block["description"] = None
                new_info_block["instruction"] = None

                source.add_info(new_info_block)

        return source


def main():
    """Example code to demonstrate the usage of the Translator class."""
    model_path = os.path.join(MODEL_DIR, "Llama-3.2-1B-Instruct")
    adapter_path = os.path.join(MODEL_DIR, "LoRA-Llama-3.2-1B-Instruct")
    translator = Translator(model=model_path)

    # Example usage with text
    cap = Cap.from_string(
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
Counties.
        """,
        instruction="""TAKE COVER NOW! Move to a basement or an interior room on the lowest
floor of a sturdy building. Avoid windows. If you are outdoors, in a
mobile home, or in a vehicle, move to the closest substantial shelter
and protect yourself from flying debris.
        """
    )

    print(translator.translate(cap))

    # Example usage with CAP data
    file_path = DATA_DIR / "IpawsArchivedAlerts.jsonl"
    batch = next(read_jsonl_in_batches(file_path, batch_size=10))
    for cap in batch:
        translated_cap = translator.translate(cap)
        print(translated_cap)


if __name__ == "__main__":
    main()
