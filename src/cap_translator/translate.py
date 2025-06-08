import copy
import os.path
from typing import Union, Optional

import jsonschema.exceptions

from src.data.cap import Cap
from src.data.enums import Language, Event
from src.extraction.llm import Extractor, SCHEMA
from src.preprocess.preprocess import EVENT_MAP
from src.utils.file import read_jsonl_in_batches, read_json
from src.utils.paths import DATA_DIR, MODEL_DIR


class Translator:
    """
    A class to translate Common Alerting Protocol (CAP) data into multiple languages.
    Events and languages supported by the FCC Wireless Emergency Alert Templates are translated using these templates and an LLM for information extraction.
    The English info block serves as the source for information extraction by the LLM.
    If the CAP data already includes information in a specific language, the original information is preserved.
    """
    TEMPLATES = read_json("./cap_templates.json")  # Todo: Move path to a config file
    LANGUAGES = {language for language in next(iter(TEMPLATES.values())).keys()}

    def __init__(self, model: str, adapter: Optional[str] = None) -> None:
        """
        Initializes the Translator with a specified LLM.

        Parameters:
            model: A local path or Hugging Face model identifier for the model used in extraction.
            adapter: Optional path to a model adapter.
        """
        self.model = model
        self.extractor = Extractor(model, adapter=adapter)

    @staticmethod
    def _get_language(info_block: dict) -> Language:
        """
        Extracts language information from the info block.

        Parameters:
            info_block: The info block to extract from.

        Returns:
            The language code; defaults to 'en' if unspecified.
        """
        return info_block.get("language", Language.en.name).split("-")[0]

    def _extract_info(self, en_info: dict) -> dict:
        """
        Extracts relevant information from the English info block using the LLM.

        Parameters:
            en_info: The English info block containing the alert texts.

        Returns:
            A dictionary containing the extracted information, including event, expires, sender, url, and location.
        """
        if not en_info:
            return {}

        # Extract alert texts to use for extraction
        headline = en_info.get("headline", "")
        description = en_info.get("description", "")
        instruction = en_info.get("instruction", "")

        return self.extractor.extract(headline=headline, description=description, instruction=instruction)

    def translate(self, source: Union[dict, Cap]) -> Cap:
        """
        Translates the source CAP data into a multilingual CAP (Common Alerting Protocol) data.
        Only the events and languages supported by the FCC Wireless Emergency Alert Templates are translated.
        Original information in the source is always prioritized.

        Parameters:
            source: A dictionary or a Cap object containing the CAP data to be translated. Dictionary should conform to the CAP schema.

        Returns:
            A Cap object containing the translated data. It maybe the same as the source if translation is not applicable.
        """
        # Convert source to Cap object
        if isinstance(source, dict):
            try:
                source = Cap.from_dict(source)
            except jsonschema.exceptions.ValidationError as e:
                print(f"Validation error: {e.message}")
                return source

        # Get English info block
        info = source.content.get("info", [])
        en_info = next((item for item in info if Translator._get_language(item) == Language.en.name), None)

        # If no English info block is found, return the source as is
        if en_info is None:
            return source

        # Extract required information from the English info block for preservation
        required_info = {key: None for key in SCHEMA['properties'].keys()}
        if event := EVENT_MAP.get(en_info.get("event", None), None):  # Default to Event.Other if not found
            # Other is not supported by the FCC templates
            if event == Event.Other:
                return source
            required_info["event"] = event
        if expires := en_info.get("expires", None):
            required_info["expires"] = expires
        if sender := en_info.get("senderName", None):
            required_info["sender"] = sender
        if url := en_info.get("web", None):
            required_info["url"] = url
        if location := [areaDesc for area in en_info.get("area", []) if (areaDesc := area.get("areaDesc", None))]:
            required_info["location"] = "; ".join(location)

        # If any required information is missing, try extract it using the LLM
        if any(missing_info := [k for k, v in required_info.items() if v is None]):
            try:
                extracted_info = self._extract_info(en_info)
            except RuntimeError as e:
                print(f"Error extracting information: {e}")
                return source

            # Other is not supported by the FCC templates
            if extracted_info["event"] == Event.Other:
                return source

            # Update required_info with extracted information
            for key in missing_info:
                required_info[key] = extracted_info[key]

        # Add multilingual info blocks using the templates
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
    """Example code for using the Translator class."""
    model_path = os.path.join(MODEL_DIR, "Llama-3.2-1B-Instruct")
    adapter_path = os.path.join(MODEL_DIR, "LoRA-Llama-3.2-1B-Instruct")
    translator = Translator(model=model_path, adapter=adapter_path)

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
    batch = next(read_jsonl_in_batches(file_path, batch_size=1))
    for cap in batch:
        translated_cap = translator.translate(cap)
        print(translated_cap)


if __name__ == "__main__":
    main()
