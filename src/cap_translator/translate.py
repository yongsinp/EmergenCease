import argparse
import copy
import json
import os.path
from typing import Union, Optional

import jsonschema.exceptions
from markdown_it.rules_block.table import headerLineRe

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
    TEMPLATES = read_json(os.path.join(os.path.dirname(__file__), "cap_templates.json"))  # Todo: Move path to a config file
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

    parser = argparse.ArgumentParser(description="Script for evaluating LLM extractors.")

    # Model arguments
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B-Instruct',
                        help='Model to use for extraction (default: meta-llama/Llama-3.2-1B-Instruct)')
    parser.add_argument('--adapter', type=str, default='LoRA-Llama-3.2-1B-Instruct',
                        help='Path to the model adapter (default: LoRA-Llama-3.2-1B-Instruct)')

    # Inputs
    parser.add_argument('--headline', type=str, default=None,
                        help='Headline of the alert')
    parser.add_argument('--description', type=str, default=None,
                        help='Description of the alert')
    parser.add_argument('--instruction', type=str, default=None,
                        help='Instruction of the alert')
    parser.add_argument('--cap', type=str, default=None,
                        help='Input in CAP format (JSON). If provided, it will be used instead of the headline, description, and instruction.')

    args = parser.parse_args()

    translator = Translator(model=args.model, adapter=args.adapter)

    if any((args.headline, args.description, args.instruction, args.cap)):
        if args.cap:
            try:
                cap = Cap.from_dict(json.loads(args.cap))
            except jsonschema.exceptions.ValidationError as e:
                print(f"Validation error: {e.message}")
                return
        else:
            cap = Cap.from_string(
                headline=args.headline,
                description=args.description,
                instruction=args.instruction
            )
    # Example usage with text
    else:
        input = json.loads(r"""{"originalMessage": "", "identifier": "urn:oid:2.49.0.1.840.0.170b3c727eba78980b24f1eea430ca1ac88e163f.001.1.IPAWS", "sender": "w-nws.webmaster@noaa.gov", "sent": "2023-08-05T21:59:00.000Z", "status": "Actual", "msgType": "Alert", "source": null, "scope": "Public", "restriction": null, "addresses": null, "code": ["IPAWSv1.0"], "note": null, "searchGeometry": {"type": "MultiPolygon", "coordinates": [[[[-104.05, 38.61], [-104.13, 38.65], [-104.32, 38.59], [-104.02, 38.28], [-103.73, 38.44], [-103.88, 38.52], [-104.05, 38.52], [-104.05, 38.61]]], [[[-104.05, 38.61], [-104.13, 38.65], [-104.32, 38.59], [-104.02, 38.28], [-103.73, 38.44], [-103.88, 38.52], [-104.05, 38.52], [-104.05, 38.61]]]]}, "incidents": null, "cogId": 200032, "id": "186249c4-0b60-4667-8258-404b10ea5967", "xmlns": "urn:oasis:names:tc:emergency:cap:1.2", "info": [{"web": "http://www.weather.gov", "area": [{"geocode": [{"id": 675, "name": "SAME", "value": "008025"}, {"id": 3125, "name": "SAME", "value": "008041"}, {"id": 773, "name": "SAME", "value": "008101"}, {"id": 535, "name": "UGC", "value": "COC025"}, {"id": 46218, "name": "UGC", "value": "COC041"}, {"id": 584, "name": "UGC", "value": "COC101"}], "polygon": {"type": "Polygon", "coordinates": [[[-104.05, 38.61], [-104.05, 38.52], [-103.88, 38.52], [-103.73, 38.44], [-104.02, 38.28], [-104.32, 38.59], [-104.13, 38.65], [-104.05, 38.61]]]}, "areaDesc": "Crowley, CO; El Paso, CO; Pueblo, CO"}], "event": "Tornado Warning", "onset": "2023-08-05T15:59:00-06:00", "expires": "2023-08-05T16:30:00-06:00", "urgency": "Immediate", "category": ["Met"], "headline": "Tornado Warning issued August 5 at 3:59PM MDT until August 5 at 4:30PM MDT by NWS Pueblo CO", "language": "en-US", "severity": "Extreme", "certainty": "Observed", "effective": "2023-08-05T15:59:00-06:00", "eventCode": [{"id": 11, "name": "SAME", "value": "TOR"}, {"id": 1621, "name": "NationalWeatherService", "value": "TOW"}], "parameter": [{"name": "AWIPSidentifier", "value": "TORPUB"}, {"name": "WMOidentifier", "value": "WFUS55 KPUB 052159"}, {"name": "eventMotionDescription", "value": "2023-08-05T21:59:00Z...storm...313DEG...20KT...38.57,-104.15"}, {"name": "maxHailSize", "value": "Up to .75"}, {"name": "tornadoDetection", "value": "RADAR INDICATED"}, {"name": "BLOCKCHANNEL", "value": "EAS"}, {"name": "BLOCKCHANNEL", "value": "NWEM"}, {"name": "EAS-ORG", "value": "WXR"}, {"name": "VTEC", "value": "/O.NEW.KPUB.TO.W.0085.230805T2159Z-230805T2230Z/"}, {"name": "eventEndingTime", "value": "2023-08-05T16:30:00-06:00"}, {"name": "WEAHandling", "value": "Imminent Threat"}, {"name": "CMAMtext", "value": "NWS: TORNADO WARNING in this area til 4:30 PM MDT. Take shelter now. Check media."}, {"name": "CMAMlongtext", "value": "National Weather Service: TORNADO WARNING in this area until 4:30 PM MDT. Take shelter now in a basement or an interior room on the lowest floor of a sturdy building. If you are outdoors, in a mobile home, or in a vehicle, move to the closest substantial shelter and protect yourself from flying debris. Check media."}], "senderName": "NWS Pueblo CO", "description": "The National Weather Service in Pueblo has issued a\n\n* Tornado Warning for...\nSoutheastern El Paso County in east central Colorado...\nNorthwestern Crowley County in southeastern Colorado...\nNortheastern Pueblo County in southeastern Colorado...\n\n* Until 430 PM MDT.\n\n* At 359 PM MDT, a severe thunderstorm capable of producing a tornado\nwas located 11 miles south of Truckton, or 26 miles northeast of\nPueblo Airport, moving southeast at 25 mph.\n\nHAZARD...Tornado.\n\nSOURCE...Radar indicated rotation.\n\nIMPACT...Flying debris will be dangerous to those caught without\nshelter. Mobile homes will be damaged or destroyed.\nDamage to roofs, windows, and vehicles will occur.  Tree\ndamage is likely.\n\n* This tornadic thunderstorm will remain over mainly rural areas of\nsoutheastern El Paso, northwestern Crowley and northeastern Pueblo\nCounties.\n", "instruction": "TAKE COVER NOW! Move to a basement or an interior room on the lowest\nfloor of a sturdy building. Avoid windows. If you are outdoors, in a\nmobile home, or in a vehicle, move to the closest substantial shelter\nand protect yourself from flying debris.", "responseType": ["Shelter"]}, {"web": "http://www.weather.gov", "area": [{"geocode": [{"id": 675, "name": "SAME", "value": "008025"}, {"id": 3125, "name": "SAME", "value": "008041"}, {"id": 773, "name": "SAME", "value": "008101"}, {"id": 535, "name": "UGC", "value": "COC025"}, {"id": 46218, "name": "UGC", "value": "COC041"}, {"id": 584, "name": "UGC", "value": "COC101"}], "polygon": {"type": "Polygon", "coordinates": [[[-104.05, 38.61], [-104.05, 38.52], [-103.88, 38.52], [-103.73, 38.44], [-104.02, 38.28], [-104.32, 38.59], [-104.13, 38.65], [-104.05, 38.61]]]}, "areaDesc": "Crowley, CO; El Paso, CO; Pueblo, CO"}], "event": "Tornado Warning", "onset": "2023-08-05T15:59:00-06:00", "expires": "2023-08-05T16:30:00-06:00", "urgency": "Immediate", "category": ["Met"], "headline": "Tornado Warning issued August 5 at 3:59PM MDT until August 5 at 4:30PM MDT by NWS Pueblo CO", "language": "es-US", "severity": "Extreme", "certainty": "Observed", "effective": "2023-08-05T15:59:00-06:00", "eventCode": [{"id": 11, "name": "SAME", "value": "TOR"}, {"id": 1621, "name": "NationalWeatherService", "value": "TOW"}], "parameter": [{"name": "AWIPSidentifier", "value": "TORPUB"}, {"name": "WMOidentifier", "value": "WFUS55 KPUB 052159"}, {"name": "eventMotionDescription", "value": "2023-08-05T21:59:00Z...storm...313DEG...20KT...38.57,-104.15"}, {"name": "maxHailSize", "value": "Up to .75"}, {"name": "tornadoDetection", "value": "RADAR INDICATED"}, {"name": "BLOCKCHANNEL", "value": "EAS"}, {"name": "BLOCKCHANNEL", "value": "NWEM"}, {"name": "EAS-ORG", "value": "WXR"}, {"name": "VTEC", "value": "/O.NEW.KPUB.TO.W.0085.230805T2159Z-230805T2230Z/"}, {"name": "eventEndingTime", "value": "2023-08-05T16:30:00-06:00"}, {"name": "WEAHandling", "value": "Imminent Threat"}, {"name": "CMAMtext", "value": "SNM:AVISO DE TORNADO hasta las 4:30 PM MDT. Refugiese ahora. Consulte medios informativos"}, {"name": "CMAMlongtext", "value": "Servicio Nacional de Meteorologia: AVISO DE TORNADO en efecto hasta las 4:30 PM MDT. Busque refugio ahora en un sotano o en el interior de un cuarto en el nivel mas bajo de un edificio seguro. Si esta al aire libre, en una casa movil, o en un vehiculo, debe ir al edificio seguro mas cercano y protegerse de proyectiles. Consulte los medios informativos."}], "senderName": "NWS Pueblo CO", "description": "The National Weather Service in Pueblo has issued a\n\n* Tornado Warning for...\nSoutheastern El Paso County in east central Colorado...\nNorthwestern Crowley County in southeastern Colorado...\nNortheastern Pueblo County in southeastern Colorado...\n\n* Until 430 PM MDT.\n\n* At 359 PM MDT, a severe thunderstorm capable of producing a tornado\nwas located 11 miles south of Truckton, or 26 miles northeast of\nPueblo Airport, moving southeast at 25 mph.\n\nHAZARD...Tornado.\n\nSOURCE...Radar indicated rotation.\n\nIMPACT...Flying debris will be dangerous to those caught without\nshelter. Mobile homes will be damaged or destroyed.\nDamage to roofs, windows, and vehicles will occur.  Tree\ndamage is likely.\n\n* This tornadic thunderstorm will remain over mainly rural areas of\nsoutheastern El Paso, northwestern Crowley and northeastern Pueblo\nCounties.\n", "instruction": "TAKE COVER NOW! Move to a basement or an interior room on the lowest\nfloor of a sturdy building. Avoid windows. If you are outdoors, in a\nmobile home, or in a vehicle, move to the closest substantial shelter\nand protect yourself from flying debris.", "responseType": ["Shelter"]}]}""")
        cap = Cap.from_dict(input)

    print(translator.translate(cap))


if __name__ == "__main__":
    main()
