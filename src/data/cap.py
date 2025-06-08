import copy
import json
from typing import Mapping, Union

import jsonschema
from jsonschema import RefResolver

from src.data.enums import Status, MsgType, Scope, Urgency, Category, Severity, Certainty, ResponseType, Language
from src.utils.file import read_jsonl_in_batches
from src.utils.paths import DATA_DIR

SCHEMA = {
    "title": "Common Alerting Protocol Version 1.2",
    "type": "object",
    # "required": ["identifier", "sender", "sent", "status", "msgType", "scope"],
    "properties": {
        "originalMessage": {"type": ["string", "null"]},

        "identifier": {"type": "string"},
        "sender": {"type": "string"},
        "sent": {
            "type": "string",
            # "pattern": "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}[+-]\\d{2}:\\d{2}$"
        },
        "status": {
            "type": "string",
            "enum": list(Status.__members__.keys())
        },
        "msgType": {
            "type": "string",
            "enum": list(MsgType.__members__.keys())
        },
        "source": {"type": ["string", "null"]},
        "scope": {
            "type": "string",
            "enum": list(Scope.__members__.keys())
        },
        "restriction": {"type": ["string", "null"]},
        "addresses": {"type": ["string", "null"]},
        "code": {"type": "array", "items": {"type": "string"}},
        "note": {"type": ["string", "null"]},
        "searchGeometry": {
            "type": ["object", "null"],
            "required": ["type", "coordinates"],
            "properties": {
                "type": {"type": ["string", "null"]},
                "coordinates": {"type": "array"}
            }
        },
        "incidents": {"type": ["string", "null"]},
        "cogId": {"type": ["number", "null"]},
        "id": {"type": ["string", "null"]},
        "xmlns": {"type": ["string", "null"]},

        "info": {"type": ["array"], "items": {"$ref": "#/$defs/info"}}
    },

    "$defs": {
        "info": {
            "type": "object",
            # "required": ["category", "event", "urgency", "severity", "certainty"],
            "properties": {
                "web": {"type": ["string", "null"], "format": "uri"},
                "area": {"type": "array", "items": {"$ref": "#/$defs/area"}},
                "event": {"type": ["string", "null"]},
                "onset": {"$ref": "#/properties/sent"},
                "expires": {"$ref": "#/properties/sent"},
                "urgency": {
                    "type": "string",
                    "enum": list(Urgency.__members__.keys())
                },
                "category": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(Category.__members__.keys())
                    }
                },
                "headline": {"type": ["string", "null"]},
                "language": {"type": ["string", "null"], "default": "en-US"},
                "severity": {
                    "type": "string",
                    "enum": list(Severity.__members__.keys())
                },
                "certainty": {
                    "type": "string",
                    "enum": list(Certainty.__members__.keys())
                },
                "effective": {"$ref": "#/properties/sent"},
                "eventCode": {"type": "array", "items": {"$ref": "#/$defs/kvPair"}},
                "parameter": {"type": "array", "items": {"$ref": "#/$defs/kvPair"}},
                "senderName": {"type": ["string", "null"]},
                "description": {"type": ["string", "null"]},
                "instruction": {"type": ["string", "null"]},
                "responseType": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(ResponseType.__members__.keys()) + ["None"]
                    }
                }
            }
        },

        "area": {
            "type": "object",
            # "required": ["areaDesc"],
            "properties": {
                "areaDesc": {"type": "string"},
                "polygon": {"type": "object", "items": {"$ref": "#/$defs/polygon"}},
                "circle": {"type": "array", "items": {"type": "string"}},
                "geocode": {"type": "array", "items": {"$ref": "#/$defs/kvPair"}},
                "altitude": {"type": "number"},
                "ceiling": {"type": "number"}
            }
        },

        "polygon": {
            "type": "object",
            "required": ["type", "coordinates"],
            "properties": {
                "type": {"type": "string"},
                "coordinates": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                }
            }
        },

        "kvPair": {
            "type": "object",
            "properties": {
                "valueName": {"type": "string"},
                "value": {"type": "string"},
                "name": {"type": "string"},
                "id": {"type": ["number", "string"]}
            },
            "additionalProperties": True
        }
    },

    "additionalProperties": False
}


class Cap:
    """
    A class representing a CAP (Common Alerting Protocol) content.
    """

    def __init__(self):
        """
        Initializes the Cap instance with a content.

        Parameters:
            content: A dictionary representing the CAP content.
        """
        self.content = Cap.create_empty_cap(SCHEMA)

    def __repr__(self):
        return f"Cap(content={json.dumps(self.content, indent=4)})"

    @staticmethod
    def create_empty_cap(schema) -> dict:
        t = schema.get("type")
        if t == "object" or "object" in t:
            obj = {}
            for k, sub in schema.get("properties", {}).items():
                if "array" in sub.get("type"):
                    obj[k] = []
                elif "object" in sub.get("type"):
                    obj[k] = Cap.create_empty_cap(sub)
                else:
                    obj[k] = None
            return obj
        elif t == "array" or "array" in t:
            return []
        else:
            return None

    @staticmethod
    def create_info() -> dict:
        return Cap.create_empty_cap(SCHEMA.get('$defs', {}).get('info', {}))

    @staticmethod
    def _update_template(original: dict, updates: dict) -> dict:
        """
        Creates a new dictionary by applying updates to the original content.

        Parameters:
            original: Original content dictionary.
            updates: Updates to be applied.

        Returns:
            dict: New updated content dictionary.
        """
        original = copy.deepcopy(original)

        for key, value in updates.items():
            if isinstance(value, Mapping) and key in original:
                original[key] = Cap._update_template(original.get(key, {}) or {}, value)
            else:
                original[key] = value

        return original

    @classmethod
    def from_string(cls, headline: str, description: str, instruction: str, language: Language = Language.en) -> 'Cap':
        """ Creates a Cap instance from string inputs."""
        cap = cls()
        info = {
            "headline": headline,
            "description": description,
            "instruction": instruction,
            "language": language.name
        }
        cap.add_info(info)
        return cap

    @classmethod
    def from_dict(cls, content: dict) -> 'Cap':
        """ Creates a Cap instance from a dictionary conforming to the CAP schema."""
        jsonschema.validate(instance=content, schema=SCHEMA)

        # Create a new Cap instance and set its content
        cap = cls()
        cap.update(content)
        return cap

    def add_info(self, new_info: Union[dict, str]) -> None:
        """
        Adds an info section to the CAP content.

        Parameters:
            new_info: A dictionary or JSON string representing the info section.
        """
        if isinstance(new_info, str):
            new_info = json.loads(new_info)

        # Validate the new_info against the info schema
        resolver = RefResolver.from_schema(SCHEMA)
        jsonschema.validate(instance=new_info, schema=SCHEMA['$defs']['info'], resolver=resolver)

        self.content['info'].append(new_info)

    def update(self, updates: Union[str, dict]) -> None:
        """
        Updates the CAP content with the provided updates.

        Parameters:
            updates: A dictionary containing updates to be applied to the content.
        """
        if isinstance(updates, str):
            updates = json.loads(updates)

        self.content = Cap._update_template(self.content, updates)


def main():
    """Example code for creating a Cap instance."""
    # Example usage with test
    cap = Cap.from_string(
        headline="This is an alert headline.",
        description="This is an alert description.",
        instruction="This is an alert instruction.",
        language=Language.en
    )

    # Example usage with dictionary conforming to the CAP schema
    data_path = DATA_DIR / "IpawsArchivedAlerts.jsonl"
    batch = next(read_jsonl_in_batches(data_path, batch_size=10))
    caps = [Cap.from_dict(alert) for alert in batch]
    for cap in caps:
        print(cap)


if __name__ == "__main__":
    main()
