import re
import subprocess
import sys

import pandas as pd
import spacy

__all__ = ["get_urls"]

REGEX_URL = re.compile(r"(?i)\b(?:https?:\/\/)?(?:www\.)[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:\/\S*)?\b")
TIME_ENTITIES = {"DATE", "TIME"}
KEYWORDS = {"until", "expiring"}

# Download and load the spaCy model
try:
    MODEL_NAME = "en_core_web_sm"
    nlp = spacy.load(MODEL_NAME)
except OSError:
    print(f"Model '{MODEL_NAME}' not found. Downloading...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", MODEL_NAME])
    nlp = spacy.load(MODEL_NAME)


def get_urls(input: str) -> list[str]:
    """
    Returns a list of URLs found in the given input.

    Parameters:
        input: The input string from which to extract URLs.

    Returns:
        A list of URLs found in the input string. If no URLs are found, an empty list is returned.
    """
    if pd.isna(input) or not str(input).strip():  # Check for NaN and empty string
        return []

    urls = REGEX_URL.findall(input)
    return urls if urls else []


def get_time(input: str) -> str:
    """
    Extracts the expiration time from the input string.

    Parameters:
        input: The input string from which to extract the time.

    Returns:
        The time if found, otherwise an empty string.
    """
    if pd.isna(input) or not str(input).strip():  # Check for NaN and empty string
        return []

    return [ent for ent in nlp(input).ents if ent.label_ in TIME_ENTITIES]
