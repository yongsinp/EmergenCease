import re
import subprocess
import sys

import pandas as pd
import spacy

__all__ = ["get_urls"]

# URL extraction
REGEX_URL = re.compile(r"(?i)\b(?:https?:\/\/)?(?:www\.)?[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:\/\S*)?\b")

# Date time extraction
MODEL_NAME = "en_core_web_sm"
TIME_ENTITIES = {"DATE", "TIME"}
KEYWORDS = {"until", "expiring"}

# Download and load the spaCy model
try:
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


def get_datetime(input: str) -> str:
    """
    Extracts the expiration time from the input string.

    Parameters:
        input: The input string from which to extract the time.

    Returns:
        The time if found, otherwise an empty string.
    """
    if pd.isna(input) or not str(input).strip():  # Check for NaN and empty string
        return ""

    return [ent for ent in nlp(input).ents if ent.label_ in TIME_ENTITIES]


def get_sending_agency(input: str) -> str:
    """
    Extracts the sending agency from the input string.

    Parameters:
        input: The input string from which to extract the sending agency.

    Returns:
        The sending agency if found, otherwise an empty string.
    """
    if pd.isna(input) or not str(input).strip():  # Check for NaN and empty string
        return ""

    return [ent.text for ent in nlp(input).ents if ent.label_ == "ORG"]


def get_location(input: str) -> str:
    """
    Extracts the location from the input string.

    Parameters:
        input: The input string from which to extract the location.

    Returns:
        The location if found, otherwise an empty string.
    """
    if pd.isna(input) or not str(input).strip():  # Check for NaN and empty string
        return []

    # Location information is too critical to be extracted without proper care - but the mismatch between the 'location' column and the information in 'headline', 'description', and 'instruction' columns is too high.
    return [ent.text for ent in nlp(input).ents if ent.label_ == "LOC"]


def main():
    """Example usage of the extraction functions."""
    sample_text = """Hurricane Warning issued October 8 at 5:42PM EDT by NWS Tallahassee FL

A Hurricane Warning means hurricane-force winds are expected
somewhere within this area within the next 36 hours

* LOCATIONS AFFECTED
- Bonifay
- Ponce De Leon

* WIND
- LATEST LOCAL FORECAST: Equivalent Tropical Storm force wind
- Peak Wind Forecast: 35-45 mph with gusts to 65 mph
- Window for Tropical Storm force winds: Wednesday morning
until early Thursday morning

- POTENTIAL THREAT TO LIFE AND PROPERTY: Potential for wind
greater than 110 mph
- The wind threat has increased from the previous assessment.
- PLAN: Plan for extreme wind of equivalent CAT 3 hurricane
force or higher due to possible forecast changes in track,
size, or intensity.
- PREPARE: Efforts to protect life and property should now be
underway. Prepare for catastrophic wind damage.
- ACT: Act now to complete preparations before the wind
becomes hazardous.

- POTENTIAL IMPACTS: Devastating to Catastrophic
- Structural damage to sturdy buildings, some with complete
roof and wall failures. Complete destruction of mobile
homes. Damage greatly accentuated by large airborne
projectiles. Locations may be uninhabitable for weeks or
months.
- Numerous large trees snapped or uprooted along with fences
and roadway signs blown over.
- Many roads impassable from large debris, and more within
urban or heavily wooded places. Many bridges, causeways,
and access routes impassable.
- Widespread power and communications outages.

* FLOODING RAIN
- LATEST LOCAL FORECAST:
- Peak Rainfall Amounts: Additional 2-4 inches, with locally
higher amounts

- POTENTIAL THREAT TO LIFE AND PROPERTY: Potential for localized
flooding rain
- The flooding rain threat has remained nearly steady from
the previous assessment.
- PLAN: Emergency plans should include the potential for
localized flooding from heavy rain.
- PREPARE: Consider protective actions if you are in an area
vulnerable to flooding.
- ACT: Heed any flood watches and warnings.

- POTENTIAL IMPACTS: Limited
- Localized rainfall flooding may prompt a few evacuations.
- Rivers and tributaries may quickly rise with swifter
currents. Small streams, creeks, and ditches may become
swollen and overflow in spots.
- Flood waters can enter a few structures, especially in
usually vulnerable spots. A few places where rapid ponding
of water occurs at underpasses, low-lying spots, and poor
drainage areas. Several storm drains and retention ponds
become near-full and begin to overflow. Some brief road and
bridge closures.

* TORNADO
- LATEST LOCAL FORECAST:
- Situation is unfavorable for tornadoes

- POTENTIAL THREAT TO LIFE AND PROPERTY: Tornadoes not expected
- The tornado threat has remained nearly steady from the
previous assessment.
- PLAN: Tornadoes are not expected. Showers and thunderstorms
with gusty winds may still occur.
- PREPARE: Little to no preparations needed to protect
against tornadoes at this time. Keep informed of the latest
tornado situation.
- ACT: Listen for changes in the forecast.

- POTENTIAL IMPACTS: Little to None
- Little to no potential impacts from tornadoes.

* FOR MORE INFORMATION:
- Local Weather Conditions and Forecasts: NWS Tallahassee
- https://www.weather.gov/tallahassee
- Information from the Florida Division of Emergency Management
- https://www.floridadisaster.org"""

    print("Extracted URLs:", get_urls(sample_text))
    print("Extracted Date/Time:", get_datetime(sample_text))
    print("Extracted Sending Agency:", get_sending_agency(sample_text))
    print("Extracted Location:", get_location(sample_text))


if __name__ == "__main__":
    main()
