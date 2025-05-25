import re

import pandas as pd

from src.utils.paths import DATA_DIR

__all__ = ["get_urls"]

REGEX_URL = re.compile(r"(?i)\b(?:https?:\/\/)?(?:www\.)[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:\/\S*)?\b")

def get_urls(input: str) -> list[str]:
    """
    Returns a list of URLs found in the given input.

    Parameters:
        input: The input string from which to extract URLs.

    Returns:
        A list of URLs found in the input string. If no URLs are found, an empty list is returned.
    """
    try:
        urls = REGEX_URL.findall(input)
        return urls if urls else []
    except TypeError as e:
        return []

