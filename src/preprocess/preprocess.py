import json
import os
from functools import singledispatch
from typing import Iterator, Any, Union

import pandas as pd
import yaml
from tqdm import tqdm

from src.data.download import download_ipaws_data
from src.utils.paths import DATA_DIR


def read_jsonl_in_batches(file_path: str, batch_size=1000) -> Iterator[list[dict[str, Any]]]:
    """Yields batches of parsed JSON objects from a JSONL file."""
    batch = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    batch.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
                if len(batch) == batch_size:
                    yield batch
                    batch = []

        if batch:
            yield batch


def write_jsonl(file_path: str, data: list[dict[str, Any]]) -> None:
    """Writes a list of dictionaries to a JSONL file."""
    with open(file_path, 'a', encoding='utf-8') as a:
        for item in data:
            a.write(json.dumps(item, ensure_ascii=True) + "\n")


def write_csv(file_path: str, data: list[dict[str, Any]]) -> None:
    """Writes a list of dictionaries to a CSV file."""
    df = pd.DataFrame(data)
    header = not os.path.exists(file_path)
    df.to_csv(file_path, mode='a', index=False, encoding='utf-8', header=header)


def get_config(file_path: str) -> dict:
    """Reads a YAML configuration file and returns its content as a dictionary."""
    with open(file_path, 'r') as r:
        return yaml.load(r, Loader=yaml.SafeLoader)


def get_nested_value(data: dict, key_path: str, default: None) -> Any:
    """
    Retrieves a nested value from a dictionary using a dot-separated key path.

    Parameters:
        data: The dictionary to search.
        key_path: A dot-separated string representing the path to the value.
        default: The value to return if the key path does not exist.
    """
    keys = key_path.split('.')
    result = data

    try:
        for key in keys:
            result = result[key]
            if isinstance(result, list) and result:
                result = result[0]
        return result
    except (KeyError, TypeError, IndexError):
        return default


@singledispatch
def extract(data: Union[dict, list[dict]], config: dict) -> Union[dict, list[dict]]:
    """Extracts relevant fields from the data."""
    raise NotImplementedError(f"Unsupported type: {type(data)}")


@extract.register(dict)
def _(data: dict, config: dict) -> dict:
    """Extracts relevant fields from the data."""
    new_data = {}

    for key, value in config.items():
        new_data[value] = get_nested_value(data, key, None)

    return new_data


@extract.register(list)
def _(data: list[dict], config: dict) -> list[dict]:
    """Extracts relevant fields from a list of data."""
    return [extract(item, config) for item in data]


if __name__ == "__main__":
    # Download data
    download_ipaws_data()

    # Set file paths
    config_file = "ner_config.yaml"
    input_file = DATA_DIR / "IpawsArchivedAlerts.jsonl"
    output_file = DATA_DIR / "extracted_data.csv"

    # Load config
    config = get_config(config_file)

    # Delete existing output file
    try:
        os.remove(output_file)
    except FileNotFoundError:
        pass

    # Todo: Add multi-processing
    # Process data
    for batch in tqdm(read_jsonl_in_batches(input_file), desc="Processing"):
        extracted_data = extract(batch, config)
        write_csv(output_file, extracted_data)
