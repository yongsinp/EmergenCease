import json
import os
from typing import Any, Iterator

import pandas as pd
import yaml


def read_json(file_path: str) -> Any:
    """Reads a JSON file and returns its content."""
    with open(file_path, 'r', encoding='utf-8') as r:
        return json.load(r)


def read_jsonl_in_batches(file_path: str, batch_size=1000) -> Iterator[list[dict[str, Any]]]:
    """
    Yields batches of parsed JSON objects from a JSONL file.

    Parameters:
        file_path: Path to the JSONL file.
        batch_size: Number of JSON objects to include in each batch.

    Yields:
        A batch of parsed JSON objects.
    """
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


def read_csv_in_batches(file_path: str, batch_size: int = 10):
    """
    Reads a CSV file in batches.

    Parameters:
        file_path: Path to the CSV file.
        batch_size: Number of rows per batch.

    Yields:
        A DataFrame containing a batch of rows from the CSV file.
    """
    for chunk in pd.read_csv(file_path, chunksize=batch_size):
        yield chunk


def read_yaml(file_path: str) -> dict:
    """
    Reads a YAML configuration file and returns its content as a dictionary.

    Parameters:
        file_path: Path to the YAML file.

    Returns:
        A dictionary containing the YAML file's content.
    """
    with open(file_path, 'r') as r:
        return yaml.load(r, Loader=yaml.SafeLoader)


def write_json(file_path: str, data: dict[str, Any], mode: str = 'w') -> None:
    """
    Writes a dictionary to a JSON file.

    Parameters:
        file_path: Path to the JSON file.
        data: The dictionary to write to the file.
        mode: File mode. ['w': write, 'a': append]
    """
    with open(file_path, mode, encoding='utf-8') as w:
        json.dump(data, w, ensure_ascii=True, indent=4)


def write_jsonl(file_path: str, data: list[dict[str, Any]], mode: str = 'a') -> None:
    """
    Writes a list of dictionaries to a JSONL file.

    Parameters:
        file_path: Path to the JSONL file.
        data: A list of dictionaries to write to the file.
        mode: File mode. ['w': write, 'a': append]
    """
    with open(file_path, mode, encoding='utf-8') as w:
        for item in data:
            w.write(json.dumps(item, ensure_ascii=True) + "\n")


def write_csv(file_path: str, data: list[dict[str, Any]], mode: str = 'a') -> None:
    """
    Writes a list of dictionaries to a CSV file.

    Parameters:
        file_path: Path to the CSV file.
        data: A list of dictionaries to write to the file.
        mode: File mode. ['w': write, 'a': append]
    """
    df = pd.DataFrame(data)
    header = not os.path.exists(file_path)
    df.to_csv(file_path, mode=mode, index=False, encoding='utf-8', header=header)
