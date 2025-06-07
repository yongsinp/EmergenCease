import json
import os
from typing import Any, Iterator

import pandas as pd
import yaml


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


def read_yaml(file_path: str) -> dict:
    """Reads a YAML configuration file and returns its content as a dictionary."""
    with open(file_path, 'r') as r:
        return yaml.load(r, Loader=yaml.SafeLoader)


def write_json(file_path: str, data: dict[str, Any], mode: str = 'w') -> None:
    """Writes a dictionary to a JSON file."""
    with open(file_path, mode, encoding='utf-8') as w:
        json.dump(data, w, ensure_ascii=True, indent=4)


def write_jsonl(file_path: str, data: list[dict[str, Any]], mode: str = 'a') -> None:
    """Writes a list of dictionaries to a JSONL file."""
    with open(file_path, mode, encoding='utf-8') as w:
        for item in data:
            w.write(json.dumps(item, ensure_ascii=True) + "\n")


def write_csv(file_path: str, data: list[dict[str, Any]], mode: str = 'a') -> None:
    """Writes a list of dictionaries to a CSV file."""
    df = pd.DataFrame(data)
    header = not os.path.exists(file_path)
    df.to_csv(file_path, mode=mode, index=False, encoding='utf-8', header=header)
