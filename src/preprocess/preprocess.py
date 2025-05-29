import os
from functools import singledispatch
from typing import Union

from tqdm import tqdm

from src.data.download import download_ipaws_data
from src.data.file import read_yaml, read_jsonl_in_batches, write_csv
from src.utils.attribute_filter import get_nested_value, is_valid_alert
from src.utils.paths import DATA_DIR


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
    config = read_yaml(config_file)

    # Delete existing output file
    try:
        os.remove(output_file)
    except FileNotFoundError:
        pass

    # Todo: Add multi-processing
    # Process data
    for batch in tqdm(read_jsonl_in_batches(input_file), desc="Processing"):
        batch = [item for item in batch if is_valid_alert(item)]
        extracted_data = extract(batch, config)
        write_csv(output_file, extracted_data)
