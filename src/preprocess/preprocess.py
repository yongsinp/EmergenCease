import os
from functools import singledispatch
from typing import Union

import pandas as pd
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


def split_dataset(file_path: str, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
                  random_seed: int = 4):
    """
    Splits a csv file into training, validation, and test sets.
    The sum of train_ratio, val_ratio, and test_ratio must be 1.

    Parameters:
        file_path: Path to the input CSV file
        train_ratio: Ratio for training data
        val_ratio: Ratio for validation data
        test_ratio: Ratio for test data
        random_seed: Random seed for reproducibility
    """
    # Check the sum of ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "The sum of ratios must be equal to 1."

    # Read dataset
    df = pd.read_csv(file_path)

    # Shuffle dataset
    shuffled_df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Calculate split points
    n = len(shuffled_df)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    # Split dataframe
    train_df = shuffled_df[:train_end]
    val_df = shuffled_df[train_end:val_end]
    test_df = shuffled_df[val_end:]

    # Generate output paths
    base_path = os.path.splitext(file_path)[0]
    train_path = f"{base_path}_train.csv"
    val_path = f"{base_path}_val.csv"
    test_path = f"{base_path}_test.csv"

    # Save the splits
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Dataset split into: train={len(train_df)} ({train_ratio:.1%}), "
          f"val={len(val_df)} ({val_ratio:.1%}), test={len(test_df)} ({test_ratio:.1%})")

    return train_path, val_path, test_path


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
        batch = [item for item in batch if is_valid_alert(item)]  # Filter valid alerts
        extracted_data = extract(batch, config)  # Extract relevant fields
        write_csv(output_file, extracted_data)

    # Split data
    split_dataset(output_file, 0.8, 0.1, 0.1, 4)
