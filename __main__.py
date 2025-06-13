# -*- coding: utf-8 -*-
import os

import pandas as pd
from tqdm import tqdm

from src.cap_translator.translate import Translator
from src.data.cap import Cap
from src.utils.file import write_file
from src.utils.paths import DATA_DIR


def normalize(input: str) -> str:
    """Normalizes the input string by stripping whitespace and converting to lowercase."""
    return str(input).strip().lower()


def main():
    data_path = os.path.join(DATA_DIR, 'finetune', 'finetune_val.csv')
    data = pd.read_csv(data_path)

    results = []
    translator = Translator(model='meta-llama/Llama-3.2-1B-Instruct', adapter='LoRA-Llama-3.2-1B-Instruct')
    for index, row in tqdm(data.iterrows(), total=len(data), desc="Evaluating"):
        input = {
            'headline': normalize(row['headline']),
            'description': normalize(row['description']),
            'instruction': normalize(row['instruction'])
        }
        cap = Cap.from_string(**input)
        results.append(str(translator.translate(cap)))

    write_file(os.path.join(DATA_DIR, 'finetune', 'finetune_val_translated.txt'), results)


if __name__ == '__main__':
    main()
