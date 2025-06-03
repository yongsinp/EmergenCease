"""Converts csv files to JSON files for torchtune."""

import pandas as pd

from src.data.file import write_json
from src.extraction.llm import USER_PROMPT
from src.utils.paths import DATA_DIR


def main():
    files = [
        "finetune_train.csv",
        "finetune_val.csv",
        "finetune_test.csv",
    ]

    for file in files:
        input_csv = DATA_DIR / "finetune" / file
        df = pd.read_csv(input_csv)
        df = df.fillna("").replace({None: ""})

        data = []
        for _, row in df.iterrows():
            headline = row["headline"]
            description = row["description"]
            instruction = row["instruction"]
            user_prompt = USER_PROMPT.format(
                headline=headline,
                description=description,
                instruction=instruction,
            )
            json_output = {
                "url": row["url"],
                "event": row["event"],
                "expires": row["time"],
                "location": row["location"],
                "sender": row["sending_agency"]
            }

            data.append(
                {
                    "user_prompt": user_prompt,
                    "json_output": json_output
                }
            )

        output_json = DATA_DIR / "finetune" / file.replace(".csv", ".json")
        write_json(output_json, data)


if __name__ == "__main__":
    main()
