import argparse
import os
import subprocess
from typing import Optional

from src.utils.paths import MODEL_DIR


def check_model(model: str) -> bool:
    """
    Checks if the model exists in the MODEL_DIR.

    Parameters:
        model: Model to check.

    Returns:
        bool: True if the model exists, False otherwise.
    """
    model_path = os.path.join(MODEL_DIR, model.split("/")[-1])
    return os.path.exists(model_path)


def download_model(model: str, hf_token: Optional[str] = None) -> None:
    """
    Downloads the model from Hugging Face.

    Parameters:
        model: Model to download from Hugging Face.
        hf_token: Hugging Face token for authentication. Searches environment variables if not provided.
    """
    if hf_token is None:
        hf_token = os.getenv('HF_TOKEN', None)

    model_path = os.path.join(MODEL_DIR, model.split("/")[-1])

    # Download the model if it doesn't exist.
    if not os.path.exists(model_path):
        print(f"Downloading {model} model to: {model_path}")
        print("This may take a while.")

        cmd = [
            "tune", "download",
            model,
            "--hf-token", str(hf_token),
            "--output-dir", model_path,
            "--ignore-patterns", "original/consolidated.00.pth",
        ]
        subprocess.run(cmd, check=True)
    elif __name__ == "__main__":
        print(f"Model {model} already exists at {model_path}. Skipping download.")


def main():
    """Main function to download the model."""
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face.")

    parser.add_argument("--model", type=str, default='meta-llama/Llama-3.2-1B-Instruct',
                        help="Model identifier on Hugging Face.")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="Hugging Face token for authentication (optional).", )

    args = parser.parse_args()
    download_model(args.model, args.hf_token)


if __name__ == "__main__":
    main()
