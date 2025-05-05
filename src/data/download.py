import os

import requests

from src.data.paths import DATA_DIR

from tqdm import tqdm


def download_file(url: str, destination: str) -> None:
    """
    Downloads a file from a given URL and saves it to the specified destination.

    Parameters:
        url: The URL of the file to download.
        destination: The local path where the file should be saved.
    """
    if os.path.exists(destination):
        print(f"File already exists at {destination}. Skipping download.")
        return

    # Create ancestor directories if they don't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    # Download file
    print(f"Downloading data from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(destination, "wb") as w:
        with tqdm(
            unit="B",
            unit_scale=True,
            desc="Downloading",
    ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    w.write(chunk)
                    progress_bar.update(len(chunk))

    print(f"File downloaded to {destination}")


def download_data(url: str, file_name: str = None) -> None:
    """
    Downloads a file from a given URL and saves it under data directory.

    Parameters:
        url: The URL of the file to download.
        file_name: The name of the file to be saved. The file name will be derived from the URL if not provided.
    """
    if not file_name:
        file_name = os.path.basename(url)

    download_file(url, DATA_DIR / file_name)


def download_ipaws_data() -> None:
    """Downloads the IPAWS Archived Alerts (JSON) file."""
    download_data("https://www.fema.gov/api/open/v1/IpawsArchivedAlerts.jsonl")


if __name__ == "__main__":
    download_ipaws_data()
