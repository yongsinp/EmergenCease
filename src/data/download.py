import os
import requests
from pathlib import Path


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

    # Download file
    print(f"Downloading data from {url}...")
    response = requests.get(url)
    response.raise_for_status()

    # Create ancestor directories if they don't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    # Save file
    with open(destination, "wb") as w:
        w.write(response.content)

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

    # Paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"

    download_file(url, data_dir / file_name)


def download_ipaws_data() -> None:
    """Downloads the IPAWS Archived Alerts (JSON) file."""
    download_data("https://www.fema.gov/api/open/v1/IpawsArchivedAlerts.jsonl")


if __name__ == "__main__":
    download_ipaws_data()
