import os
import argparse
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset(api_uri, parent_dir):
    """
    Download the dataset using Kaggle API if it doesn't already exist.

    Args:
        api_uri (str): Kaggle dataset URI.
        parent_dir (str): Parent directory to download the dataset into.
    """
    dataset_dir = os.path.join(parent_dir, 'kaggle_3m/')  # From notebook

    if not os.path.exists(dataset_dir):
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(api_uri, path=parent_dir, unzip=True)
        print(f"Dataset downloaded to {parent_dir}")
    else:
        print(f"Data already exists at {dataset_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LGG MRI Segmentation Dataset")
    parser.add_argument('--api_uri', type=str, default='mateuszbuda/lgg-mri-segmentation', help="Kaggle dataset URI")
    parser.add_argument('--parent_dir', type=str, default='../dataset/', help="Parent directory for dataset")

    args = parser.parse_args()
    download_dataset(args.api_uri, args.parent_dir)