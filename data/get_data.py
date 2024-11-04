import os
from pathlib import Path


def main() -> None:
    dataset_name = "marquis03/plants-classification"
    download_path = "data/plants"
    Path(download_path).mkdir(exist_ok=True)
    os.system(f"kaggle datasets download -d {dataset_name} -p {download_path} --unzip")


if __name__ == "__main__":
    main()
