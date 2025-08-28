import os
from pathlib import Path
import subprocess
import argparse
import zipfile

def main(datasets_dir: str, kaggle_dir: str = "./kaggle"):
    datasets_path = Path(datasets_dir).resolve()
    datasets_path.mkdir(parents=True, exist_ok=True)

    os.environ["KAGGLE_CONFIG_DIR"] = str(Path(kaggle_dir).resolve())

    zip_path = datasets_path / "cat-and-dog.zip"

    # Kaggleからダウンロード
    subprocess.run([
        "kaggle", "datasets", "download", "tongpython/cat-and-dog", "-p", str(datasets_path)
    ], check=True)

    # zip解凍
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(datasets_path)

    # zip削除
    zip_path.unlink()
    print(f"[INFO] Dataset downloaded and extracted to: {datasets_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", type=str, default="./datasets")
    parser.add_argument("--kaggle_dir", type=str, default="./kaggle")
    args = parser.parse_args()

    main(args.datasets_dir, args.kaggle_dir)
