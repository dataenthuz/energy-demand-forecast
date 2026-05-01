"""
Download the PJM Hourly Energy Consumption dataset from Kaggle.

Setup:
  1. pip install kaggle
  2. kaggle.com > Account > Settings > API > Create New API Token
  3. Save kaggle.json to ~/.kaggle/kaggle.json (Windows: C:\\Users\\YOU\\.kaggle\\kaggle.json)
"""

import subprocess
import sys
from pathlib import Path

DATASET = "robikscube/hourly-energy-consumption"
DATA_DIR = Path(__file__).parent
EXPECTED = DATA_DIR / "PJME_hourly.csv"


def main():
    if EXPECTED.exists():
        size_mb = EXPECTED.stat().st_size / 1e6
        print(f"Already downloaded: {EXPECTED} ({size_mb:.0f} MB)")
        return

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("Kaggle credentials not found.")
        print(f"  Expected: {kaggle_json}")
        print("  Steps:")
        print("    pip install kaggle")
        print("    Go to kaggle.com > Account > Settings > API > Create New API Token")
        print("    Save kaggle.json to ~/.kaggle/kaggle.json")
        sys.exit(1)

    print(f"Downloading {DATASET} to {DATA_DIR} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(DATA_DIR), "--unzip"],
        check=True,
    )
    if EXPECTED.exists():
        size_mb = EXPECTED.stat().st_size / 1e6
        print(f"Done. {EXPECTED} ({size_mb:.0f} MB)")
    else:
        # dataset has multiple files - list what was downloaded
        files = list(DATA_DIR.glob("*.csv"))
        print(f"Downloaded {len(files)} CSV files: {[f.name for f in files]}")
        print("Looking for PJME_hourly.csv - rename if needed.")


if __name__ == "__main__":
    main()
