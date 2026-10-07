"""Dataset download utility using Kaggle CLI for audiovisual deepfake benchmarks."""
import os
import sys
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Recommended benchmarks for cross-modal audiovisual deepfake research
BENCHMARK_DATASETS: Dict[str, Dict[str, str]] = {
    "fakeavceleb": {
        "slug": "clmentbisaillon/fakeavceleb",
        "description": "Multimodal audiovisual deepfake dataset containing synchronized and manipulated talking heads.",
    },
    "dfdc": {
        "slug": "c/deepfake-detection-challenge",
        "description": "Deepfake Detection Challenge dataset containing extensive facial manipulation video samples.",
    },
    "voxceleb": {
        "slug": "prateekagnihotri/voxceleb-subset",
        "description": "Audio-visual dataset consisting of short clips of human speakers extracted from YouTube.",
    },
}


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API key (kaggle.json or KAGGLE_USERNAME/KEY env vars) is present."""
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True

    home = Path.home()
    kaggle_json = home / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def download_benchmark_dataset(
    dataset_key: str,
    output_dir: str = "data/datasets",
    unzip: bool = True,
) -> bool:
    """Download deepfake dataset using the Kaggle CLI.
    
    Args:
        dataset_key: Key in BENCHMARK_DATASETS or custom Kaggle slug (e.g. 'owner/dataset-name').
        output_dir: Directory where files will be stored.
        unzip: Automatically unzip archive after download.
        
    Returns:
        True if successful.
    """
    if not check_kaggle_credentials():
        logger.error(
            "Kaggle API credentials not found. Please place 'kaggle.json' in '~/.kaggle/' "
            "or set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        )
        print(
            "\n[!] Kaggle credentials not found!\n"
            "To download datasets via Kaggle CLI:\n"
            "1. Visit https://www.kaggle.com/settings -> 'Create New Token'\n"
            "2. Place kaggle.json in ~/.kaggle/ (or set KAGGLE_USERNAME & KAGGLE_KEY)\n"
        )
        return False

    slug = BENCHMARK_DATASETS.get(dataset_key, {}).get("slug", dataset_key)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(out_path)]
    if unzip:
        cmd.append("--unzip")

    logger.info(f"Downloading Kaggle dataset '{slug}' to {out_path}...")
    print(f"Executing: {' '.join(cmd)}")

    try:
        res = subprocess.run(cmd, check=True)
        logger.info(f"Dataset '{slug}' downloaded successfully.")
        return res.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Kaggle CLI failed with exit code {e.returncode}: {e}")
        return False
    except FileNotFoundError:
        logger.error("Kaggle command not found on PATH. Run 'pip install kaggle'.")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download deepfake benchmark datasets via Kaggle CLI.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="fakeavceleb",
        help=f"Dataset slug or key. Available keys: {', '.join(BENCHMARK_DATASETS.keys())}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/datasets",
        help="Destination directory for dataset files.",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Do not extract downloaded zip archive.",
    )

    args = parser.parse_args()
    success = download_benchmark_dataset(
        dataset_key=args.dataset,
        output_dir=args.output_dir,
        unzip=not args.no_unzip,
    )
    sys.exit(0 if success else 1)
