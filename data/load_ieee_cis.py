import argparse
import logging
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_PATH = ROOT / "data" / "ieee_cis_merged.parquet"


def download_with_kaggle_api():
    try:
        import kaggle

        logger.info("Downloading IEEE-CIS dataset from Kaggle...")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        kaggle.api.competition_download_files(
            "ieee-fraud-detection", path=str(RAW_DIR), quiet=False
        )
        import zipfile

        zip_path = RAW_DIR / "ieee-fraud-detection.zip"
        if zip_path.exists():
            logger.info(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(RAW_DIR)
            zip_path.unlink()
            logger.info("Download complete")
        else:
            logger.error("Download failed — check Kaggle API credentials")
            sys.exit(1)
    except ImportError:
        logger.error("Kaggle API not installed. Run: pip install kaggle")
        logger.error(
            "Or download manually from: https://www.kaggle.com/competitions/ieee-fraud-detection/data"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Kaggle API error: {e}")
        logger.error("Try manual download instead")
        sys.exit(1)


def load_and_merge():
    tx_path = RAW_DIR / "train_transaction.csv"
    id_path = RAW_DIR / "train_identity.csv"
    if not tx_path.exists() or not id_path.exists():
        logger.error(f"Missing CSV files in {RAW_DIR}/")
        logger.error("Expected: train_transaction.csv, train_identity.csv")
        logger.error(
            "\nDownload from: https://www.kaggle.com/competitions/ieee-fraud-detection/data"
        )
        sys.exit(1)
    logger.info(f"Loading {tx_path.name}...")
    df_tx = pd.read_csv(tx_path)
    logger.info(f"Loaded {len(df_tx):,} transactions, {len(df_tx.columns)} columns")
    logger.info(f"Loading {id_path.name}...")
    df_id = pd.read_csv(id_path)
    logger.info(f"Loaded {len(df_id):,} identity records, {len(df_id.columns)} columns")
    logger.info("Merging transaction + identity...")
    df_merged = df_tx.merge(df_id, on="TransactionID", how="left")
    logger.info(
        f"Merged shape: {len(df_merged):,} rows × {len(df_merged.columns)} columns"
    )
    fraud_count = df_merged["isFraud"].sum()
    fraud_rate = df_merged["isFraud"].mean()
    logger.info(f"Fraud: {fraud_count:,} ({fraud_rate * 100:.2f}%)")
    logger.info(f"Legit: {(df_merged['isFraud'] == 0).sum():,}")
    missing_pct = (
        df_merged.isnull().sum().sum() / (len(df_merged) * len(df_merged.columns)) * 100
    )
    logger.info(f"Missing values: {missing_pct:.1f}%")
    return df_merged


def save_parquet(df: pd.DataFrame):
    logger.info(f"Saving to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, engine="pyarrow", compression="snappy", index=False)
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {size_mb:.1f} MB {OUTPUT_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Load IEEE-CIS fraud detection dataset"
    )
    parser.add_argument(
        "--download", action="store_true", help="Download from Kaggle API"
    )
    args = parser.parse_args()
    logger.info("=" * 60)
    logger.info("IEEE-CIS FRAUD DETECTION DATASET LOADER")
    logger.info("=" * 60)
    if args.download:
        download_with_kaggle_api()
    df = load_and_merge()
    save_parquet(df)
    logger.info("\n" + "=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)
    logger.info(f"\nNext steps:")
    logger.info(f"1. Train XGBoost: python models/xgboost_trainer.py --quick")
    logger.info(f"2. Load Neo4j graph: python graph_db/loader.py")
    logger.info(f"3. Run simulation: python data/simulate_realtime.py --n 100")


if __name__ == "__main__":
    main()
