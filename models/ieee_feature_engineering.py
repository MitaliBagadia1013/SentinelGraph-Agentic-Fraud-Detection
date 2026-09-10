import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")


class IEEEFeatureEngineer:

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns: List[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n" + "=" * 70)
        print("IEEE-CIS FEATURE ENGINEERING PIPELINE")
        print("=" * 70)
        print(f"Input shape: {df.shape}")
        print(f"Fraud rate: {df['isFraud'].mean() * 100:.2f}%")
        df = df.copy()
        df = self._engineer_time_features(df)
        df = self._encode_categorical(df)
        df = self._handle_missing(df)
        df = self._create_interactions(df)
        df = self._drop_noisy_columns(df)
        print(f"\nOutput shape: {df.shape}")
        print("=" * 70)
        return df

    def _engineer_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\nStep 1: Time Features")
        df["TransactionDT_day"] = df["TransactionDT"] // (24 * 3600)
        df["TransactionDT_hour"] = df["TransactionDT"] % (24 * 3600) // 3600
        df["TransactionDT_dow"] = df["TransactionDT_day"] % 7
        df["is_business_hours"] = (
            (df["TransactionDT_hour"] >= 9) & (df["TransactionDT_hour"] <= 17)
        ).astype(int)
        df["is_weekend"] = df["TransactionDT_dow"].isin([5, 6]).astype(int)
        print(f"Created: TransactionDT_day, _hour, _dow, is_business_hours, is_weekend")
        return df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\nStep 2: Categorical Encoding")
        cat_cols = [
            "ProductCD",
            "card4",
            "card6",
            "P_emaildomain",
            "R_emaildomain",
            "DeviceType",
            "DeviceInfo",
        ]
        m_cols = [f"M{i}" for i in range(1, 10)]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna("missing").astype(str)
                le = LabelEncoder()
                df[f"{col}_encoded"] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                print(f"{col}: {len(le.classes_)} categories encoded")
                df = df.drop(columns=[col])
        for col in m_cols:
            if col in df.columns:
                df[col] = df[col].map({"T": 1, "F": 0}).fillna(-1).astype(int)
        print(f"M1-M9: T/F 1/0")
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\nStep 3: Missing Data Handling")
        missing_pct = df.isnull().mean()
        drop_cols = missing_pct[missing_pct > 0.95].index.tolist()
        if drop_cols:
            print(f"Dropping {len(drop_cols)} cols with >95% missing")
            df = df.drop(columns=drop_cols)
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Filling {missing_count:,} missing values with -999")
            df = df.fillna(-999)
        print(f"Remaining columns: {len(df.columns)}")
        return df

    def _create_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\nStep 4: Interaction Features")
        if "TransactionAmt" in df.columns and "card1" in df.columns:
            df["amt_x_card1"] = df["TransactionAmt"] * df["card1"]
            print(f"amt_x_card1")
        if "TransactionDT_hour" in df.columns and "DeviceType_encoded" in df.columns:
            df["hour_x_device"] = df["TransactionDT_hour"] * df["DeviceType_encoded"]
            print(f"hour_x_device")
        if "TransactionAmt" in df.columns and "P_emaildomain_encoded" in df.columns:
            df["amt_x_email"] = df["TransactionAmt"] * df["P_emaildomain_encoded"]
            print(f"amt_x_email")
        return df

    def _drop_noisy_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\nStep 5: Drop Noisy Columns")
        drop_always = ["TransactionID", "TransactionDT"]
        to_drop = [col for col in drop_always if col in df.columns]
        if to_drop:
            print(f"Dropping: {', '.join(to_drop)}")
            df = df.drop(columns=to_drop)
        return df

    def prepare_for_training(
        self, df: pd.DataFrame, add_aggregations: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        print("\nPreparing for Training")
        y = df["isFraud"].values
        X = df.drop(columns=["isFraud"]).copy()
        if add_aggregations and "card1" in df.columns:
            print("add_aggregations=True: Compute card1_fraud_rate AFTER split!")
        X = X.select_dtypes(include=[np.number])
        self.feature_columns = X.columns.tolist()
        print(f"Features: {len(self.feature_columns)}")
        print(f"Samples: {len(X):,}")
        print(f"Fraud rate: {y.mean() * 100:.2f}%")
        return (X, y)

    def add_train_aggregations(
        self, X_train: pd.DataFrame, y_train: np.ndarray
    ) -> pd.DataFrame:
        print("\nAdding Post-Split Aggregations")
        df_temp = X_train.copy()
        df_temp["is_fraud_temp"] = y_train
        if "card1" in df_temp.columns:
            card1_fraud = df_temp.groupby("card1")["is_fraud_temp"].mean()
            X_train["card1_fraud_rate"] = (
                X_train["card1"].map(card1_fraud).fillna(0.035)
            )
            print(f"card1_fraud_rate (computed from train set only)")
        if "addr1" in df_temp.columns:
            addr1_fraud = df_temp.groupby("addr1")["is_fraud_temp"].mean()
            X_train["addr1_fraud_rate"] = (
                X_train["addr1"].map(addr1_fraud).fillna(0.035)
            )
            print(f"addr1_fraud_rate")
        return X_train

    def add_test_aggregations(
        self, X_test: pd.DataFrame, X_train: pd.DataFrame, y_train: np.ndarray
    ) -> pd.DataFrame:
        df_temp = X_train.copy()
        df_temp["is_fraud_temp"] = y_train
        if "card1" in df_temp.columns and "card1" in X_test.columns:
            card1_fraud = df_temp.groupby("card1")["is_fraud_temp"].mean()
            X_test["card1_fraud_rate"] = X_test["card1"].map(card1_fraud).fillna(0.035)
        if "addr1" in df_temp.columns and "addr1" in X_test.columns:
            addr1_fraud = df_temp.groupby("addr1")["is_fraud_temp"].mean()
            X_test["addr1_fraud_rate"] = X_test["addr1"].map(addr1_fraud).fillna(0.035)
        return X_test


if __name__ == "__main__":
    from pathlib import Path

    data_path = Path(__file__).parent.parent / "data" / "ieee_cis_merged.parquet"
    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        print(f"\nRun: python data/load_ieee_cis.py")
        exit(1)
    print("Loading IEEE-CIS dataset...")
    df = pd.read_parquet(data_path)
    fe = IEEEFeatureEngineer()
    df_eng = fe.fit_transform(df)
    X, y = fe.prepare_for_training(df_eng, add_aggregations=False)
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING TEST COMPLETE")
    print("=" * 70)
    print(f"Features: {len(X.columns)}")
    print(f"Samples: {len(X):,}")
    print(f"Fraud rate: {y.mean() * 100:.2f}%")
