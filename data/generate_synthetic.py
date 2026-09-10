import random

import numpy as np
import pandas as pd
from faker import Faker

def generate_synthetic_data(num_records=1000000):
    fake = Faker()
    data = []
    print(f"Generating {num_records:,} synthetic records...")
    for i in range(num_records):
        record = {
            "user_id": fake.uuid4(),
            "age": random.randint(18, 80),
            "gender": random.choice(["M", "F", "Other"]),
            "transaction_id": fake.uuid4(),
            "amount": np.random.lognormal(3.5, 1.2),
            "transaction_count_7d": random.randint(0, 50),
            "hour_of_day": random.randint(0, 23),
            "latitude": fake.latitude(),
            "device_id": fake.uuid4(),
            "amount_zscore": np.random.randn(),
        }
        data.append(record)
        if (i + 1) % 100000 == 0:
            print(f"Generated {i + 1:,} / {num_records:,} records...")
    df = pd.DataFrame(data)
    df.to_parquet(
        "data/user_features_1M.parquet",
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    print(f"Saved {len(df):,} records with {len(df.columns)} features")
    return df


if __name__ == "__main__":
    generate_synthetic_data()
