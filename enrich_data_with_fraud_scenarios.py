import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

random.seed(42)
np.random.seed(42)


class FraudScenarioDirector:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.fraud_rate = 0.01
        self.num_fraud_cases = int(len(df) * self.fraud_rate)
        print(f" Loaded dataset: {len(df):,} records")
        print(f" Target fraud rate: {self.fraud_rate * 100}%")
        print(f" Will inject {self.num_fraud_cases:,} fraud scenarios")

    def scenario_impossible_travel(self, user_data: dict) -> list:
        base_time = datetime.now() - timedelta(days=random.randint(1, 30))
        transactions = [
            {
                **user_data,
                "timestamp": base_time,
                "amount": round(random.uniform(5, 20), 2),
                "location": "New York",
                "merchant_category": "coffee_shop",
                "device_id": f"trusted_iphone_{random.randint(1, 15)}",
                "is_fraud": 0,
                "fraud_scenario": None,
            },
            {
                **user_data,
                "timestamp": base_time + timedelta(minutes=5),
                "amount": round(random.uniform(1000, 2500), 2),
                "location": "London",
                "merchant_category": "electronics",
                "device_id": f"unknown_android_{random.randint(1, 100)}",
                "is_fraud": 1,
                "fraud_scenario": "impossible_travel",
            },
        ]
        return transactions

    def scenario_velocity_attack(self, user_data: dict) -> list:
        base_time = datetime.now() - timedelta(days=random.randint(1, 30))
        transactions = []
        transactions.append(
            {
                **user_data,
                "timestamp": base_time - timedelta(hours=2),
                "amount": round(random.uniform(20, 100), 2),
                "location": "Chicago",
                "merchant_category": "grocery",
                "device_id": "trusted_device",
                "is_fraud": 0,
                "fraud_scenario": None,
            }
        )
        for i in range(20):
            transactions.append(
                {
                    **user_data,
                    "timestamp": base_time + timedelta(seconds=i * 6),
                    "amount": round(random.uniform(1, 5), 2),
                    "location": random.choice(["Miami", "Las Vegas", "Atlantic City"]),
                    "merchant_category": random.choice(
                        ["gas_station", "convenience_store"]
                    ),
                    "device_id": f"bot_device_{random.randint(1, 1000)}",
                    "is_fraud": 1,
                    "fraud_scenario": "velocity_attack",
                }
            )
        return transactions

    def scenario_card_testing(self, user_data: dict) -> list:
        base_time = datetime.now() - timedelta(days=random.randint(1, 30))
        transactions = []
        for i in range(5):
            transactions.append(
                {
                    **user_data,
                    "timestamp": base_time + timedelta(minutes=i * 2),
                    "amount": 1.0,
                    "location": random.choice(["Unknown", "VPN_masked"]),
                    "merchant_category": "online_service",
                    "device_id": f"unknown_device_{random.randint(1, 500)}",
                    "is_fraud": 1,
                    "fraud_scenario": "card_testing",
                }
            )
        transactions.append(
            {
                **user_data,
                "timestamp": base_time + timedelta(minutes=15),
                "amount": round(random.uniform(500, 3000), 2),
                "location": random.choice(["Unknown", "VPN_masked"]),
                "merchant_category": "electronics",
                "device_id": f"unknown_device_{random.randint(1, 500)}",
                "is_fraud": 1,
                "fraud_scenario": "card_testing_final",
            }
        )
        return transactions

    def scenario_time_anomaly(self, user_data: dict) -> list:
        transactions = []
        base_date = datetime.now() - timedelta(days=random.randint(1, 30))
        for day in range(5):
            transactions.append(
                {
                    **user_data,
                    "timestamp": base_date.replace(
                        hour=random.randint(9, 20), minute=random.randint(0, 59)
                    )
                    + timedelta(days=day),
                    "amount": round(random.uniform(20, 150), 2),
                    "location": "Home_City",
                    "merchant_category": random.choice(
                        ["grocery", "restaurant", "retail"]
                    ),
                    "device_id": "trusted_device",
                    "is_fraud": 0,
                    "fraud_scenario": None,
                }
            )
        transactions.append(
            {
                **user_data,
                "timestamp": base_date.replace(hour=3, minute=random.randint(0, 59))
                + timedelta(days=6),
                "amount": round(random.uniform(500, 2000), 2),
                "location": "Foreign_Country",
                "merchant_category": "online_crypto",
                "device_id": "unknown_device",
                "is_fraud": 1,
                "fraud_scenario": "time_anomaly",
            }
        )
        return transactions

    def scenario_amount_spike(self, user_data: dict) -> list:
        base_time = datetime.now() - timedelta(days=random.randint(1, 30))
        transactions = []
        for i in range(10):
            transactions.append(
                {
                    **user_data,
                    "timestamp": base_time + timedelta(days=i),
                    "amount": round(random.uniform(30, 70), 2),
                    "location": "Home_City",
                    "merchant_category": random.choice(
                        ["grocery", "gas", "restaurant"]
                    ),
                    "device_id": "trusted_device",
                    "is_fraud": 0,
                    "fraud_scenario": None,
                }
            )
        transactions.append(
            {
                **user_data,
                "timestamp": base_time + timedelta(days=11),
                "amount": round(random.uniform(4000, 6000), 2),
                "location": random.choice(["Unknown", "High_Risk_Country"]),
                "merchant_category": "jewelry",
                "device_id": "unknown_device",
                "is_fraud": 1,
                "fraud_scenario": "amount_spike",
            }
        )
        return transactions

    def scenario_merchant_category_anomaly(self, user_data: dict) -> list:
        base_time = datetime.now() - timedelta(days=random.randint(1, 30))
        transactions = []
        for i in range(8):
            transactions.append(
                {
                    **user_data,
                    "timestamp": base_time + timedelta(days=i),
                    "amount": round(random.uniform(20, 100), 2),
                    "location": "Home_City",
                    "merchant_category": random.choice(["grocery", "gas", "pharmacy"]),
                    "device_id": "trusted_device",
                    "is_fraud": 0,
                    "fraud_scenario": None,
                }
            )
        transactions.append(
            {
                **user_data,
                "timestamp": base_time + timedelta(days=9),
                "amount": round(random.uniform(1000, 5000), 2),
                "location": "Unknown",
                "merchant_category": "crypto_exchange",
                "device_id": "unknown_device",
                "is_fraud": 1,
                "fraud_scenario": "merchant_category_anomaly",
            }
        )
        return transactions

    def scenario_device_switching(self, user_data: dict) -> list:
        base_time = datetime.now() - timedelta(days=random.randint(1, 30))
        transactions = []
        trusted_device = "iPhone_15_Pro_trusted"
        for i in range(7):
            transactions.append(
                {
                    **user_data,
                    "timestamp": base_time + timedelta(days=i),
                    "amount": round(random.uniform(20, 150), 2),
                    "location": "Home_City",
                    "merchant_category": random.choice(
                        ["retail", "restaurant", "entertainment"]
                    ),
                    "device_id": trusted_device,
                    "is_fraud": 0,
                    "fraud_scenario": None,
                }
            )
        transactions.append(
            {
                **user_data,
                "timestamp": base_time + timedelta(days=8),
                "amount": round(random.uniform(800, 2000), 2),
                "location": "Foreign_Country",
                "merchant_category": "electronics",
                "device_id": f"unknown_android_{random.randint(1000, 9999)}",
                "is_fraud": 1,
                "fraud_scenario": "device_switching",
            }
        )
        return transactions

    def inject_fraud_scenarios(self):
        print("\nStarting fraud scenario injection...")
        if "user_id" not in self.df.columns:
            print("No 'user_id' column found. Creating synthetic user groups...")
            self.df["user_id"] = [f"user_{i // 10}" for i in range(len(self.df))]
        unique_users = self.df["user_id"].unique()
        num_fraud_users = int(len(unique_users) * self.fraud_rate)
        fraud_users = np.random.choice(unique_users, num_fraud_users, replace=False)
        print(f" Selected {len(fraud_users):,} users for fraud scenarios")
        scenarios = [
            self.scenario_impossible_travel,
            self.scenario_velocity_attack,
            self.scenario_card_testing,
            self.scenario_time_anomaly,
            self.scenario_amount_spike,
            self.scenario_merchant_category_anomaly,
            self.scenario_device_switching,
        ]
        all_fraud_records = []
        for i, user_id in enumerate(fraud_users):
            scenario_func = random.choice(scenarios)
            user_base = self.df[self.df["user_id"] == user_id].iloc[0].to_dict()
            fraud_transactions = scenario_func(user_base)
            all_fraud_records.extend(fraud_transactions)
            if (i + 1) % 1000 == 0:
                print(
                    f"   Processed {i + 1:,} / {len(fraud_users):,} fraud scenarios..."
                )
        print(f"\nGenerated {len(all_fraud_records):,} fraud transaction records")
        fraud_df = pd.DataFrame(all_fraud_records)
        if "is_fraud" not in self.df.columns:
            self.df["is_fraud"] = 0
        if "fraud_scenario" not in self.df.columns:
            self.df["fraud_scenario"] = None
        enriched_df = pd.concat([self.df, fraud_df], ignore_index=True)
        print(f"\nFinal Dataset Statistics:")
        print(f"   Total records: {len(enriched_df):,}")
        print(
            f"   Fraud cases: {enriched_df['is_fraud'].sum():,} ({enriched_df['is_fraud'].mean() * 100:.2f}%)"
        )
        print(f"   Legitimate cases: {(enriched_df['is_fraud'] == 0).sum():,}")
        if "fraud_scenario" in enriched_df.columns:
            print(f"\nFraud Scenario Distribution:")
            scenario_counts = enriched_df[enriched_df["is_fraud"] == 1][
                "fraud_scenario"
            ].value_counts()
            for scenario, count in scenario_counts.items():
                print(f"   - {scenario}: {count:,}")
        return enriched_df


def main():
    print("=" * 70, flush=True)
    print("FRAUD SCENARIO DIRECTOR - Enriching Synthetic Data", flush=True)
    print("=" * 70, flush=True)
    input_file = Path("data/user_features_1M.parquet")
    output_file = Path("data/user_features_1M_enriched.parquet")
    print(f"\nLoading: {input_file}", flush=True)
    df = pd.read_parquet(input_file)
    print(f" Loaded successfully!", flush=True)
    director = FraudScenarioDirector(df)
    print("\nStarting fraud injection...", flush=True)
    enriched_df = director.inject_fraud_scenarios()
    print(f"\nSaving enriched data to: {output_file}", flush=True)
    enriched_df.to_parquet(output_file, index=False)
    print(f"\nCOMPLETE! Enriched dataset saved.", flush=True)
    print(f"   Original file: {input_file} (preserved)", flush=True)
    print(f"   Enriched file: {output_file} (new)", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
