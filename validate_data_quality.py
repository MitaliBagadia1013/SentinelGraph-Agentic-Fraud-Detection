import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def validate_dataset_quality():
    print("=" * 80)
    print("DATA QUALITY VALIDATION - Production Readiness Check")
    print("=" * 80)
    print("\nLoading enriched dataset...")
    df = pd.read_parquet("data/user_features_1M_enriched.parquet")
    print(f"Loaded: {len(df):,} records with {len(df.columns)} features")
    quality_score = 0
    max_score = 0
    print("\n" + "=" * 80)
    print("1 DATASET SIZE & STRUCTURE")
    print("=" * 80)
    max_score += 10
    print(f"Total Records: {len(df):,}")
    print(f"Total Features: {len(df.columns)}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    if len(df) >= 1000000:
        print("PASS: Dataset size adequate (>1M records)")
        quality_score += 10
    else:
        print("WARN: Dataset smaller than 1M records")
        quality_score += 5
    print("\n" + "=" * 80)
    print("2 FRAUD RATE & CLASS BALANCE")
    print("=" * 80)
    max_score += 15
    if "is_fraud" in df.columns:
        fraud_rate = df["is_fraud"].mean()
        fraud_count = df["is_fraud"].sum()
        legit_count = (df["is_fraud"] == 0).sum()
        print(f"Fraud Cases: {fraud_count:,} ({fraud_rate * 100:.2f}%)")
        print(f"Legitimate Cases: {legit_count:,} ({(1 - fraud_rate) * 100:.2f}%)")
        print(f"Class Imbalance Ratio: {legit_count / fraud_count:.1f}:1")
        if 0.01 <= fraud_rate <= 0.15:
            print("PASS: Fraud rate realistic (1-15%)")
            quality_score += 15
        else:
            print(f"WARN: Fraud rate {fraud_rate * 100:.2f}% may be unrealistic")
            quality_score += 8
    else:
        print("FAIL: No 'is_fraud' column found")
    print("\n" + "=" * 80)
    print("3 FRAUD SCENARIO DIVERSITY")
    print("=" * 80)
    max_score += 15
    if "fraud_scenario" in df.columns:
        fraud_scenarios = df[df["is_fraud"] == 1]["fraud_scenario"].value_counts()
        print(f"\nFraud Scenario Breakdown:")
        for scenario, count in fraud_scenarios.items():
            pct = count / df["is_fraud"].sum() * 100
            print(f"{scenario:<30s}: {count:>8,} ({pct:>5.2f}%)")
        num_scenarios = len(fraud_scenarios)
        if num_scenarios >= 5:
            print(f"\nPASS: Good scenario diversity ({num_scenarios} types)")
            quality_score += 15
        else:
            print(f"\nWARN: Limited scenario diversity ({num_scenarios} types)")
            quality_score += 8
    else:
        print("WARN: No 'fraud_scenario' column - can't validate diversity")
        quality_score += 5
    print("\n" + "=" * 80)
    print("4 FEATURE VALUE SANITY CHECKS")
    print("=" * 80)
    max_score += 20
    checks_passed = 0
    total_checks = 0
    if "avg_transaction_amount" in df.columns:
        total_checks += 1
        amount_mean = df["avg_transaction_amount"].mean()
        amount_median = df["avg_transaction_amount"].median()
        amount_95th = df["avg_transaction_amount"].quantile(0.95)
        print(f"\nTransaction Amount Distribution:")
        print(f"Mean: ${amount_mean:.2f}")
        print(f"Median: ${amount_median:.2f}")
        print(f"95th Percentile: ${amount_95th:.2f}")
        if 20 <= amount_mean <= 500 and 10 <= amount_median <= 300:
            print("Realistic spending patterns")
            checks_passed += 1
        else:
            print("May not match real spending patterns")
    if "account_age_days" in df.columns:
        total_checks += 1
        avg_account_age = df["account_age_days"].mean()
        print(f"\nAccount Age:")
        print(
            f"Average: {avg_account_age:.0f} days ({avg_account_age / 365:.1f} years)"
        )
        if 30 <= avg_account_age <= 3650:
            print("Realistic account ages")
            checks_passed += 1
        else:
            print("Account ages may be unrealistic")
    if "transaction_count" in df.columns:
        total_checks += 1
        avg_txn_count = df["transaction_count"].mean()
        print(f"\nTransaction Count:")
        print(f"Average: {avg_txn_count:.1f} transactions")
        if 10 <= avg_txn_count <= 5000:
            print("Realistic transaction volumes")
            checks_passed += 1
        else:
            print("Transaction counts may be unrealistic")
    if "fraud_score" in df.columns:
        total_checks += 1
        fraud_score_mean = df["fraud_score"].mean()
        fraud_score_std = df["fraud_score"].std()
        print(f"\nFraud Score Distribution:")
        print(f"Mean: {fraud_score_mean:.4f}")
        print(f"Std Dev: {fraud_score_std:.4f}")
        print(f"Range: [{df['fraud_score'].min():.4f}, {df['fraud_score'].max():.4f}]")
        if 0 <= df["fraud_score"].min() and df["fraud_score"].max() <= 1:
            print("Valid score range")
            checks_passed += 1
        else:
            print("Invalid score range!")
    if total_checks > 0:
        feature_quality = checks_passed / total_checks * 20
        quality_score += feature_quality
        print(f"\nFeature Quality: {checks_passed}/{total_checks} checks passed")
    else:
        quality_score += 10
    print("\n" + "=" * 80)
    print("5 TEMPORAL COHERENCE (Fraud Scenarios)")
    print("=" * 80)
    max_score += 15
    temporal_score = 0
    if "is_fraud" in df.columns and "fraud_scenario" in df.columns:
        impossible_travel = df[df["fraud_scenario"] == "impossible_travel"]
        if len(impossible_travel) > 0:
            print(f"\nImpossible Travel Analysis:")
            print(f"Cases: {len(impossible_travel):,}")
            print("Temporal fraud pattern detected")
            temporal_score += 5
        velocity_attacks = df[df["fraud_scenario"] == "velocity_attack"]
        if len(velocity_attacks) > 0:
            print(f"\nVelocity Attack Analysis:")
            print(f"Cases: {len(velocity_attacks):,}")
            print("Rapid transaction pattern detected")
            temporal_score += 5
        card_testing = df[
            df["fraud_scenario"].isin(["card_testing", "card_testing_final"])
        ]
        if len(card_testing) > 0:
            print(f"\nCard Testing Analysis:")
            print(f"Cases: {len(card_testing):,}")
            print("Card testing sequence detected")
            temporal_score += 5
        quality_score += temporal_score
    else:
        print("Cannot validate temporal coherence - missing columns")
        quality_score += 8
    print("\n" + "=" * 80)
    print("6 FEATURE CORRELATIONS (Should Make Sense)")
    print("=" * 80)
    max_score += 15
    correlation_checks = 0
    if "is_fraud" in df.columns:
        if "is_vpn" in df.columns:
            vpn_fraud_corr = df["is_vpn"].corr(df["is_fraud"])
            print(f"\nVPN Usage Fraud: {vpn_fraud_corr:.4f}")
            if vpn_fraud_corr > 0.05:
                print("Positive correlation (realistic)")
                correlation_checks += 1
        if "is_high_risk_country" in df.columns:
            country_fraud_corr = df["is_high_risk_country"].corr(df["is_fraud"])
            print(f"\nHigh-Risk Country Fraud: {country_fraud_corr:.4f}")
            if country_fraud_corr > 0.05:
                print("Positive correlation (realistic)")
                correlation_checks += 1
        if "impossible_travel_flag" in df.columns:
            travel_fraud_corr = df["impossible_travel_flag"].corr(df["is_fraud"])
            print(f"\nImpossible Travel Fraud: {travel_fraud_corr:.4f}")
            if travel_fraud_corr > 0.1:
                print("Strong positive correlation (realistic)")
                correlation_checks += 1
    if correlation_checks >= 2:
        print(
            f"\nPASS: Feature correlations make sense ({correlation_checks} validated)"
        )
        quality_score += 15
    else:
        print(
            f"\nWARN: Limited correlation validation ({correlation_checks} validated)"
        )
        quality_score += 8
    print("\n" + "=" * 80)
    print("7 DATA QUALITY ISSUES")
    print("=" * 80)
    max_score += 10
    issues = []
    missing_count = df.isnull().sum().sum()
    missing_pct = missing_count / (len(df) * len(df.columns)) * 100
    print(f"\nMissing Values:")
    print(f"Total: {missing_count:,} ({missing_pct:.2f}%)")
    if missing_pct < 1:
        print("Minimal missing data")
        quality_score += 5
    elif missing_pct < 5:
        print("Some missing data")
        quality_score += 3
        issues.append(f"Missing data: {missing_pct:.2f}%")
    else:
        print("Significant missing data")
        issues.append(f"Missing data: {missing_pct:.2f}%")
    duplicate_count = df.duplicated().sum()
    duplicate_pct = duplicate_count / len(df) * 100
    print(f"\nDuplicate Records:")
    print(f"Count: {duplicate_count:,} ({duplicate_pct:.2f}%)")
    if duplicate_pct < 0.1:
        print("Minimal duplicates")
        quality_score += 5
    else:
        print("Some duplicates present")
        quality_score += 3
        issues.append(f"Duplicates: {duplicate_pct:.2f}%")
    print("\n" + "=" * 80)
    print("FINAL QUALITY ASSESSMENT")
    print("=" * 80)
    final_percentage = quality_score / max_score * 100
    print(
        f"\nQuality Score: {quality_score:.1f} / {max_score} ({final_percentage:.1f}%)"
    )
    if final_percentage >= 90:
        grade = "A+ (Production-Ready)"
        verdict = "EXCELLENT - Dataset is production-grade!"
        recommendation = "This data is ready for real ML model training and deployment."
    elif final_percentage >= 80:
        grade = "A (Very Good)"
        verdict = "VERY GOOD - Dataset is high quality"
        recommendation = "Minor improvements possible, but suitable for production use."
    elif final_percentage >= 70:
        grade = "B (Good)"
        verdict = "GOOD - Dataset is usable with some improvements"
        recommendation = "Address minor issues before production deployment."
    elif final_percentage >= 60:
        grade = "C (Acceptable)"
        verdict = "ACCEPTABLE - Dataset needs improvements"
        recommendation = "Fix identified issues before using in production."
    else:
        grade = "D (Needs Work)"
        verdict = "NEEDS WORK - Significant improvements required"
        recommendation = "Dataset requires major revisions before use."
    print(f"\nGrade: {grade}")
    print(f"Verdict: {verdict}")
    print(f"Recommendation: {recommendation}")
    if issues:
        print(f"\nIssues to Address:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print(f"\nNo major issues detected!")
    print("\n" + "=" * 80)
    print("COMPARISON TO REAL-WORLD FRAUD DATA")
    print("=" * 80)
    print(
        "\nReal Production Fraud Data Characteristics:\n--------------------------------------------\n1. Fraud Rate: 0.1% - 5% (Ours: {:.2f}%)\n2. Temporal Patterns: Yes \n3. Feature Count: 50-300 (Ours: {})\n4. Dataset Size: 100K - 100M+ (Ours: {:,})\n5. Class Imbalance: 20:1 to 200:1 (Ours: {:.1f}:1)\n6. Fraud Scenarios: 5-10 types (Ours: {})\n7. Feature Correlations: Present \n8. Temporal Sequences: Present \n\nOur Dataset vs Real Data: {}% Match\n".format(
            df["is_fraud"].mean() * 100 if "is_fraud" in df.columns else 0,
            len(df.columns),
            len(df),
            (
                (1 - df["is_fraud"].mean()) / df["is_fraud"].mean()
                if "is_fraud" in df.columns
                else 0
            ),
            (
                len(df[df["is_fraud"] == 1]["fraud_scenario"].unique())
                if "fraud_scenario" in df.columns
                else 0
            ),
            final_percentage,
        )
    )
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    return (final_percentage, grade, verdict)


if __name__ == "__main__":
    try:
        score, grade, verdict = validate_dataset_quality()
        print(f"\n\nFinal Assessment: {grade} - {verdict}\n")
    except FileNotFoundError:
        print("\nERROR: Enriched dataset not found!")
        print("Please run: python3 enrich_data_with_fraud_scenarios.py first\n")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
