"""
GSaaS Synthetic Dataset Quality Audit
=====================================

Audits the synthetic GSaaS event dataset before it is supplied
to any machine-learning model.

This script DOES NOT train the AE or MLP and DOES NOT modify
the generated dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "gsaas_generated_data/gsaas_events_seed42.csv"
)

NUMERIC_FEATURES = [
    "duration",
    "packet_count",
    "byte_count",
    "request_rate",
    "connection_rate",
    "failed_auth_rate",
    "unique_destinations",
    "telecommand_rate",
    "telemetry_rate",
    "file_transfer_rate",
    "api_request_rate",
    "resource_utilization",
]

CATEGORICAL_FEATURES = [
    "protocol",
    "service",
    "state",
    "tenant_id",
    "ground_station_id",
    "satellite_id",
    "mission_phase",
]


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find dataset:\n{DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)

    print("=" * 70)
    print("GSaaS SYNTHETIC DATASET QUALITY AUDIT")
    print("=" * 70)

    print(f"\nDataset: {DATA_PATH}")
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


# ============================================================
# 3. BASIC INTEGRITY AUDIT
# ============================================================

def audit_integrity(df):

    print("\n" + "=" * 70)
    print("1. DATA INTEGRITY")
    print("=" * 70)

    missing = df.isna().sum()
    missing = missing[missing > 0]

    print("\nMissing values:")

    if len(missing) == 0:
        print("  None")
    else:
        print(missing)

    numeric_df = df[NUMERIC_FEATURES]

    inf_count = np.isinf(
        numeric_df.to_numpy(dtype=float)
    ).sum()

    print(
        f"\nInfinite numeric values: {inf_count:,}"
    )

    print("\nNegative-value checks:")

    for feature in NUMERIC_FEATURES:

        count = int(
            (df[feature] < 0).sum()
        )

        print(
            f"  {feature:<24} {count:>8,}"
        )

    print("\nResource-utilization bounds:")

    below = int(
        (df["resource_utilization"] < 0).sum()
    )

    above = int(
        (df["resource_utilization"] > 100).sum()
    )

    print(f"  Below 0%:   {below:,}")
    print(f"  Above 100%: {above:,}")

    duplicate_ids = int(
        df["event_id"].duplicated().sum()
    )

    print(
        f"\nDuplicate event IDs: {duplicate_ids:,}"
    )


# ============================================================
# 4. NUMERICAL FEATURE DISTRIBUTIONS
# ============================================================

def audit_numeric_features(df):

    print("\n" + "=" * 70)
    print("2. NUMERICAL FEATURE DISTRIBUTIONS")
    print("=" * 70)

    stats = df[NUMERIC_FEATURES].describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    ).T

    display_cols = [
        "min",
        "1%",
        "5%",
        "50%",
        "95%",
        "99%",
        "max",
        "mean",
        "std",
    ]

    print(
        stats[display_cols].round(3).to_string()
    )


# ============================================================
# 5. BENIGN VS THREAT COMPARISON
# ============================================================

def audit_benign_vs_threat(df):

    print("\n" + "=" * 70)
    print("3. BENIGN VS THREAT FEATURE COMPARISON")
    print("=" * 70)

    benign = df[
        df["is_anomaly"] == 0
    ]

    threat = df[
        df["is_anomaly"] == 1
    ]

    print(
        f"\nBenign events: {len(benign):,}"
    )

    print(
        f"Threat events: {len(threat):,}"
    )

    rows = []

    for feature in NUMERIC_FEATURES:

        benign_median = benign[feature].median()
        threat_median = threat[feature].median()

        benign_p95 = benign[feature].quantile(0.95)
        threat_p95 = threat[feature].quantile(0.95)

        if benign_median > 0:
            median_ratio = (
                threat_median / benign_median
            )
        else:
            median_ratio = np.nan

        rows.append(
            {
                "feature": feature,
                "benign_median": benign_median,
                "threat_median": threat_median,
                "median_ratio": median_ratio,
                "benign_p95": benign_p95,
                "threat_p95": threat_p95,
            }
        )

    result = pd.DataFrame(rows)

    print(
        "\n"
        + result.round(3).to_string(
            index=False
        )
    )


# ============================================================
# 6. NORMAL / BENIGN-UNUSUAL / THREAT COMPARISON
# ============================================================

def audit_operational_overlap(df):

    print("\n" + "=" * 70)
    print("4. NORMAL / BENIGN-UNUSUAL / THREAT OVERLAP")
    print("=" * 70)

    # Threat rows only during active malicious periods.
    threat = df[
        df["is_anomaly"] == 1
    ].copy()

    normal = df[
        df["episode_type"] == "NORMAL"
    ].copy()

    benign_unusual = df[
        (
            df["episode_type"]
            == "BENIGN_UNUSUAL"
        )
        & (
            df["is_anomaly"] == 0
        )
    ].copy()

    groups = {
        "NORMAL": normal,
        "BENIGN_UNUSUAL": benign_unusual,
        "THREAT": threat,
    }

    rows = []

    for name, subset in groups.items():

        for feature in NUMERIC_FEATURES:

            rows.append(
                {
                    "group": name,
                    "feature": feature,
                    "median": subset[
                        feature
                    ].median(),
                    "p95": subset[
                        feature
                    ].quantile(0.95),
                    "p99": subset[
                        feature
                    ].quantile(0.99),
                }
            )

    result = pd.DataFrame(rows)

    pivot_median = result.pivot(
        index="feature",
        columns="group",
        values="median",
    )

    print("\nMedian values:")
    print(
        pivot_median.round(3).to_string()
    )

    pivot_p95 = result.pivot(
        index="feature",
        columns="group",
        values="p95",
    )

    print("\n95th-percentile values:")
    print(
        pivot_p95.round(3).to_string()
    )


# ============================================================
# 7. THREAT-SPECIFIC FEATURE AUDIT
# ============================================================

def audit_threat_profiles(df):

    print("\n" + "=" * 70)
    print("5. THREAT-SPECIFIC FEATURE PROFILES")
    print("=" * 70)

    threat = df[
        df["is_anomaly"] == 1
    ]

    profile = (
        threat
        .groupby("attack_type")[
            NUMERIC_FEATURES
        ]
        .median()
        .round(3)
    )

    print(
        "\nMedian feature values by threat:\n"
    )

    print(profile.to_string())


# ============================================================
# 8. SEVERITY DISTRIBUTION
# ============================================================

def audit_severity(df):

    print("\n" + "=" * 70)
    print("6. THREAT SEVERITY DISTRIBUTION")
    print("=" * 70)

    threat_episodes = (
        df.loc[
            df["is_anomaly"] == 1,
            [
                "episode_id",
                "attack_type",
                "severity",
            ],
        ]
        .drop_duplicates(
            subset=["episode_id"]
        )
    )

    print("\nEpisodes by severity:")

    print(
        threat_episodes[
            "severity"
        ].value_counts()
    )

    print(
        "\nThreat type × severity:"
    )

    table = pd.crosstab(
        threat_episodes["attack_type"],
        threat_episodes["severity"],
    )

    print(table.to_string())


# ============================================================
# 9. FEATURE CORRELATION AUDIT
# ============================================================

def audit_correlations(df):

    print("\n" + "=" * 70)
    print("7. NUMERICAL FEATURE CORRELATIONS")
    print("=" * 70)

    corr = df[
        NUMERIC_FEATURES
    ].corr()

    pairs = []

    for i in range(len(NUMERIC_FEATURES)):

        for j in range(i + 1, len(NUMERIC_FEATURES)):

            f1 = NUMERIC_FEATURES[i]
            f2 = NUMERIC_FEATURES[j]

            value = corr.loc[f1, f2]

            pairs.append(
                {
                    "feature_1": f1,
                    "feature_2": f2,
                    "correlation": value,
                    "abs_correlation": abs(value),
                }
            )

    pairs_df = pd.DataFrame(pairs)

    pairs_df = pairs_df.sort_values(
        "abs_correlation",
        ascending=False,
    )

    print(
        "\nTop 15 absolute correlations:\n"
    )

    print(
        pairs_df[
            [
                "feature_1",
                "feature_2",
                "correlation",
            ]
        ]
        .head(15)
        .round(3)
        .to_string(index=False)
    )


# ============================================================
# 10. SIMPLE UNIVARIATE SEPARABILITY AUDIT
# ============================================================

def audit_separability(df):

    print("\n" + "=" * 70)
    print("8. SIMPLE FEATURE SEPARABILITY CHECK")
    print("=" * 70)

    benign = df[
        df["is_anomaly"] == 0
    ]

    threat = df[
        df["is_anomaly"] == 1
    ]

    rows = []

    for feature in NUMERIC_FEATURES:

        benign_p99 = benign[
            feature
        ].quantile(0.99)

        proportion_above = (
            threat[feature] > benign_p99
        ).mean()

        rows.append(
            {
                "feature": feature,
                "benign_p99": benign_p99,
                "threat_above_benign_p99":
                    proportion_above,
            }
        )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "threat_above_benign_p99",
        ascending=False,
    )

    result[
        "threat_above_benign_p99"
    ] *= 100

    print(
        "\nPercentage of threat events above "
        "the benign 99th percentile:\n"
    )

    print(
        result.round(3).to_string(
            index=False
        )
    )


# ============================================================
# 11. CATEGORICAL DISTRIBUTIONS
# ============================================================

def audit_categorical_features(df):

    print("\n" + "=" * 70)
    print("9. CATEGORICAL FEATURE DISTRIBUTIONS")
    print("=" * 70)

    for feature in CATEGORICAL_FEATURES:

        print(f"\n{feature}:")

        proportions = (
            df[feature]
            .value_counts(
                normalize=True
            )
            .mul(100)
            .round(2)
        )

        print(proportions.to_string())


# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":

    df = load_data()

    audit_integrity(df)
    audit_numeric_features(df)
    audit_benign_vs_threat(df)
    audit_operational_overlap(df)
    audit_threat_profiles(df)
    audit_severity(df)
    audit_correlations(df)
    audit_separability(df)
    audit_categorical_features(df)

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)