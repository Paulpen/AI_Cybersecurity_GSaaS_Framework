import pandas as pd
import numpy as np

from gsaas_preprocessing import (
    load_splits,
    validate_feature_schema,
    preprocess_all_splits,
    extract_labels,
)



# ============================================================
# CONFIGURATION
# ============================================================

ANOMALY_THRESHOLD = 1.0

# Temporal correlation window.
# This is a policy parameter, not a learned threshold.
CORRELATION_WINDOW_SECONDS = 60


# ============================================================
# 1. LOAD FROZEN TEST DATA
# ============================================================

print("\n" + "=" * 88)
print("GSaaS CROSS-TENANT ANOMALY-CORRELATION EXPERIMENT")
print("=" * 88)

train_df, validation_df, test_df = load_splits()

risk_df = pd.read_csv(
    "gsaas_test_risk_scores.csv"
)

print(f"\nFrozen test events: {len(test_df):,}")
print(f"Risk-score events:  {len(risk_df):,}")


# ============================================================
# 2. MERGE FROZEN METADATA WITH EXISTING A/R RESULTS
# ============================================================

risk_fields = risk_df[
    [
        "event_id",
        "A",
        "R",
    ]
].copy()

df = test_df.merge(
    risk_fields,
    on="event_id",
    how="left",
    validate="one_to_one",
)

if df["A"].isna().any():
    raise ValueError(
        "Some frozen test events could not be matched "
        "to anomaly/risk scores."
    )

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

print(
    "\nMerged events:",
    f"{len(df):,}"
)

print(
    "Tenants:",
    df["tenant_id"].nunique()
)

print(
    "Ground stations:",
    df["ground_station_id"].nunique()
)
# ============================================================
# 3. IDENTIFY STRONG UNSUPERVISED ANOMALIES
# ============================================================

df["strong_anomaly"] = (
    df["A"] >= ANOMALY_THRESHOLD
)

strong = df[
    df["strong_anomaly"]
].copy()

print("\n" + "=" * 88)
print("STRONG UNSUPERVISED ANOMALIES")
print("=" * 88)

print(
    f"Strong anomaly events: "
    f"{len(strong):,}"
)

print(
    f"Percentage of test set: "
    f"{len(strong) / len(df):.2%}"
)

print("\nBy tenant:")

print(
    strong["tenant_id"]
    .value_counts()
    .to_string()
)
# ============================================================
# 4. CROSS-TENANT TEMPORAL CORRELATION
# ============================================================

correlations = []

strong = strong.sort_values(
    "timestamp"
).reset_index(drop=True)

for i in range(len(strong)):

    event_a = strong.iloc[i]

    for j in range(i + 1, len(strong)):

        event_b = strong.iloc[j]

        delta_seconds = (
            event_b["timestamp"]
            -
            event_a["timestamp"]
        ).total_seconds()

        # Because events are time-sorted, once we leave the
        # temporal window there is no need to search further.
        if delta_seconds > CORRELATION_WINDOW_SECONDS:
            break

        # Must involve different tenants.
        if (
            event_a["tenant_id"]
            ==
            event_b["tenant_id"]
        ):
            continue

        # Must share the same ground-station resource.
        if (
            event_a["ground_station_id"]
            !=
            event_b["ground_station_id"]
        ):
            continue

        # Require service-level similarity.
        if (
            event_a["service"]
            !=
            event_b["service"]
        ):
            continue

        correlations.append(
            {
                "event_a": event_a["event_id"],
                "event_b": event_b["event_id"],

                "tenant_a": event_a["tenant_id"],
                "tenant_b": event_b["tenant_id"],

                "ground_station":
                    event_a["ground_station_id"],

                "service":
                    event_a["service"],

                "delta_seconds":
                    delta_seconds,

                "A_a": event_a["A"],
                "A_b": event_b["A"],

                "R_a": event_a["R"],
                "R_b": event_b["R"],

                # Ground truth retained ONLY for
                # retrospective evaluation.
                "episode_a":
                    event_a["episode_id"],

                "episode_b":
                    event_b["episode_id"],

                "episode_type_a":
                    event_a["episode_type"],

                "episode_type_b":
                    event_b["episode_type"],
            }
        )


correlation_df = pd.DataFrame(
    correlations
)
# ============================================================
# 5. CORRELATION SUMMARY
# ============================================================

print("\n" + "=" * 88)
print("CROSS-TENANT CORRELATION SUMMARY")
print("=" * 88)

print(
    f"Cross-tenant correlated anomaly pairs: "
    f"{len(correlation_df):,}"
)

if len(correlation_df) > 0:

    print(
        "\nMean temporal separation: "
        f"{correlation_df['delta_seconds'].mean():.2f} seconds"
    )

    print(
        "Median temporal separation: "
        f"{correlation_df['delta_seconds'].median():.2f} seconds"
    )

    print("\nPairs by service:")

    print(
        correlation_df["service"]
        .value_counts()
        .to_string()
    )
# ============================================================
# 6. RETROSPECTIVE GROUND-TRUTH ANALYSIS
# ============================================================

if len(correlation_df) > 0:

    correlation_df["both_threat"] = (
        (
            correlation_df["episode_type_a"]
            ==
            "THREAT"
        )
        &
        (
            correlation_df["episode_type_b"]
            ==
            "THREAT"
        )
    )

    correlation_df["same_episode"] = (
        correlation_df["episode_a"]
        ==
        correlation_df["episode_b"]
    )

    correlation_df[
        "same_threat_episode"
    ] = (
        correlation_df["both_threat"]
        &
        correlation_df["same_episode"]
    )

    total_pairs = len(
        correlation_df
    )

    threat_pairs = int(
        correlation_df[
            "both_threat"
        ].sum()
    )

    same_threat_episode = int(
        correlation_df[
            "same_threat_episode"
        ].sum()
    )

    print("\n" + "=" * 88)
    print("RETROSPECTIVE GROUND-TRUTH ANALYSIS")
    print("=" * 88)

    print(
        f"Total correlation flags:          "
        f"{total_pairs:,}"
    )

    print(
        f"Both events ground-truth THREAT:  "
        f"{threat_pairs:,} "
        f"({threat_pairs / total_pairs:.2%})"
    )

    print(
        f"Same ground-truth threat episode: "
        f"{same_threat_episode:,} "
        f"({same_threat_episode / total_pairs:.2%})"
    )

    benign_or_mixed = (
        total_pairs
        -
        threat_pairs
    )

    print(
        f"Benign/mixed correlation flags:   "
        f"{benign_or_mixed:,} "
        f"({benign_or_mixed / total_pairs:.2%})"
    )
