import numpy as np
import pandas as pd


# ============================================================
# 1. LOAD FROZEN EVENT-LEVEL RISK RESULTS
# ============================================================

print("\n" + "=" * 78)
print("GSaaS EDGE/CLOUD DECISION-LOGIC EXPERIMENT")
print("=" * 78)

df = pd.read_csv(
    "gsaas_test_risk_scores.csv"
)

print(f"\nEvents loaded: {len(df):,}")

print(
    f"Risk-score range: "
    f"{df['R'].min():.4f} "
    f"to {df['R'].max():.4f}"
)
# ============================================================
# 2. NOMINAL EDGE/CLOUD POLICY
# ============================================================

LOW_MEDIUM_THRESHOLD = 0.40
MEDIUM_HIGH_THRESHOLD = 0.60


def assign_risk_tier(r):

    if r < LOW_MEDIUM_THRESHOLD:
        return "LOW"

    elif r < MEDIUM_HIGH_THRESHOLD:
        return "MEDIUM"

    else:
        return "HIGH"


df["risk_tier"] = (
    df["R"]
    .apply(assign_risk_tier)
)


def assign_processing_action(tier):

    if tier == "LOW":
        return "EDGE_LOCAL"

    elif tier == "MEDIUM":
        return "EDGE_PLUS_CLOUD"

    else:
        return "EDGE_CONTAIN_AND_ESCALATE"


df["processing_action"] = (
    df["risk_tier"]
    .apply(assign_processing_action)
)
# ============================================================
# 3. EDGE/CLOUD ROUTING SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("EDGE/CLOUD ROUTING SUMMARY")
print("=" * 78)

routing_summary = (
    df
    .groupby(
        [
            "risk_tier",
            "processing_action",
        ]
    )
    .agg(
        events=("R", "size"),
        mean_R=("R", "mean"),
    )
    .reset_index()
)

routing_summary["percentage"] = (
    100.0
    * routing_summary["events"]
    / len(df)
)

print(
    routing_summary.to_string(
        index=False,
        formatters={
            "mean_R": lambda x: f"{x:.4f}",
            "percentage": lambda x: f"{x:.2f}%",
        }
    )
)
# ============================================================
# 4. ROUTING BY EVENT CLASS
# ============================================================

print("\n" + "=" * 78)
print("EDGE/CLOUD ROUTING — BENIGN VS ATTACK")
print("=" * 78)

class_routing = pd.crosstab(
    df["is_attack"],
    df["risk_tier"],
)

class_routing.index = [
    "BENIGN" if x == 0 else "ATTACK"
    for x in class_routing.index
]

# Ensure consistent column order.
for tier in ["LOW", "MEDIUM", "HIGH"]:

    if tier not in class_routing.columns:
        class_routing[tier] = 0

class_routing = class_routing[
    [
        "LOW",
        "MEDIUM",
        "HIGH",
    ]
]

class_routing["TOTAL"] = (
    class_routing.sum(axis=1)
)

print(
    class_routing.to_string()
)
# ============================================================
# 5. CLOUD-DEPENDENCE / LOCAL-HANDLING SUMMARY
# ============================================================

local_only = (
    df["processing_action"]
    == "EDGE_LOCAL"
)

cloud_involved = ~local_only

print("\n" + "=" * 78)
print("LOCAL PROCESSING VS CLOUD INVOLVEMENT")
print("=" * 78)

print(
    f"Edge-local only: "
    f"{local_only.sum():,} "
    f"({local_only.mean():.2%})"
)

print(
    f"Cloud involved:  "
    f"{cloud_involved.sum():,} "
    f"({cloud_involved.mean():.2%})"
)
# ============================================================
# 6. AE-AWARE EDGE/CLOUD DECISION POLICY
# ============================================================

# A >= 1.0 means reconstruction error has reached or exceeded
# the benign-validation P99 reference used to normalize A.
#
# This is therefore an anomaly-evidence escalation trigger,
# not a newly fitted attack threshold.

AE_ESCALATION_THRESHOLD = 1.0


def assign_ae_aware_action(row):

    R = row["R"]
    A = row["A"]

    # --------------------------------------------------------
    # HIGH composite operational risk
    # Immediate bounded local containment + cloud escalation.
    # --------------------------------------------------------
    if R >= MEDIUM_HIGH_THRESHOLD:
        return "EDGE_CONTAIN_AND_ESCALATE"

    # --------------------------------------------------------
    # Strong unsupervised anomaly evidence.
    #
    # Even if composite R is below HIGH, an event reaching
    # the benign-validation P99 anomaly reference should not
    # remain edge-local only.
    # --------------------------------------------------------
    if A >= AE_ESCALATION_THRESHOLD:
        return "EDGE_PLUS_CLOUD"

    # --------------------------------------------------------
    # MEDIUM composite risk
    # Local handling with cloud involvement.
    # --------------------------------------------------------
    if R >= LOW_MEDIUM_THRESHOLD:
        return "EDGE_PLUS_CLOUD"

    # --------------------------------------------------------
    # LOW risk and no strong AE escalation signal
    # --------------------------------------------------------
    return "EDGE_LOCAL"


df["ae_aware_action"] = df.apply(
    assign_ae_aware_action,
    axis=1,
)
# ============================================================
# 7. EFFECT OF UNSUPERVISED AE ESCALATION
# ============================================================

risk_only_local = (
    df["processing_action"] == "EDGE_LOCAL"
)

ae_aware_cloud = (
    df["ae_aware_action"] != "EDGE_LOCAL"
)

ae_rescued = (
    risk_only_local
    & ae_aware_cloud
)

print("\n" + "=" * 78)
print("EFFECT OF UNSUPERVISED AE ESCALATION")
print("=" * 78)

print(
    f"Risk-only EDGE_LOCAL events: "
    f"{risk_only_local.sum():,}"
)

print(
    f"Additional events escalated by AE: "
    f"{ae_rescued.sum():,}"
)

print(
    f"Of these, attacks: "
    f"{((df['is_attack'] == 1) & ae_rescued).sum():,}"
)

print(
    f"Of these, benign:  "
    f"{((df['is_attack'] == 0) & ae_rescued).sum():,}"
)
# ============================================================
# 8. FINAL AE-AWARE ROUTING SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("AE-AWARE EDGE/CLOUD ROUTING SUMMARY")
print("=" * 78)

ae_routing = (
    df["ae_aware_action"]
    .value_counts()
    .rename_axis("action")
    .reset_index(name="events")
)

ae_routing["percentage"] = (
    100.0
    * ae_routing["events"]
    / len(df)
)

print(
    ae_routing.to_string(
        index=False,
        formatters={
            "percentage": lambda x: f"{x:.2f}%"
        }
    )
)
# ============================================================
# 9. ANALYZE ATTACKS ROUTED AS LOW RISK
# ============================================================

low_attacks = df[
    (df["is_attack"] == 1)
    &
    (df["risk_tier"] == "LOW")
].copy()

print("\n" + "=" * 78)
print("ANALYSIS OF ATTACKS ROUTED AS LOW RISK")
print("=" * 78)

print(
    f"Low-risk attacks: "
    f"{len(low_attacks):,}"
)

print(
    f"Mean A:   "
    f"{low_attacks['A'].mean():.4f}"
)

print(
    f"Median A: "
    f"{low_attacks['A'].median():.4f}"
)

print(
    f"P75 A:    "
    f"{np.percentile(low_attacks['A'], 75):.4f}"
)

print(
    f"P90 A:    "
    f"{np.percentile(low_attacks['A'], 90):.4f}"
)

print(
    f"Maximum A:"
    f" {low_attacks['A'].max():.4f}"
)
print("\nLOW-RISK ATTACKS BY ATTACK TYPE")
print("-" * 78)

low_attack_types = (
    low_attacks["attack_type"]
    .value_counts()
)

print(
    low_attack_types.to_string()
)
# ============================================================
# 10. SUPERVISED RECOVERY OF LOW-RISK ATTACKS
# ============================================================

print("\n" + "=" * 82)
print("SUPERVISED RECOVERY OF LOW-RISK ATTACKS")
print("=" * 82)

# Load frozen predictions from the already evaluated MLP_27+AE.
mlp_predictions = pd.read_csv(
    "gsaas_mlp27ae_test_predictions.csv"
)

# Merge by event ID rather than assuming row ordering.
analysis_df = df.merge(
    mlp_predictions[
        [
            "event_id",
            "mlp_27_ae_pred",
        ]
    ],
    on="event_id",
    how="left",
    validate="one_to_one",
)

# Safety check.
if analysis_df["mlp_27_ae_pred"].isna().any():
    raise ValueError(
        "Some test events do not have MLP predictions."
    )

analysis_df["mlp_27_ae_pred"] = (
    analysis_df["mlp_27_ae_pred"]
    .astype(int)
)

# Class 0 = BENIGN.
# Any non-zero class means the supervised MLP predicts
# one of the six attack classes.
analysis_df["mlp_detects_threat"] = (
    analysis_df["mlp_27_ae_pred"] != 0
)

# Ground-truth attacks that the composite risk-only path
# assigned to LOW.
low_risk_attacks = analysis_df[
    (analysis_df["is_attack"] == 1)
    &
    (analysis_df["risk_tier"] == "LOW")
].copy()

# How many of these were recovered by the supervised detector?
recovered = low_risk_attacks[
    low_risk_attacks["mlp_detects_threat"]
].copy()

missed_by_both = low_risk_attacks[
    ~low_risk_attacks["mlp_detects_threat"]
].copy()

n_low = len(low_risk_attacks)
n_recovered = len(recovered)
n_missed_both = len(missed_by_both)

recovery_rate = (
    n_recovered / n_low
    if n_low > 0
    else np.nan
)

print(
    f"\nGround-truth attacks routed LOW by risk-only path: "
    f"{n_low:,}"
)

print(
    f"Recovered by MLP_27+AE: "
    f"{n_recovered:,} "
    f"({recovery_rate:.2%})"
)

print(
    f"Missed by BOTH paths:   "
    f"{n_missed_both:,} "
    f"({n_missed_both / n_low:.2%})"
)


# ============================================================
# 11. RECOVERY BY ATTACK TYPE
# ============================================================

print("\n" + "=" * 82)
print("SUPERVISED RECOVERY BY ATTACK TYPE")
print("=" * 82)

recovery_by_type = (
    low_risk_attacks
    .groupby("attack_type")
    .agg(
        low_risk_attacks=(
            "event_id",
            "size",
        ),
        recovered_by_mlp=(
            "mlp_detects_threat",
            "sum",
        ),
    )
    .reset_index()
)

recovery_by_type["still_missed"] = (
    recovery_by_type["low_risk_attacks"]
    -
    recovery_by_type["recovered_by_mlp"]
)

recovery_by_type["recovery_rate"] = (
    recovery_by_type["recovered_by_mlp"]
    /
    recovery_by_type["low_risk_attacks"]
)

print(
    recovery_by_type.to_string(
        index=False,
        formatters={
            "recovery_rate":
                lambda x: f"{x:.2%}"
        }
    )
)