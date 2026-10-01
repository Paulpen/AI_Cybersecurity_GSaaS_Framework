import pandas as pd


# ============================================================
# POST-ORCHESTRATION FALSE-ALARM BURDEN (FAB) ANALYSIS
# ============================================================

print("\n" + "=" * 92)
print("POST-ORCHESTRATION FALSE-ALARM BURDEN (FAB) ANALYSIS")
print("=" * 92)


# ============================================================
# 1. LOAD FROZEN OUTPUTS
# ============================================================

pred_df = pd.read_csv(
    "gsaas_mlp27ae_test_predictions.csv"
)

risk_df = pd.read_csv(
    "gsaas_test_risk_scores.csv"
)

print(f"\nMLP prediction events: {len(pred_df):,}")
print(f"Risk-score events:     {len(risk_df):,}")


# ============================================================
# 2. MERGE MLP PREDICTIONS WITH RISK SCORES
# ============================================================

df = pred_df.merge(
    risk_df[
        [
            "event_id",
            "R",
        ]
    ],
    on="event_id",
    how="left",
    validate="one_to_one",
)

if df["R"].isna().any():
    raise ValueError(
        "Some prediction events could not be matched "
        "to their risk scores."
    )

print(f"Merged events:         {len(df):,}")


# ============================================================
# 3. IDENTIFY RAW DETECTOR FALSE POSITIVES
# ============================================================

# BENIGN = class index 0.
#
# A false positive occurs when:
# actual class = BENIGN
# MLP prediction = any attack class.

df["false_positive"] = (
    (df["actual_class"] == 0)
    &
    (df["predicted_class"] != 0)
)

total_benign = int(
    (df["actual_class"] == 0).sum()
)

raw_fp = int(
    df["false_positive"].sum()
)

raw_fpr = (
    raw_fp / total_benign
)

print("\n" + "=" * 92)
print("RAW DETECTOR FALSE-POSITIVE LOAD")
print("=" * 92)

print(
    f"Benign test events:          "
    f"{total_benign:,}"
)

print(
    f"Raw detector false positives:"
    f" {raw_fp:,}"
)

print(
    f"Raw detector FPR:            "
    f"{raw_fpr:.2%}"
)


# ============================================================
# 4. APPLY THE FROZEN RISK-ROUTING POLICY
# ============================================================

LOW_MEDIUM_THRESHOLD = 0.40
MEDIUM_HIGH_THRESHOLD = 0.60


def assign_risk_tier(r):

    if r >= MEDIUM_HIGH_THRESHOLD:
        return "HIGH"

    elif r >= LOW_MEDIUM_THRESHOLD:
        return "MEDIUM"

    return "LOW"


df["risk_tier"] = (
    df["R"].apply(assign_risk_tier)
)


# ============================================================
# 5. DEFINE ANALYST-FACING EVENTS
# ============================================================

# This follows the response-orchestration policy:
#
# LOW:
#     EDGE_LOCAL -> MONITOR
#
# MEDIUM:
#     EDGE_PLUS_CLOUD -> ALERT + RATE_LIMIT + ESCALATE
#
# HIGH:
#     EDGE_CONTAIN_AND_ESCALATE -> ALERT + containment
#
# Therefore MEDIUM and HIGH events generate analyst-facing
# alerts in this FAB model.

df["analyst_facing"] = (
    df["risk_tier"].isin(
        ["MEDIUM", "HIGH"]
    )
)


# ============================================================
# 6. TRACE FALSE POSITIVES THROUGH ORCHESTRATION
# ============================================================

fp_df = df[
    df["false_positive"]
].copy()

post_fp = int(
    fp_df["analyst_facing"].sum()
)

edge_local_fp = (
    raw_fp - post_fp
)

filtering_rate = (
    edge_local_fp / raw_fp
    if raw_fp > 0
    else 0.0
)


print("\n" + "=" * 92)
print("FALSE-POSITIVE ROUTING")
print("=" * 92)

print(
    f"Raw detector false positives:     "
    f"{raw_fp:,}"
)

print(
    f"Retained EDGE_LOCAL:              "
    f"{edge_local_fp:,}"
)

print(
    f"Analyst-facing false positives:   "
    f"{post_fp:,}"
)

print(
    f"False positives filtered locally: "
    f"{filtering_rate:.2%}"
)


print("\nFalse positives by risk tier:")

tier_counts = (
    fp_df["risk_tier"]
    .value_counts()
    .reindex(
        ["LOW", "MEDIUM", "HIGH"],
        fill_value=0,
    )
)

print(
    tier_counts.to_string()
)


# ============================================================
# 7. FAB CAPACITY SCENARIOS
# ============================================================

ANALYST_COUNTS = [
    1,
    2,
    4,
]

TRIAGE_MINUTES = [
    2,
    5,
    10,
]

HOURS_PER_ANALYST = 8

fab_rows = []


for analysts in ANALYST_COUNTS:

    for triage_min in TRIAGE_MINUTES:

        # Number of alerts that the available analysts
        # could process during the assumed 8-hour window.

        capacity = (
            60
            * analysts
            * HOURS_PER_ANALYST
            / triage_min
        )

        # Raw burden assumes every detector false positive
        # becomes an analyst-facing alert.

        fab_raw = (
            raw_fp / capacity
        )

        # Post-orchestration burden includes only false
        # positives actually escalated to analysts.

        fab_post = (
            post_fp / capacity
        )

        fab_reduction = (
            1.0
            -
            (fab_post / fab_raw)
            if fab_raw > 0
            else 0.0
        )

        fab_rows.append(
            {
                "analysts": analysts,
                "triage_min": triage_min,
                "capacity_alerts": capacity,
                "FAB_raw": fab_raw,
                "FAB_post": fab_post,
                "FAB_reduction": fab_reduction,
            }
        )


fab_results = pd.DataFrame(
    fab_rows
)


# ============================================================
# 8. PRINT RAW VS POST-ORCHESTRATION FAB
# ============================================================

print("\n" + "=" * 92)
print("RAW VS POST-ORCHESTRATION FAB")
print("=" * 92)

print(
    fab_results.to_string(
        index=False,
        formatters={
            "capacity_alerts":
                lambda x: f"{x:.0f}",

            "FAB_raw":
                lambda x: f"{x:.2f}",

            "FAB_post":
                lambda x: f"{x:.2f}",

            "FAB_reduction":
                lambda x: f"{x:.2%}",
        }
    )
)


# ============================================================
# 9. FAB MATRICES FOR EASY PAPER REPORTING
# ============================================================

print("\n" + "=" * 92)
print("RAW FAB MATRIX")
print("=" * 92)

raw_matrix = fab_results.pivot(
    index="triage_min",
    columns="analysts",
    values="FAB_raw",
)

raw_matrix.columns = [
    f"{x} analyst"
    if x == 1
    else f"{x} analysts"
    for x in raw_matrix.columns
]

print(
    raw_matrix.to_string(
        formatters={
            col: lambda x: f"{x:.2f}"
            for col in raw_matrix.columns
        }
    )
)


print("\n" + "=" * 92)
print("POST-ORCHESTRATION FAB MATRIX")
print("=" * 92)

post_matrix = fab_results.pivot(
    index="triage_min",
    columns="analysts",
    values="FAB_post",
)

post_matrix.columns = [
    f"{x} analyst"
    if x == 1
    else f"{x} analysts"
    for x in post_matrix.columns
]

print(
    post_matrix.to_string(
        formatters={
            col: lambda x: f"{x:.2f}"
            for col in post_matrix.columns
        }
    )
)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 92)
print("FAB SUMMARY")
print("=" * 92)

print(
    f"Detector false positives:          "
    f"{raw_fp:,}"
)

print(
    f"Analyst-facing false positives:    "
    f"{post_fp:,}"
)

print(
    f"Edge-local false positives:        "
    f"{edge_local_fp:,}"
)

print(
    f"Reduction in analyst-facing "
    f"false-alarm burden: "
    f"{filtering_rate:.2%}"
)