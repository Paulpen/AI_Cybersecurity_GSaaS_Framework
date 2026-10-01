import pandas as pd


# ============================================================
# ANALYST CAPACITY-AWARE FALSE-ALARM BURDEN (FAB) MODEL
# ============================================================

print("\n" + "=" * 96)
print("ANALYST CAPACITY-AWARE FALSE-ALARM BURDEN (FAB) ANALYSIS")
print("=" * 96)


# ============================================================
# 1. VALIDATION-SWEEP RESULTS
# ============================================================

N_BENIGN = 18230

lambda_results = {
    0.00: {"fpr": 0.0075, "attack_recall": 0.6305},
    0.25: {"fpr": 0.0322, "attack_recall": 0.7271},
    0.50: {"fpr": 0.0716, "attack_recall": 0.8102},
    0.75: {"fpr": 0.1317, "attack_recall": 0.8633},
    1.00: {"fpr": 0.2852, "attack_recall": 0.9215},
}


# ============================================================
# 2. ANALYST-CAPACITY SCENARIOS
# ============================================================

# We deliberately examine several operational assumptions
# rather than claiming one universal analyst capacity.

analyst_counts = [1, 2, 4]

triage_minutes = [2, 5, 10]

# One 8-hour operational shift.
HOURS_PER_ANALYST = 8


# ============================================================
# 3. COMPUTE FAB
# ============================================================

rows = []

for lam, metrics in lambda_results.items():

    fpr = metrics["fpr"]
    recall = metrics["attack_recall"]

    false_alarms = N_BENIGN * fpr

    for analysts in analyst_counts:

        for triage_time in triage_minutes:

            capacity = (
                60
                * analysts
                * HOURS_PER_ANALYST
                / triage_time
            )

            fab = (
                false_alarms
                / capacity
            )

            analyst_hours_consumed = (
                false_alarms
                * triage_time
                / 60
            )

            rows.append(
                {
                    "lambda": lam,
                    "FPR": fpr,
                    "attack_recall": recall,
                    "false_alarms": false_alarms,
                    "analysts": analysts,
                    "triage_min": triage_time,
                    "capacity_alerts": capacity,
                    "analyst_hours_consumed":
                        analyst_hours_consumed,
                    "FAB": fab,
                }
            )


fab_df = pd.DataFrame(rows)


# ============================================================
# 4. PRINT FALSE-ALARM LOAD BY LAMBDA
# ============================================================

print("\n" + "=" * 96)
print("FALSE-ALARM LOAD IMPLIED BY CLASS-WEIGHT CALIBRATION")
print("=" * 96)

base = (
    fab_df[
        [
            "lambda",
            "FPR",
            "attack_recall",
            "false_alarms",
        ]
    ]
    .drop_duplicates()
    .sort_values("lambda")
)

print(
    base.to_string(
        index=False,
        formatters={
            "FPR":
                lambda x: f"{x:.2%}",
            "attack_recall":
                lambda x: f"{x:.2%}",
            "false_alarms":
                lambda x: f"{x:.1f}",
        }
    )
)


# ============================================================
# 5. PRINT FAB SCENARIO MATRIX
# ============================================================

print("\n" + "=" * 96)
print("FAB SCENARIO MATRIX")
print("=" * 96)

for triage_time in triage_minutes:

    print(
        f"\nMean triage time = "
        f"{triage_time} minutes per false alarm"
    )

    subset = fab_df[
        fab_df["triage_min"]
        ==
        triage_time
    ]

    matrix = subset.pivot(
        index="lambda",
        columns="analysts",
        values="FAB",
    )

    matrix.columns = [
        f"{c} analyst"
        if c == 1
        else f"{c} analysts"
        for c in matrix.columns
    ]

    print(
        matrix.to_string(
            formatters={
                col: lambda x: f"{x:.2f}"
                for col in matrix.columns
            }
        )
    )


# ============================================================
# 6. SELECTED OPERATING POINT — LAMBDA 0.50
# ============================================================

print("\n" + "=" * 96)
print("SELECTED INTERMEDIATE OPERATING POINT — lambda = 0.50")
print("=" * 96)

selected = fab_df[
    fab_df["lambda"] == 0.50
].copy()

print(
    selected[
        [
            "analysts",
            "triage_min",
            "false_alarms",
            "capacity_alerts",
            "analyst_hours_consumed",
            "FAB",
        ]
    ].to_string(
        index=False,
        formatters={
            "false_alarms":
                lambda x: f"{x:.1f}",
            "capacity_alerts":
                lambda x: f"{x:.0f}",
            "analyst_hours_consumed":
                lambda x: f"{x:.2f}",
            "FAB":
                lambda x: f"{x:.2f}",
        }
    )
)


# ============================================================
# 7. ESCALATION BEYOND LAMBDA 0.50
# ============================================================

print("\n" + "=" * 96)
print("FALSE-ALARM COST ESCALATION BEYOND lambda = 0.50")
print("=" * 96)

fa_050 = (
    N_BENIGN
    *
    lambda_results[0.50]["fpr"]
)

for lam in [0.75, 1.00]:

    false_alarms = (
        N_BENIGN
        *
        lambda_results[lam]["fpr"]
    )

    increase = (
        false_alarms
        -
        fa_050
    )

    multiplier = (
        false_alarms
        /
        fa_050
    )

    recall_gain = (
        lambda_results[lam]["attack_recall"]
        -
        lambda_results[0.50]["attack_recall"]
    )

    print(
        f"\nlambda {lam:.2f} vs 0.50:"
    )

    print(
        f"  Additional false alarms: "
        f"{increase:.1f}"
    )

    print(
        f"  False-alarm multiplier:  "
        f"{multiplier:.2f}x"
    )

    print(
        f"  Attack-recall gain:      "
        f"{recall_gain:.2%}"
    )