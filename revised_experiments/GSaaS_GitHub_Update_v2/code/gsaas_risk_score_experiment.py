import numpy as np
import pandas as pd
import tensorflow as tf

from gsaas_preprocessing import (
    load_splits,
    validate_feature_schema,
    preprocess_all_splits,
    extract_labels,
)

# ============================================================
# 1. LOAD FROZEN DATASET
# ============================================================

print("\n" + "=" * 70)
print("GSaaS COMPOSITE RISK-SCORE EXPERIMENT")
print("=" * 70)

print("\nLoading frozen dataset...")

train_df, validation_df, test_df = load_splits()

labels_test = extract_labels(test_df)

y_test = labels_test["is_anomaly"]

print(f"Test events: {len(test_df):,}")
print(f"Benign events: {np.sum(y_test == 0):,}")
print(f"Attack events: {np.sum(y_test == 1):,}")
# ============================================================
# 2. SYSTEM CRITICALITY — S
# ============================================================

SYSTEM_CRITICALITY = {
    "MONITORING": 0.20,
    "API": 0.40,
    "FILE_TRANSFER": 0.50,
    "TELEMETRY": 0.60,
    "AUTH": 0.80,
    "TELECOMMAND": 1.00,
}

S = (
    test_df["service"]
    .map(SYSTEM_CRITICALITY)
    .to_numpy(dtype=float)
)

if np.isnan(S).any():
    raise ValueError(
        "Unknown service encountered while constructing S."
    )

print("\n" + "=" * 70)
print("SYSTEM CRITICALITY — S")
print("=" * 70)

for service, value in SYSTEM_CRITICALITY.items():
    count = int(
        (test_df["service"] == service).sum()
    )

    print(
        f"{service:<20}"
        f"S = {value:.2f}"
        f"    events = {count:,}"
    )

print(
    f"\nS range: {S.min():.2f} to {S.max():.2f}"
)
# ============================================================
# 3. CONTEXTUAL FACTOR — C
# ============================================================

MISSION_CONTEXT = {
    "MAINTENANCE": 0.20,
    "NOMINAL": 0.30,
    "DOWNLINK": 0.40,
    "PASS": 0.60,
    "COMMAND": 1.00,
}

C_m = (
    test_df["mission_phase"]
    .map(MISSION_CONTEXT)
    .to_numpy(dtype=float)
)

if np.isnan(C_m).any():
    raise ValueError(
        "Unknown mission phase encountered while constructing C_m."
    )

# Resource utilization is already bounded from 0 to 100.
C_r = (
    test_df["resource_utilization"]
    .to_numpy(dtype=float)
    / 100.0
)

C_r = np.clip(
    C_r,
    0.0,
    1.0,
)

# Equal contribution of mission context and resource stress.
C = (
    0.50 * C_m
    +
    0.50 * C_r
)

print("\n" + "=" * 70)
print("CONTEXTUAL FACTOR — C")
print("=" * 70)

print(
    f"Mission context C_m range: "
    f"{C_m.min():.2f} to {C_m.max():.2f}"
)

print(
    f"Resource context C_r range: "
    f"{C_r.min():.4f} to {C_r.max():.4f}"
)

print(
    f"Composite context C range: "
    f"{C.min():.4f} to {C.max():.4f}"
)
# ============================================================
# 4. AUTOENCODER ANOMALY SCORE — A_i(t)
# ============================================================

AE_MODEL_PATH = "gsaas_autoencoder_seed42.keras"

print("\n" + "=" * 70)
print("AUTOENCODER ANOMALY SCORE — A_i(t)")
print("=" * 70)

# Preprocess using the established frozen pipeline.
(
    X_train,
    X_validation,
    X_test,
    encoder,
    scaler,
    feature_names,
) = preprocess_all_splits(
    train_df,
    validation_df,
    test_df,
)

# Load the already-trained frozen AE.
autoencoder = tf.keras.models.load_model(
    AE_MODEL_PATH
)

# ------------------------------------------------------------
# Validation-benign observations determine P99.
# ------------------------------------------------------------

labels_validation = extract_labels(
    validation_df
)

validation_benign_mask = (
    labels_validation["is_anomaly"] == 0
)

X_validation_benign = X_validation[
    validation_benign_mask
]

validation_reconstructed = autoencoder.predict(
    X_validation_benign,
    batch_size=256,
    verbose=0,
)

validation_benign_errors = np.mean(
    np.square(
        X_validation_benign
        - validation_reconstructed
    ),
    axis=1,
)

T99 = float(
    np.percentile(
        validation_benign_errors,
        99.0,
    )
)

# ------------------------------------------------------------
# Reconstruction error for untouched test observations.
# ------------------------------------------------------------

test_reconstructed = autoencoder.predict(
    X_test,
    batch_size=256,
    verbose=0,
)

test_errors = np.mean(
    np.square(
        X_test - test_reconstructed
    ),
    axis=1,
)

# Normalize to [0, 1].
A = np.clip(
    test_errors / T99,
    0.0,
    1.0,
)

print(
    f"P99 benign-validation threshold: "
    f"{T99:.8f}"
)

print(
    f"Raw test-error range: "
    f"{test_errors.min():.8f} "
    f"to {test_errors.max():.8f}"
)

print(
    f"Normalized A range: "
    f"{A.min():.4f} to {A.max():.4f}"
)

print(
    f"Events saturated at A = 1: "
    f"{np.sum(A >= 1.0):,} "
    f"({np.mean(A >= 1.0):.2%})"
)
# ============================================================
# 5. NOMINAL COMPOSITE RISK SCORE
# ============================================================

ALPHA = 0.50
BETA = 0.30
GAMMA = 0.20

R = (
    ALPHA * A
    + BETA * S
    + GAMMA * C
)

print("\n" + "=" * 70)
print("NOMINAL COMPOSITE RISK SCORE")
print("=" * 70)

print(
    f"Coefficients: "
    f"alpha={ALPHA:.2f}, "
    f"beta={BETA:.2f}, "
    f"gamma={GAMMA:.2f}"
)

print(
    f"R range: "
    f"{R.min():.4f} to {R.max():.4f}"
)

print(
    f"R mean:   {R.mean():.4f}"
)

print(
    f"R median: {np.median(R):.4f}"
)
# ============================================================
# 6. RISK-SCORE SEPARATION — BENIGN VS ATTACK
# ============================================================

benign_R = R[y_test == 0]
attack_R = R[y_test == 1]

print("\n" + "=" * 70)
print("RISK-SCORE SEPARATION — BENIGN VS ATTACK")
print("=" * 70)

print(
    f"{'Group':<12}"
    f"{'Events':>10}"
    f"{'Mean':>12}"
    f"{'Median':>12}"
    f"{'P95':>12}"
    f"{'P99':>12}"
)

print("-" * 70)

for name, values in [
    ("BENIGN", benign_R),
    ("ATTACK", attack_R),
]:
    print(
        f"{name:<12}"
        f"{len(values):>10,}"
        f"{np.mean(values):>12.4f}"
        f"{np.median(values):>12.4f}"
        f"{np.percentile(values, 95):>12.4f}"
        f"{np.percentile(values, 99):>12.4f}"
    )
# ============================================================
# 7. A-ONLY VS COMPOSITE-R DISCRIMINATION
# ============================================================

from sklearn.metrics import roc_auc_score

auc_A = roc_auc_score(
    y_test,
    A,
)

auc_R = roc_auc_score(
    y_test,
    R,
)

print("\n" + "=" * 70)
print("A-ONLY VS COMPOSITE-R DISCRIMINATION")
print("=" * 70)

print(
    f"AE anomaly score A — ROC-AUC: "
    f"{auc_A:.4f}"
)

print(
    f"Composite score R  — ROC-AUC: "
    f"{auc_R:.4f}"
)

print(
    f"Difference:                 "
    f"{auc_R - auc_A:+.4f}"
)
# ============================================================
# 8. RISK PRIORITIZATION BY SYSTEM CRITICALITY
# ============================================================

risk_df = test_df.copy()

risk_df["A"] = A
risk_df["S"] = S
risk_df["C"] = C
risk_df["R"] = R
risk_df["is_attack"] = y_test

print("\n" + "=" * 86)
print("RISK PRIORITIZATION BY SYSTEM CRITICALITY")
print("=" * 86)

service_summary = (
    risk_df
    .groupby("service")
    .agg(
        events=("R", "size"),
        S=("S", "first"),
        mean_A=("A", "mean"),
        mean_C=("C", "mean"),
        mean_R=("R", "mean"),
        median_R=("R", "median"),
    )
    .sort_values("S")
)

print(
    service_summary.to_string(
        formatters={
            "S": lambda x: f"{x:.2f}",
            "mean_A": lambda x: f"{x:.4f}",
            "mean_C": lambda x: f"{x:.4f}",
            "mean_R": lambda x: f"{x:.4f}",
            "median_R": lambda x: f"{x:.4f}",
        }
    )
)
# ============================================================
# 9. PRIORITIZATION WHILE CONTROLLING FOR ANOMALY EVIDENCE
# ============================================================

A_BINS = [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.000001,
]

A_LABELS = [
    "0.0-0.2",
    "0.2-0.4",
    "0.4-0.6",
    "0.6-0.8",
    "0.8-1.0",
]

risk_df["A_band"] = pd.cut(
    risk_df["A"],
    bins=A_BINS,
    labels=A_LABELS,
    include_lowest=True,
    right=False,
)

print("\n" + "=" * 94)
print("SYSTEM-CRITICALITY PRIORITIZATION WITHIN SIMILAR ANOMALY BANDS")
print("=" * 94)

controlled_summary = (
    risk_df
    .groupby(
        ["A_band", "service"],
        observed=True,
    )
    .agg(
        events=("R", "size"),
        mean_A=("A", "mean"),
        S=("S", "first"),
        mean_C=("C", "mean"),
        mean_R=("R", "mean"),
    )
    .reset_index()
)

controlled_summary = controlled_summary[
    controlled_summary["events"] >= 20
]

print(
    controlled_summary.to_string(
        index=False,
        formatters={
            "mean_A": lambda x: f"{x:.4f}",
            "S": lambda x: f"{x:.2f}",
            "mean_C": lambda x: f"{x:.4f}",
            "mean_R": lambda x: f"{x:.4f}",
        }
    )
)
# ============================================================
# 10. CONTEXTUAL PRIORITIZATION
#     WHILE CONTROLLING FOR A AND S
# ============================================================

C_BINS = [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.000001,
]

C_LABELS = [
    "0.0-0.2",
    "0.2-0.4",
    "0.4-0.6",
    "0.6-0.8",
    "0.8-1.0",
]

risk_df["C_band"] = pd.cut(
    risk_df["C"],
    bins=C_BINS,
    labels=C_LABELS,
    include_lowest=True,
    right=False,
)

print("\n" + "=" * 100)
print(
    "CONTEXTUAL PRIORITIZATION WITHIN "
    "SIMILAR ANOMALY BANDS AND SAME SERVICE"
)
print("=" * 100)

context_summary = (
    risk_df
    .groupby(
        [
            "A_band",
            "service",
            "C_band",
        ],
        observed=True,
    )
    .agg(
        events=("R", "size"),
        mean_A=("A", "mean"),
        S=("S", "first"),
        mean_C=("C", "mean"),
        mean_R=("R", "mean"),
    )
    .reset_index()
)

# Do not interpret groups containing very few events.
context_summary = context_summary[
    context_summary["events"] >= 20
]

print(
    context_summary.to_string(
        index=False,
        formatters={
            "mean_A": lambda x: f"{x:.4f}",
            "S": lambda x: f"{x:.2f}",
            "mean_C": lambda x: f"{x:.4f}",
            "mean_R": lambda x: f"{x:.4f}",
        }
    )
)
# ============================================================
# 11. COMPOSITE RISK-SCORE COEFFICIENT SENSITIVITY ANALYSIS
# ============================================================

from scipy.stats import spearmanr

print("\n" + "=" * 110)
print("COMPOSITE RISK-SCORE COEFFICIENT SENSITIVITY ANALYSIS")
print("=" * 110)

# ------------------------------------------------------------
# Nominal configuration
# ------------------------------------------------------------

NOMINAL_ALPHA = 0.50
NOMINAL_BETA  = 0.30
NOMINAL_GAMMA = 0.20

R_nominal = (
    NOMINAL_ALPHA * A
    + NOMINAL_BETA * S
    + NOMINAL_GAMMA * C
)

# ------------------------------------------------------------
# Generate coefficient configurations.
#
# Constraints:
#   alpha + beta + gamma = 1
#   alpha >= beta >= gamma
#
# Coefficients vary in increments of 0.10.
# ------------------------------------------------------------

weight_configs = []

for alpha_int in range(0, 11):

    for beta_int in range(0, 11):

        gamma_int = 10 - alpha_int - beta_int

        if gamma_int < 0:
            continue

        alpha = alpha_int / 10.0
        beta = beta_int / 10.0
        gamma = gamma_int / 10.0

        if alpha >= beta >= gamma:

            weight_configs.append(
                (
                    alpha,
                    beta,
                    gamma,
                )
            )

# ------------------------------------------------------------
# Evaluate every configuration.
# ------------------------------------------------------------

sensitivity_results = []

for alpha, beta, gamma in weight_configs:

    R_candidate = (
        alpha * A
        + beta * S
        + gamma * C
    )

    # Rank correlation with nominal risk ordering.
    rho, _ = spearmanr(
        R_nominal,
        R_candidate,
    )

    # Mean absolute change in risk score.
    mean_abs_change = np.mean(
        np.abs(
            R_candidate
            - R_nominal
        )
    )

    # Maximum absolute change.
    max_abs_change = np.max(
        np.abs(
            R_candidate
            - R_nominal
        )
    )

    # Attack-vs-benign discrimination.
    candidate_auc = roc_auc_score(
        y_test,
        R_candidate,
    )

    sensitivity_results.append(
        {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "spearman_rho": rho,
            "mean_abs_change": mean_abs_change,
            "max_abs_change": max_abs_change,
            "roc_auc": candidate_auc,
        }
    )

sensitivity_df = pd.DataFrame(
    sensitivity_results
)

# ------------------------------------------------------------
# Distance from nominal coefficient configuration.
# Useful for distinguishing nearby perturbations from
# deliberately extreme configurations.
# ------------------------------------------------------------

sensitivity_df["weight_distance"] = (
    np.abs(
        sensitivity_df["alpha"]
        - NOMINAL_ALPHA
    )
    +
    np.abs(
        sensitivity_df["beta"]
        - NOMINAL_BETA
    )
    +
    np.abs(
        sensitivity_df["gamma"]
        - NOMINAL_GAMMA
    )
)

sensitivity_df = sensitivity_df.sort_values(
    [
        "weight_distance",
        "alpha",
        "beta",
        "gamma",
    ]
)

# ------------------------------------------------------------
# Print all configurations.
# ------------------------------------------------------------

print(
    f"\nNominal configuration: "
    f"alpha={NOMINAL_ALPHA:.1f}, "
    f"beta={NOMINAL_BETA:.1f}, "
    f"gamma={NOMINAL_GAMMA:.1f}"
)

print(
    f"Number of configurations tested: "
    f"{len(sensitivity_df)}"
)

print("\n" + "-" * 110)

print(
    f"{'alpha':>7}"
    f"{'beta':>7}"
    f"{'gamma':>7}"
    f"{'Distance':>12}"
    f"{'Spearman rho':>16}"
    f"{'Mean |ΔR|':>14}"
    f"{'Max |ΔR|':>14}"
    f"{'ROC-AUC':>12}"
)

print("-" * 110)

for _, row in sensitivity_df.iterrows():

    print(
        f"{row['alpha']:>7.2f}"
        f"{row['beta']:>7.2f}"
        f"{row['gamma']:>7.2f}"
        f"{row['weight_distance']:>12.2f}"
        f"{row['spearman_rho']:>16.4f}"
        f"{row['mean_abs_change']:>14.4f}"
        f"{row['max_abs_change']:>14.4f}"
        f"{row['roc_auc']:>12.4f}"
    )


# ============================================================
# 12. LOCAL SENSITIVITY AROUND NOMINAL CONFIGURATION
# ============================================================

# A coefficient-distance <= 0.20 corresponds to a small
# redistribution of weight around the nominal operating point.

local_sensitivity = sensitivity_df[
    sensitivity_df["weight_distance"] <= 0.20 + 1e-9
].copy()

print("\n" + "=" * 110)
print("LOCAL SENSITIVITY AROUND NOMINAL 0.5 / 0.3 / 0.2")
print("=" * 110)

print(
    f"Nearby configurations tested: "
    f"{len(local_sensitivity)}"
)

print(
    f"Minimum Spearman rho: "
    f"{local_sensitivity['spearman_rho'].min():.4f}"
)

print(
    f"Mean Spearman rho:    "
    f"{local_sensitivity['spearman_rho'].mean():.4f}"
)

print(
    f"Largest mean |ΔR|:    "
    f"{local_sensitivity['mean_abs_change'].max():.4f}"
)

print(
    f"Largest max |ΔR|:     "
    f"{local_sensitivity['max_abs_change'].max():.4f}"
)

print(
    f"ROC-AUC range:         "
    f"{local_sensitivity['roc_auc'].min():.4f}"
    f" to "
    f"{local_sensitivity['roc_auc'].max():.4f}"
)

# ============================================================
# EXPORT EVENT-LEVEL RISK SCORES FOR DOWNSTREAM EXPERIMENTS
# ============================================================

risk_export = risk_df[
    [
        "event_id",
        "episode_id",
        "episode_type",
	"tenant_id",
        "session_id",
        "attack_type",
        "service",
        "mission_phase",
        "resource_utilization",
        "A",
        "S",
        "C",
        "R",
        "is_attack",
    ]
].copy()

risk_export.to_csv(
    "gsaas_test_risk_scores.csv",
    index=False,
)

print(
    "\nSaved event-level risk scores to "
    "gsaas_test_risk_scores.csv"
)