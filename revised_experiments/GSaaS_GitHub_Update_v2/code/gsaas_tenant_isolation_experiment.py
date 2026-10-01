import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD FROZEN TEST RISK RESULTS
# ============================================================

print("\n" + "=" * 82)
print("GSaaS TENANT-ISOLATION EXPERIMENT")
print("=" * 82)

df = pd.read_csv(
    "gsaas_test_risk_scores.csv"
)
print("\nColumns available in risk-score CSV:")
for col in df.columns:
    print(" -", col)

print(f"\nEvents loaded: {len(df):,}")

print("\nTenants:")
print(
    df["tenant_id"]
    .value_counts()
    .to_string()
)
# ============================================================
# 2. ASSIGN NOMINAL RESPONSE POLICY
# ============================================================

LOW_MEDIUM_THRESHOLD = 0.40
MEDIUM_HIGH_THRESHOLD = 0.60


def assign_action(r):

    if r >= MEDIUM_HIGH_THRESHOLD:
        return "EDGE_CONTAIN_AND_ESCALATE"

    elif r >= LOW_MEDIUM_THRESHOLD:
        return "EDGE_PLUS_CLOUD"

    else:
        return "EDGE_LOCAL"


df["response_action"] = (
    df["R"].apply(assign_action)
)


# ============================================================
# 3. BUILD TENANT-SCOPED RESPONSE TARGET
# ============================================================

def build_response_target(row):

    return {
        "tenant_id": row["tenant_id"],
        "ground_station_id": row["ground_station_id"],
        "satellite_id": row["satellite_id"],
        "session_id": row["session_id"],
        "service": row["service"],
    }


df["response_target"] = df.apply(
    build_response_target,
    axis=1,
)
# ============================================================
# 4. TENANT-ISOLATION INVARIANT TEST
# ============================================================

df["target_tenant"] = df[
    "response_target"
].apply(
    lambda x: x["tenant_id"]
)

df["tenant_scope_preserved"] = (
    df["tenant_id"]
    ==
    df["target_tenant"]
)

scope_violations = df[
    ~df["tenant_scope_preserved"]
]

print("\n" + "=" * 82)
print("TENANT-SCOPE INVARIANT TEST")
print("=" * 82)

print(
    f"Responses evaluated:       "
    f"{len(df):,}"
)

print(
    f"Correctly tenant-scoped:   "
    f"{df['tenant_scope_preserved'].sum():,}"
)

print(
    f"Cross-tenant violations:   "
    f"{len(scope_violations):,}"
)

print(
    f"Isolation success rate:    "
    f"{df['tenant_scope_preserved'].mean():.2%}"
)
# ============================================================
# 5. HIGH-RISK CONTAINMENT DISTRIBUTION BY TENANT
# ============================================================

high_risk = df[
    df["response_action"]
    ==
    "EDGE_CONTAIN_AND_ESCALATE"
].copy()

print("\n" + "=" * 82)
print("HIGH-RISK TENANT-SCOPED CONTAINMENT")
print("=" * 82)

containment_summary = (
    high_risk
    .groupby("tenant_id")
    .agg(
        containment_events=("event_id", "size"),
        unique_sessions=("session_id", "nunique"),
        unique_satellites=("satellite_id", "nunique"),
        mean_R=("R", "mean"),
    )
    .reset_index()
)

print(
    containment_summary.to_string(
        index=False,
        formatters={
            "mean_R": lambda x: f"{x:.4f}"
        }
    )
)