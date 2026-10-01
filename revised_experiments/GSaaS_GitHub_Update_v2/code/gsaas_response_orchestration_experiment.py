import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

LOW_MEDIUM_THRESHOLD = 0.40
MEDIUM_HIGH_THRESHOLD = 0.60


# ============================================================
# 1. LOAD FROZEN TEST RESULTS
# ============================================================

print("\n" + "=" * 90)
print("GSaaS RESPONSE-ORCHESTRATION EXPERIMENT")
print("=" * 90)

df = pd.read_csv(
    "gsaas_test_risk_scores.csv"
)

print(f"\nEvents loaded: {len(df):,}")


# ============================================================
# 2. DEFINE BOUNDED RESPONSE ACTIONS
# ============================================================

# These are the ONLY mitigation actions that the simulated
# orchestrator is permitted to issue.

ALLOWED_ACTIONS = {
    "MONITOR",
    "ALERT",
    "RATE_LIMIT",
    "RESTRICT_ACCESS",
    "ISOLATE_WORKLOAD",
    "BLOCK_TRAFFIC",
    "ESCALATE_TO_CLOUD",
    "INITIATE_RECOVERY",
}


# ============================================================
# 3. ASSIGN RISK TIER
# ============================================================

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
# 4. RESPONSE ORCHESTRATOR
# ============================================================

def orchestrate_response(row):

    tier = row["risk_tier"]

    # --------------------------------------------------------
    # LOW
    #
    # Retain locally for monitoring.
    # No disruptive mitigation action.
    # --------------------------------------------------------
    if tier == "LOW":

        actions = [
            "MONITOR",
        ]

        processing = "EDGE_LOCAL"

    # --------------------------------------------------------
    # MEDIUM
    #
    # Alert and apply a bounded local traffic-control action,
    # while forwarding the event for cloud analysis.
    # --------------------------------------------------------
    elif tier == "MEDIUM":

        actions = [
            "ALERT",
            "RATE_LIMIT",
            "ESCALATE_TO_CLOUD",
        ]

        processing = "EDGE_PLUS_CLOUD"

    # --------------------------------------------------------
    # HIGH
    #
    # Immediate tenant-scoped containment, cloud escalation,
    # and recovery preparation.
    # --------------------------------------------------------
    else:

        actions = [
            "ALERT",
            "RESTRICT_ACCESS",
            "ISOLATE_WORKLOAD",
            "BLOCK_TRAFFIC",
            "ESCALATE_TO_CLOUD",
            "INITIATE_RECOVERY",
        ]

        processing = "EDGE_CONTAIN_AND_ESCALATE"

    return {
        "tenant_id": row["tenant_id"],
        "session_id": row["session_id"],
        "service": row["service"],
        "risk_tier": tier,
        "processing": processing,
        "actions": actions,
    }


df["response_plan"] = df.apply(
    orchestrate_response,
    axis=1,
)


# ============================================================
# 5. EXTRACT RESPONSE INFORMATION
# ============================================================

df["response_tenant"] = (
    df["response_plan"]
    .apply(lambda x: x["tenant_id"])
)

df["response_session"] = (
    df["response_plan"]
    .apply(lambda x: x["session_id"])
)

df["response_processing"] = (
    df["response_plan"]
    .apply(lambda x: x["processing"])
)

df["response_actions"] = (
    df["response_plan"]
    .apply(lambda x: x["actions"])
)


# ============================================================
# 6. TENANT/SCOPE PRESERVATION TEST
# ============================================================

df["tenant_preserved"] = (
    df["tenant_id"]
    ==
    df["response_tenant"]
)

df["session_preserved"] = (
    df["session_id"]
    ==
    df["response_session"]
)


# ============================================================
# 7. ALLOW-LIST SAFETY TEST
# ============================================================

def actions_are_allowed(actions):

    return all(
        action in ALLOWED_ACTIONS
        for action in actions
    )


df["actions_allowed"] = (
    df["response_actions"]
    .apply(actions_are_allowed)
)


# ============================================================
# 8. POLICY-CONSISTENCY TEST
# ============================================================

def policy_is_consistent(row):

    actions = set(
        row["response_actions"]
    )

    tier = row["risk_tier"]

    if tier == "LOW":

        expected = {
            "MONITOR",
        }

    elif tier == "MEDIUM":

        expected = {
            "ALERT",
            "RATE_LIMIT",
            "ESCALATE_TO_CLOUD",
        }

    else:

        expected = {
            "ALERT",
            "RESTRICT_ACCESS",
            "ISOLATE_WORKLOAD",
            "BLOCK_TRAFFIC",
            "ESCALATE_TO_CLOUD",
            "INITIATE_RECOVERY",
        }

    return actions == expected


df["policy_consistent"] = df.apply(
    policy_is_consistent,
    axis=1,
)


# ============================================================
# 9. VALIDATION SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("RESPONSE-ORCHESTRATION VALIDATION")
print("=" * 90)

print(
    f"Response plans evaluated:      "
    f"{len(df):,}"
)

print(
    f"Tenant scope preserved:        "
    f"{df['tenant_preserved'].sum():,}"
)

print(
    f"Session scope preserved:       "
    f"{df['session_preserved'].sum():,}"
)

print(
    f"Allow-list compliant:          "
    f"{df['actions_allowed'].sum():,}"
)

print(
    f"Policy-consistent responses:   "
    f"{df['policy_consistent'].sum():,}"
)

print(
    f"Tenant-scope violations:       "
    f"{(~df['tenant_preserved']).sum():,}"
)

print(
    f"Session-scope violations:      "
    f"{(~df['session_preserved']).sum():,}"
)

print(
    f"Disallowed-action violations:  "
    f"{(~df['actions_allowed']).sum():,}"
)

print(
    f"Policy violations:             "
    f"{(~df['policy_consistent']).sum():,}"
)


# ============================================================
# 10. RESPONSE DISTRIBUTION
# ============================================================

print("\n" + "=" * 90)
print("RESPONSE DISTRIBUTION BY RISK TIER")
print("=" * 90)

summary = (
    df.groupby(
        [
            "risk_tier",
            "response_processing",
        ]
    )
    .agg(
        events=("event_id", "size"),
        mean_R=("R", "mean"),
    )
    .reset_index()
)

summary["percentage"] = (
    100.0
    *
    summary["events"]
    /
    len(df)
)

print(
    summary.to_string(
        index=False,
        formatters={
            "mean_R":
                lambda x: f"{x:.4f}",
            "percentage":
                lambda x: f"{x:.2f}%",
        }
    )
)


# ============================================================
# 11. ACTION FREQUENCIES
# ============================================================

print("\n" + "=" * 90)
print("MITIGATION-ACTION FREQUENCIES")
print("=" * 90)

all_actions = (
    df["response_actions"]
    .explode()
)

action_counts = (
    all_actions
    .value_counts()
)

print(
    action_counts.to_string()
)