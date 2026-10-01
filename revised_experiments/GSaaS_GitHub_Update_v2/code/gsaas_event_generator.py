"""
GSaaS Event Generator
=====================

Generates a controlled, domain-grounded synthetic event stream for evaluating
the AI-driven GSaaS cybersecurity framework.

The generated data are synthetic and must not be interpreted as operational
traffic collected from a real GSaaS provider.

Design:
    - 3 tenants
    - 3 ground stations
    - 3 satellites
    - 1,000 episodes
    - 100 events per episode
    - 100,000 events total
    - 70% normal benign episodes
    - 15% benign-unusual episodes
    - 15% threat episodes
    - 6 balanced threat classes
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 1. GLOBAL CONFIGURATION
# ============================================================

N_EPISODES = 1000
EVENTS_PER_EPISODE = 100

TENANTS = {
    "TENANT_EO": 1.00,
    "TENANT_COMMS": 1.35,
    "TENANT_SCI": 0.80,
}

GROUND_STATIONS = {
    "GS01": 1.00,
    "GS02": 0.85,
    "GS03": 0.65,
}

SATELLITES = [
    "SAT01",
    "SAT02",
    "SAT03",
]

MISSION_PHASES = {
    "NOMINAL": 1.00,
    "PASS": 1.50,
    "COMMAND": 1.30,
    "DOWNLINK": 2.00,
    "MAINTENANCE": 0.60,
}

PROTOCOLS = ["TCP", "UDP", "ICMP"]

SERVICES = [
    "AUTH",
    "TELEMETRY",
    "TELECOMMAND",
    "FILE_TRANSFER",
    "API",
    "MONITORING",
]

STATES = [
    "NORMAL",
    "ESTABLISHED",
    "CLOSED",
    "FAILED",
    "REJECTED",
    "TIMEOUT",
]

BENIGN_UNUSUAL = [
    "SCHEDULED_DOWNLINK",
    "GROUND_STATION_MAINTENANCE",
    "MISSION_EVENT_BURST",
    "TENANT_ONBOARDING",
]

THREATS = [
    "CREDENTIAL_ABUSE",
    "TELEMETRY_EXFILTRATION",
    "TELECOMMAND_ANOMALY",
    "SERVICE_FLOODING",
    "MULTI_TENANT_PROBING",
    "SESSION_MANIPULATION",
]

SEVERITIES = ["LOW", "MEDIUM", "HIGH"]


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def positive_normal(rng, mean, sd):
    """Draw a non-negative normally distributed value."""
    return max(0.0, rng.normal(mean, sd))


def positive_int(rng, mean, sd):
    """Draw a non-negative integer."""
    return max(0, int(round(rng.normal(mean, sd))))


def severity_multiplier(rng, severity):
    """
    Scenario-design perturbation ranges.

    These are experimental assumptions rather than empirical
    measurements of real GSaaS attacks.
    """
    if severity == "LOW":
        return rng.uniform(1.10, 1.50)

    if severity == "MEDIUM":
        return rng.uniform(1.50, 2.50)

    if severity == "HIGH":
        return rng.uniform(2.50, 5.00)

    return 1.0


def temporal_factor(t):
    """
    Temporal attack profile for a 100-event episode.

    0-39  : pre-event baseline
    40-49 : onset
    50-74 : progression
    75-89 : peak
    90-99 : recovery

    Returns a normalized perturbation intensity in [0, 1].
    """
    if t < 40:
        return 0.0

    if t < 50:
        return (t - 40) / 10.0

    if t < 75:
        return 0.50 + 0.50 * ((t - 50) / 25.0)

    if t < 90:
        return 1.0

    return max(0.0, 1.0 - ((t - 90) / 10.0))


# ============================================================
# 3. NORMAL GSaaS BEHAVIOUR
# ============================================================

def generate_normal_event(
    rng,
    tenant_id,
    ground_station_id,
    satellite_id,
    mission_phase,
):
    """
    Generate one context-conditioned benign GSaaS observation.

    Numerical values are simulation-design parameters. They are
    not claimed to represent measurements from a real provider.
    """

    tenant_multiplier = TENANTS[tenant_id]
    station_multiplier = GROUND_STATIONS[ground_station_id]
    mission_multiplier = MISSION_PHASES[mission_phase]

    activity = (
        tenant_multiplier
        * station_multiplier
        * mission_multiplier
        * rng.lognormal(mean=0.0, sigma=0.15)
    )

    protocol = rng.choice(
        PROTOCOLS,
        p=[0.65, 0.30, 0.05]
    )

    service = rng.choice(
        SERVICES,
        p=[0.10, 0.25, 0.10, 0.15, 0.30, 0.10]
    )

    state = rng.choice(
        STATES,
        p=[0.05, 0.75, 0.12, 0.03, 0.03, 0.02]
    )

    duration = positive_normal(
        rng,
        mean=20.0 * activity,
        sd=5.0
    )

    request_rate = positive_normal(
        rng,
        mean=8.0 * activity,
        sd=2.0
    )

    connection_rate = positive_normal(
        rng,
        mean=3.0 * activity,
        sd=1.0
    )

    failed_auth_rate = positive_normal(
        rng,
        mean=0.10 * activity,
        sd=0.08
    )

    unique_destinations = max(
        1,
        positive_int(
            rng,
            mean=2.0 * activity,
            sd=1.0
        )
    )

    packet_count = max(
        1,
        positive_int(
            rng,
            mean=120.0 * activity,
            sd=25.0
        )
    )

    mean_packet_size = rng.lognormal(
        mean=np.log(650),
        sigma=0.20
    )

    byte_count = packet_count * mean_packet_size

    telecommand_rate = positive_normal(
        rng,
        mean=0.40 * activity,
        sd=0.15
    )

    telemetry_rate = positive_normal(
        rng,
        mean=5.0 * activity,
        sd=1.2
    )

    file_transfer_rate = positive_normal(
        rng,
        mean=1.5 * activity,
        sd=0.5
    )

    api_request_rate = positive_normal(
        rng,
        mean=4.0 * activity,
        sd=1.0
    )

    resource_utilization = np.clip(
        rng.normal(
            loc=30.0 * activity,
            scale=7.0
        ),
        0,
        100
    )

    return {
        "protocol": protocol,
        "service": service,
        "state": state,
        "duration": duration,
        "packet_count": packet_count,
        "byte_count": byte_count,
        "request_rate": request_rate,
        "connection_rate": connection_rate,
        "failed_auth_rate": failed_auth_rate,
        "unique_destinations": unique_destinations,
        "telecommand_rate": telecommand_rate,
        "telemetry_rate": telemetry_rate,
        "file_transfer_rate": file_transfer_rate,
        "api_request_rate": api_request_rate,
        "resource_utilization": resource_utilization,
    }


# ============================================================
# 4. BENIGN-UNUSUAL PERTURBATIONS
# ============================================================

def apply_benign_unusual(event, event_type, rng):
    """
    Apply legitimate but unusual GSaaS operational behaviour.

    Ground truth remains benign.
    """
    e = event.copy()

    if event_type == "SCHEDULED_DOWNLINK":
        m = rng.uniform(1.8, 3.0)

        e["byte_count"] *= m
        e["file_transfer_rate"] *= m
        e["telemetry_rate"] *= rng.uniform(1.5, 2.5)
        e["resource_utilization"] *= rng.uniform(1.2, 1.8)

        e["service"] = "FILE_TRANSFER"

    elif event_type == "GROUND_STATION_MAINTENANCE":
        e["api_request_rate"] *= rng.uniform(1.5, 2.5)
        e["connection_rate"] *= rng.uniform(1.2, 2.0)
        e["request_rate"] *= rng.uniform(1.2, 2.0)
        e["resource_utilization"] *= rng.uniform(1.1, 1.6)

        e["service"] = "MONITORING"

    elif event_type == "MISSION_EVENT_BURST":
        e["telecommand_rate"] *= rng.uniform(1.5, 2.5)
        e["telemetry_rate"] *= rng.uniform(1.5, 2.5)
        e["request_rate"] *= rng.uniform(1.2, 2.0)
        e["packet_count"] *= rng.uniform(1.3, 2.0)

    elif event_type == "TENANT_ONBOARDING":
        e["api_request_rate"] *= rng.uniform(1.5, 3.0)
        e["failed_auth_rate"] *= rng.uniform(1.2, 2.0)
        e["connection_rate"] *= rng.uniform(1.3, 2.0)
        e["unique_destinations"] += int(rng.integers(1, 4))

        e["service"] = "AUTH"

    e["resource_utilization"] = np.clip(
        e["resource_utilization"],
        0,
        100
    )

    return e


# ============================================================
# 5. THREAT PERTURBATIONS
# ============================================================

def apply_threat(event, threat_type, severity, time_factor, rng):
    """
    Apply threat-specific perturbations to an otherwise normal event.
    """
    e = event.copy()

    if time_factor <= 0:
        return e

    s = severity_multiplier(rng, severity)

    # Convert severity into a time-dependent multiplier.
    m = 1.0 + (s - 1.0) * time_factor

    if threat_type == "CREDENTIAL_ABUSE":

        e["failed_auth_rate"] *= m ** 2
        e["request_rate"] *= m
        e["connection_rate"] *= m
        e["api_request_rate"] *= m
        e["unique_destinations"] += int(
            round(2 * (m - 1))
        )

        e["duration"] /= max(m, 1.0)
        e["service"] = "AUTH"

        if time_factor > 0.5:
            e["state"] = rng.choice(
                ["FAILED", "REJECTED", "TIMEOUT"]
            )

    elif threat_type == "TELEMETRY_EXFILTRATION":

        e["byte_count"] *= m ** 2
        e["file_transfer_rate"] *= m ** 2
        e["telemetry_rate"] *= m
        e["request_rate"] *= m
        e["connection_rate"] *= m
        e["unique_destinations"] += int(
            round(2 * (m - 1))
        )

        e["service"] = "FILE_TRANSFER"

    elif threat_type == "TELECOMMAND_ANOMALY":

        e["telecommand_rate"] *= m ** 2
        e["request_rate"] *= m
        e["connection_rate"] *= m
        e["packet_count"] *= m
        e["api_request_rate"] *= m

        e["service"] = "TELECOMMAND"

    elif threat_type == "SERVICE_FLOODING":

        e["request_rate"] *= m ** 2
        e["connection_rate"] *= m ** 2
        e["packet_count"] *= m ** 2
        e["byte_count"] *= m
        e["resource_utilization"] *= (
   	1.0 + 0.75 * (m - 1.0)
	)

        e["duration"] /= max(m, 1.0)
        e["service"] = "API"

    elif threat_type == "MULTI_TENANT_PROBING":

        e["unique_destinations"] += max(
            1,
            int(round(5 * (m - 1)))
        )

        e["connection_rate"] *= m
        e["request_rate"] *= m
        e["failed_auth_rate"] *= m
        e["api_request_rate"] *= m

        e["duration"] /= max(m, 1.0)
        e["service"] = "API"

        if time_factor > 0.5:
            e["state"] = rng.choice(
                ["FAILED", "REJECTED", "TIMEOUT"]
            )

    elif threat_type == "SESSION_MANIPULATION":

        e["connection_rate"] *= m
        e["request_rate"] *= m
        e["failed_auth_rate"] *= m
        e["unique_destinations"] += int(
            round(m - 1)
        )

        if time_factor > 0.3:
            e["state"] = rng.choice(
                ["FAILED", "REJECTED", "TIMEOUT", "CLOSED"]
            )

        # Introduce variable session duration.
        if rng.random() < 0.5:
            e["duration"] *= m
        else:
            e["duration"] /= max(m, 1.0)

    # General threat-related resource effect.
    e["resource_utilization"] *= (
        1.0 + 0.20 * (m - 1.0)
    )

    # Keep selected fields physically meaningful.
    e["resource_utilization"] = np.clip(
        e["resource_utilization"],
        0,
        100
    )

    e["packet_count"] = max(
        1,
        int(round(e["packet_count"]))
    )

    e["unique_destinations"] = max(
        1,
        int(round(e["unique_destinations"]))
    )

    return e


# ============================================================
# 6. EPISODE GENERATOR
# ============================================================

def generate_episode(
    episode_number,
    episode_type,
    rng,
    assigned_threat=None,
    assigned_benign_type=None,
    assigned_severity=None,
):
    """
    Generate one 100-event GSaaS episode.
    """

    episode_id = f"EP{episode_number:04d}"

    tenant_id = rng.choice(list(TENANTS.keys()))
    ground_station_id = rng.choice(
        list(GROUND_STATIONS.keys())
    )
    satellite_id = rng.choice(SATELLITES)

    benign_type = None
    threat_type = None
    severity = "NONE"

    benign_type = "NONE"

    if episode_type == "BENIGN_UNUSUAL":
        if assigned_benign_type is not None:
            benign_type = assigned_benign_type
        else:
            benign_type = rng.choice(BENIGN_UNUSUAL)

    elif episode_type == "THREAT":
        if assigned_threat is not None:
            threat_type = assigned_threat
        else:
            threat_type = rng.choice(THREATS)

    if assigned_severity is not None:
       severity = assigned_severity
    else:
           severity = rng.choice(
           SEVERITIES,
           p=[0.35, 0.40, 0.25]
        )

    rows = []

    for t in range(EVENTS_PER_EPISODE):

        mission_phase = rng.choice(
            list(MISSION_PHASES.keys()),
            p=[0.35, 0.20, 0.15, 0.20, 0.10]
        )

        event = generate_normal_event(
            rng,
            tenant_id,
            ground_station_id,
            satellite_id,
            mission_phase,
        )

        tf = temporal_factor(t)

        # ----------------------------------------
        # Normal episode
        # ----------------------------------------

        event_class = "BENIGN"
        attack_type = "NONE"
        is_anomaly = 0

        # ----------------------------------------
        # Benign unusual episode
        # ----------------------------------------

        if episode_type == "BENIGN_UNUSUAL":

            # Apply unusual behaviour during the
            # active portion of the episode.
            if tf > 0:
                event = apply_benign_unusual(
                    event,
                    benign_type,
                    rng
                )

            attack_type = "NONE"

        # ----------------------------------------
        # Threat episode
        # ----------------------------------------

        elif episode_type == "THREAT":

            if tf > 0:
                event = apply_threat(
                    event,
                    threat_type,
                    severity,
                    tf,
                    rng
                )

                event_class = "THREAT"
                attack_type = threat_type
                is_anomaly = 1

        row = {
            "event_id": None,
            "timestamp": t,
            "tenant_id": tenant_id,
            "ground_station_id": ground_station_id,
            "satellite_id": satellite_id,
            "mission_phase": mission_phase,
            "session_id": f"{episode_id}_SESSION",
            **event,
            "event_class": event_class,
            "attack_type": attack_type,
            "severity": (
                severity
                if is_anomaly == 1
                else "NONE"
            ),
            "is_anomaly": is_anomaly,
            "episode_id": episode_id,
            "episode_type": episode_type,
"benign_event_type": benign_type,
        }

        rows.append(row)

    return rows


# ============================================================
# 7. COMPLETE DATASET GENERATOR
# ============================================================

def generate_gsaas_dataset(seed=42):
    """
    Generate the complete 100,000-event GSaaS dataset.

    Episode composition is explicitly stratified:
        - 700 normal benign episodes
        - 150 benign-unusual episodes
        - 150 threat episodes
        - exactly 25 episodes for each of the six threat classes
    """

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Exact stratified episode composition
    # --------------------------------------------------------

    episode_specs = []

    # 700 ordinary benign episodes.
    for _ in range(700):
        episode_specs.append(
            ("NORMAL", None, None, None)
        )

    # --------------------------------------------------------
    # 150 benign-unusual episodes.
    #
    # Four benign-unusual conditions cannot be divided
    # perfectly equally into 150 episodes, so use:
    # 38 + 38 + 37 + 37 = 150
    # --------------------------------------------------------

    benign_counts = [38, 38, 37, 37]

    for benign_type, count in zip(
        BENIGN_UNUSUAL,
        benign_counts
    ):
        for _ in range(count):
            episode_specs.append(
                (
                    "BENIGN_UNUSUAL",
                    None,
                    benign_type,
                    None
                )
            )

        # --------------------------------------------------------
    # 150 threat episodes:
    # exactly 25 episodes for each of the six threats.
    #
    # Severity is balanced within each threat class:
    # 8 LOW + 9 MEDIUM + 8 HIGH = 25 episodes.
    # --------------------------------------------------------

    for threat_type in THREATS:

        severity_schedule = (
            ["LOW"] * 8
            + ["MEDIUM"] * 9
            + ["HIGH"] * 8
        )

        # Randomize severity order within this threat class.
        rng.shuffle(severity_schedule)

        for severity in severity_schedule:
            episode_specs.append(
                (
                    "THREAT",
                    threat_type,
                    None,
                    severity
                )
            )

    # Sanity check before generation.
    assert len(episode_specs) == N_EPISODES

    # Randomize episode order while preserving exact composition.
    rng.shuffle(episode_specs)

    rows = []

    # --------------------------------------------------------
    # Generate all episodes
    # --------------------------------------------------------

    for episode_number, spec in enumerate(
        episode_specs,
        start=1
    ):
       episode_type, threat_type, benign_type, severity = spec

       rows.extend(
    generate_episode(
        episode_number,
        episode_type,
        rng,
        assigned_threat=threat_type,
        assigned_benign_type=benign_type,
        assigned_severity=severity,
    )
)

    # --------------------------------------------------------
    # Construct dataframe
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    # Assign unique event identifiers.
    df["event_id"] = [
        f"EV{i:06d}"
        for i in range(1, len(df) + 1)
    ]

    return df

# ============================================================
# 8. EPISODE-LEVEL DATA SPLIT
# ============================================================

def split_by_episode(df, seed=42):
    """
    Stratified episode-level train/validation/test split.

    Target proportions:
        Train      = 60%
        Validation = 20%
        Test       = 20%

    Splitting is performed at episode level so that no episode
    can appear in more than one partition.

    Threat episodes are additionally stratified by attack type.
    Benign-unusual episodes are stratified by benign event type
    as closely as possible.
    """

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Build one metadata record per episode
    # --------------------------------------------------------

    episode_meta = (
        df.groupby("episode_id")
        .agg(
            episode_type=("episode_type", "first"),
            tenant_id=("tenant_id", "first"),
            ground_station_id=("ground_station_id", "first"),
        )
        .reset_index()
    )

    # Recover threat type for each threat episode.
    threat_map = (
        df.loc[df["attack_type"] != "NONE"]
        .groupby("episode_id")["attack_type"]
        .first()
        .to_dict()
    )

    episode_meta["threat_type"] = (
        episode_meta["episode_id"]
        .map(threat_map)
        .fillna("NONE")
    )

    # Recover benign-unusual subtype.
    #
    # The current dataframe does not contain a dedicated
    # benign-event-type column, so for this version we stratify
    # benign-unusual episodes as a single episode class.
    #
    # We will add an explicit benign subtype field later if
    # required for detailed benign-event analysis.

    train_ids = []
    validation_ids = []
    test_ids = []

    # --------------------------------------------------------
    # Helper: split a list approximately 60/20/20
    # --------------------------------------------------------

    def allocate(ids):
        ids = np.array(ids, dtype=object)
        rng.shuffle(ids)

        n = len(ids)

        n_train = int(round(0.60 * n))
        n_validation = int(round(0.20 * n))

        # Ensure all remaining episodes go to test.
        n_test_start = n_train + n_validation

        train = ids[:n_train]
        validation = ids[
            n_train:n_test_start
        ]
        test = ids[n_test_start:]

        return (
            train.tolist(),
            validation.tolist(),
            test.tolist(),
        )

    # --------------------------------------------------------
    # 1. Normal episodes
    # 700 -> 420 / 140 / 140
    # --------------------------------------------------------

    normal_ids = episode_meta.loc[
        episode_meta["episode_type"] == "NORMAL",
        "episode_id"
    ].tolist()

    tr, va, te = allocate(normal_ids)

    train_ids.extend(tr)
    validation_ids.extend(va)
    test_ids.extend(te)

    # --------------------------------------------------------
    # 2. Benign-unusual episodes
    # 150 -> 90 / 30 / 30
    # --------------------------------------------------------

    benign_unusual_ids = episode_meta.loc[
        episode_meta["episode_type"] == "BENIGN_UNUSUAL",
        "episode_id"
    ].tolist()

    tr, va, te = allocate(benign_unusual_ids)

    train_ids.extend(tr)
    validation_ids.extend(va)
    test_ids.extend(te)

    # --------------------------------------------------------
    # 3. Threat episodes
    #
    # Each threat has exactly 25 episodes:
    #     15 train
    #      5 validation
    #      5 test
    # --------------------------------------------------------

    for threat_type in THREATS:

        threat_ids = episode_meta.loc[
            (
                episode_meta["episode_type"] == "THREAT"
            )
            & (
                episode_meta["threat_type"] == threat_type
            ),
            "episode_id"
        ].tolist()

        assert len(threat_ids) == 25, (
            f"{threat_type}: expected 25 episodes, "
            f"found {len(threat_ids)}"
        )

        rng.shuffle(threat_ids)

        train_ids.extend(
            threat_ids[:15]
        )

        validation_ids.extend(
            threat_ids[15:20]
        )

        test_ids.extend(
            threat_ids[20:25]
        )

    # --------------------------------------------------------
    # Construct dataframe partitions
    # --------------------------------------------------------

    train_df = df[
        df["episode_id"].isin(train_ids)
    ].copy()

    validation_df = df[
        df["episode_id"].isin(validation_ids)
    ].copy()

    test_df = df[
        df["episode_id"].isin(test_ids)
    ].copy()

    # --------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------

    train_set = set(train_ids)
    validation_set = set(validation_ids)
    test_set = set(test_ids)

    assert train_set.isdisjoint(validation_set)
    assert train_set.isdisjoint(test_set)
    assert validation_set.isdisjoint(test_set)

    assert len(train_set) == 600
    assert len(validation_set) == 200
    assert len(test_set) == 200

    assert (
        len(
            train_set
            | validation_set
            | test_set
        )
        == 1000
    )

    return train_df, validation_df, test_df


# ============================================================
# 9. DATASET VALIDATION
# ============================================================

def validate_dataset(df):
    """
    Basic integrity checks.
    """

    assert len(df) == 100000, (
        f"Expected 100,000 events; found {len(df):,}"
    )

    assert df["event_id"].is_unique

    assert df["episode_id"].nunique() == 1000

    assert set(df["event_class"].unique()).issubset(
        {"BENIGN", "THREAT"}
    )

    assert set(df["is_anomaly"].unique()).issubset(
        {0, 1}
    )

    assert (
        df.loc[
            df["is_anomaly"] == 0,
            "event_class"
        ] == "BENIGN"
    ).all()

    assert (
        df.loc[
            df["is_anomaly"] == 1,
            "event_class"
        ] == "THREAT"
    ).all()

    numeric_nonnegative = [
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

    for col in numeric_nonnegative:
        assert (df[col] >= 0).all(), (
            f"Negative values found in {col}"
        )

    assert (
        df["resource_utilization"] <= 100
    ).all()


# ============================================================
# 10. SUMMARY
# ============================================================

def print_dataset_summary(df):
    """Print generator diagnostics."""

    print("\n" + "=" * 60)
    print("  GSaaS SYNTHETIC EVENT DATASET")
    print("=" * 60)

    print(f"\nEvents:   {len(df):,}")
    print(
        f"Episodes: {df['episode_id'].nunique():,}"
    )

    print("\nEpisode composition:")
    print(
        df.groupby("episode_type")[
            "episode_id"
        ].nunique()
    )

    print("\nEvent-level ground truth:")
    print(
        df["event_class"].value_counts()
    )

    print("\nThreat-event distribution:")
    print(
        df.loc[
            df["is_anomaly"] == 1,
            "attack_type"
        ].value_counts()
    )

    print("\nTenants:")
    print(df["tenant_id"].value_counts())

    print("\nGround stations:")
    print(
        df["ground_station_id"].value_counts()
    )


# ============================================================
# 11. STANDALONE EXECUTION
# ============================================================

if __name__ == "__main__":

    SEED = 42

    print(
        "\nGenerating GSaaS synthetic event dataset..."
    )

    gsaas_df = generate_gsaas_dataset(
        seed=SEED
    )

    validate_dataset(gsaas_df)

    train_df, validation_df, test_df = (
        split_by_episode(
            gsaas_df,
            seed=SEED
        )
    )

    print_dataset_summary(gsaas_df)

    print("\nEpisode-level split:")
    print(
        f"  Train:      "
        f"{train_df['episode_id'].nunique()} episodes, "
        f"{len(train_df):,} events"
    )

    print(
        f"  Validation: "
        f"{validation_df['episode_id'].nunique()} episodes, "
        f"{len(validation_df):,} events"
    )

    print(
        f"  Test:       "
        f"{test_df['episode_id'].nunique()} episodes, "
        f"{len(test_df):,} events"
    )

    print("\nThreat episodes by partition:")

    for split_name, split_df in [
        ("Train", train_df),
        ("Validation", validation_df),
        ("Test", test_df),
    ]:

        threat_episode_summary = (
            split_df.loc[
                split_df["attack_type"] != "NONE"
            ]
            .groupby("attack_type")["episode_id"]
            .nunique()
        )

        print(f"\n{split_name}:")
        print(threat_episode_summary)

    # Check that no episode appears in multiple partitions.
    train_eps = set(train_df["episode_id"])
    val_eps = set(validation_df["episode_id"])
    test_eps = set(test_df["episode_id"])

    assert train_eps.isdisjoint(val_eps)
    assert train_eps.isdisjoint(test_eps)
    assert val_eps.isdisjoint(test_eps)

    output_dir = Path("gsaas_generated_data")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    gsaas_df.to_csv(
        output_dir / "gsaas_events_seed42.csv",
        index=False
    )

    train_df.to_csv(
        output_dir / "gsaas_train_seed42.csv",
        index=False
    )

    validation_df.to_csv(
        output_dir / "gsaas_validation_seed42.csv",
        index=False
    )

    test_df.to_csv(
        output_dir / "gsaas_test_seed42.csv",
        index=False
    )


    # Confirm benign-unusual subtype metadata is retained.
    benign_subtypes = (
        gsaas_df.loc[
            gsaas_df["episode_type"] == "BENIGN_UNUSUAL",
            "benign_event_type"
        ]
        .value_counts()
    )

    assert "NONE" not in benign_subtypes.index, (
        "Some BENIGN_UNUSUAL events are missing "
        "their benign_event_type."
    )

    assert len(benign_subtypes) == len(BENIGN_UNUSUAL), (
        "Not all benign-unusual subtypes are present."
    )

    print("\nBenign-unusual subtype distribution:")
    print(benign_subtypes)

    print("\nDataset validation: PASSED")
   

    print(
        "\nFiles written to:"
    )

    print(
        f"  {output_dir.resolve()}"
    )