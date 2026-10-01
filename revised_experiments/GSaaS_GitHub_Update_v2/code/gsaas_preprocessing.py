"""
GSaaS Cybersecurity Framework
Preprocessing Pipeline
======================

Prepares the frozen GSaaS synthetic dataset for machine-learning
experiments.

Important:
    - Preprocessing is fitted on TRAINING DATA ONLY.
    - Validation and test data are transformed using the fitted
      training preprocessing objects.
    - Ground-truth and scenario metadata are excluded from model
      inputs.
    - No autoencoder is trained in this module.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


# ============================================================
# 1. DATA PATHS
# ============================================================

DATA_DIR = Path("gsaas_generated_data")

TRAIN_PATH = DATA_DIR / "gsaas_train_seed42.csv"
VALIDATION_PATH = DATA_DIR / "gsaas_validation_seed42.csv"
TEST_PATH = DATA_DIR / "gsaas_test_seed42.csv"


# ============================================================
# 2. DETECTOR FEATURE DEFINITION
# ============================================================

CATEGORICAL_FEATURES = [
    "protocol",
    "service",
    "state",
]

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

DETECTOR_FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)


# ============================================================
# 3. METADATA THAT MUST NOT ENTER THE DETECTOR
# ============================================================

EXCLUDED_METADATA = [
    "event_id",
    "timestamp",
    "tenant_id",
    "ground_station_id",
    "satellite_id",
    "mission_phase",
    "session_id",
    "event_class",
    "attack_type",
    "severity",
    "is_anomaly",
    "episode_id",
    "episode_type",
    "benign_event_type",
]


# ============================================================
# 4. LOAD DATA
# ============================================================

def load_splits():

    for path in [
        TRAIN_PATH,
        VALIDATION_PATH,
        TEST_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset not found:\n"
                f"{path.resolve()}"
            )

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(
        VALIDATION_PATH
    )
    test_df = pd.read_csv(TEST_PATH)

    return (
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# 5. VERIFY FEATURE SCHEMA
# ============================================================

def validate_feature_schema(
    train_df,
    validation_df,
    test_df,
):

    print("\n" + "=" * 70)
    print("FEATURE-SCHEMA VALIDATION")
    print("=" * 70)

    required = set(
        DETECTOR_FEATURES
        + EXCLUDED_METADATA
    )

    for name, df in [
        ("Train", train_df),
        ("Validation", validation_df),
        ("Test", test_df),
    ]:

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} is missing columns: "
                f"{sorted(missing)}"
            )

        print(
            f"{name:<12}: "
            f"{len(df):>7,} events, "
            f"{df['episode_id'].nunique():>4} episodes"
        )

    # Ensure the three partitions contain
    # completely separate episodes.

    train_eps = set(
        train_df["episode_id"]
    )

    validation_eps = set(
        validation_df["episode_id"]
    )

    test_eps = set(
        test_df["episode_id"]
    )

    assert train_eps.isdisjoint(
        validation_eps
    )

    assert train_eps.isdisjoint(
        test_eps
    )

    assert validation_eps.isdisjoint(
        test_eps
    )

    print(
        "\nEpisode leakage: NONE"
    )

    # Explicitly verify that metadata is not
    # accidentally present in detector features.

    leakage = (
        set(DETECTOR_FEATURES)
        & set(EXCLUDED_METADATA)
    )

    assert not leakage, (
        f"Metadata leakage detected: {leakage}"
    )

    print(
        "Metadata leakage into detector "
        "feature list: NONE"
    )


# ============================================================
# 6. FIT PREPROCESSING ON TRAIN ONLY
# ============================================================

def fit_preprocessor(train_df):

    print("\n" + "=" * 70)
    print("FITTING PREPROCESSOR — TRAIN ONLY")
    print("=" * 70)

    # --------------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------------

    categorical_train = train_df[
        CATEGORICAL_FEATURES
    ].astype(str)

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

    except TypeError:
        # Compatibility with older scikit-learn.
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )

    encoder.fit(categorical_train)

    # --------------------------------------------------------
    # Numerical preprocessing
    # --------------------------------------------------------

    numerical_train = train_df[
        NUMERIC_FEATURES
    ].astype(float)

    scaler = MinMaxScaler()

    scaler.fit(numerical_train)

    print(
        "Categorical encoder fitted on "
        f"{len(train_df):,} training events."
    )

    print(
        "Numerical scaler fitted on "
        f"{len(train_df):,} training events."
    )

    return encoder, scaler


# ============================================================
# 7. TRANSFORM ONE PARTITION
# ============================================================

def transform_split(
    df,
    encoder,
    scaler,
):

    categorical = (
        df[CATEGORICAL_FEATURES]
        .astype(str)
    )

    numerical = (
        df[NUMERIC_FEATURES]
        .astype(float)
    )

    X_cat = encoder.transform(
        categorical
    )

    X_num = scaler.transform(
        numerical
    )

    # Categorical first, then numerical.
    X = np.hstack(
        [
            X_cat,
            X_num,
        ]
    ).astype(np.float32)

    return X


# ============================================================
# 8. CONSTRUCT TRANSFORMED FEATURE NAMES
# ============================================================

def get_transformed_feature_names(
    encoder,
):

    categorical_names = (
        encoder.get_feature_names_out(
            CATEGORICAL_FEATURES
        ).tolist()
    )

    feature_names = (
        categorical_names
        + NUMERIC_FEATURES
    )

    return feature_names


# ============================================================
# 9. TRANSFORM ALL PARTITIONS
# ============================================================

def preprocess_all_splits(
    train_df,
    validation_df,
    test_df,
):

    encoder, scaler = fit_preprocessor(
        train_df
    )

    X_train = transform_split(
        train_df,
        encoder,
        scaler,
    )

    X_validation = transform_split(
        validation_df,
        encoder,
        scaler,
    )

    X_test = transform_split(
        test_df,
        encoder,
        scaler,
    )

    feature_names = (
        get_transformed_feature_names(
            encoder
        )
    )

    return (
        X_train,
        X_validation,
        X_test,
        encoder,
        scaler,
        feature_names,
    )


# ============================================================
# 10. PREPROCESSING VALIDATION
# ============================================================

def validate_transformed_data(
    X_train,
    X_validation,
    X_test,
    feature_names,
):

    print("\n" + "=" * 70)
    print("TRANSFORMED-DATA VALIDATION")
    print("=" * 70)

    print(
        f"Train shape:      {X_train.shape}"
    )

    print(
        f"Validation shape: "
        f"{X_validation.shape}"
    )

    print(
        f"Test shape:       {X_test.shape}"
    )

    assert (
        X_train.shape[1]
        == X_validation.shape[1]
        == X_test.shape[1]
    )

    assert (
        X_train.shape[1]
        == len(feature_names)
    )

    for name, X in [
        ("Train", X_train),
        ("Validation", X_validation),
        ("Test", X_test),
    ]:

        assert not np.isnan(X).any(), (
            f"{name} contains NaN values."
        )

        assert not np.isinf(X).any(), (
            f"{name} contains infinite values."
        )

    print(
        "\nNaN values: NONE"
    )

    print(
        "Infinite values: NONE"
    )

    print(
        f"Final detector dimensions: "
        f"{X_train.shape[1]}"
    )


# ============================================================
# 11. LABEL AND METADATA EXTRACTION
# ============================================================

def extract_labels(df):

    return {
        "is_anomaly":
            df["is_anomaly"]
            .to_numpy(dtype=np.int32),

        "attack_type":
            df["attack_type"]
            .astype(str)
            .to_numpy(),

        "severity":
            df["severity"]
            .astype(str)
            .to_numpy(),

        "episode_id":
            df["episode_id"]
            .astype(str)
            .to_numpy(),

        "episode_type":
            df["episode_type"]
            .astype(str)
            .to_numpy(),

        "benign_event_type":
            df["benign_event_type"]
            .astype(str)
            .to_numpy(),
    }


# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\nLoading frozen GSaaS dataset..."
    )

    (
        train_df,
        validation_df,
        test_df,
    ) = load_splits()

    validate_feature_schema(
        train_df,
        validation_df,
        test_df,
    )

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

    validate_transformed_data(
        X_train,
        X_validation,
        X_test,
        feature_names,
    )

    train_labels = extract_labels(
        train_df
    )

    validation_labels = extract_labels(
        validation_df
    )

    test_labels = extract_labels(
        test_df
    )

    # --------------------------------------------------------
    # Benign subsets for future AE training/calibration
    # --------------------------------------------------------

    train_benign_mask = (
        train_labels["is_anomaly"] == 0
    )

    validation_benign_mask = (
        validation_labels["is_anomaly"] == 0
    )

    X_train_benign = X_train[
        train_benign_mask
    ]

    X_validation_benign = X_validation[
        validation_benign_mask
    ]

    print("\n" + "=" * 70)
    print("BENIGN SUBSETS")
    print("=" * 70)

    print(
        f"Benign training events:   "
        f"{len(X_train_benign):,}"
    )

    print(
        f"Benign validation events: "
        f"{len(X_validation_benign):,}"
    )

    print("\nPreprocessing pipeline: PASSED")