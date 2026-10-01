"""
GSaaS Cybersecurity Framework
Autoencoder Experiment
======================

Stage 1:
    Load the frozen GSaaS dataset through the established
    preprocessing pipeline and prepare benign training data.

The autoencoder itself will be added after this stage passes.
"""

import os
import random
import pandas as pd
from sklearn.metrics import confusion_matrix

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42
AE_MODEL_PATH = "gsaas_autoencoder_seed42.keras"

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from gsaas_preprocessing import (
    load_splits,
    validate_feature_schema,
    preprocess_all_splits,
    extract_labels,
)


# ============================================================
# 1. LOAD AND PREPROCESS FROZEN DATA
# ============================================================

def prepare_experiment_data():

    print("\n" + "=" * 70)
    print("GSaaS AUTOENCODER EXPERIMENT")
    print("=" * 70)

    print("\nLoading frozen dataset...")

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

    # --------------------------------------------------------
    # Extract labels/metadata separately.
    # These NEVER enter the autoencoder input.
    # --------------------------------------------------------

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
    # Autoencoder training data:
    # benign TRAINING observations only.
    # --------------------------------------------------------

    train_benign_mask = (
        train_labels["is_anomaly"] == 0
    )

    X_train_benign = X_train[
        train_benign_mask
    ]

    # --------------------------------------------------------
    # Benign validation observations.
    # These will later determine thresholds.
    # They are NOT used to fit the AE.
    # --------------------------------------------------------

    validation_benign_mask = (
        validation_labels["is_anomaly"] == 0
    )

    X_validation_benign = X_validation[
        validation_benign_mask
    ]

    return {
        "train_df": train_df,
        "validation_df": validation_df,
        "test_df": test_df,

        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,

        "X_train_benign":
            X_train_benign,

        "X_validation_benign":
            X_validation_benign,

        "train_labels":
            train_labels,

        "validation_labels":
            validation_labels,

        "test_labels":
            test_labels,

        "encoder": encoder,
        "scaler": scaler,
        "feature_names": feature_names,
    }


# ============================================================
# 2. VERIFY EXPERIMENT INPUT
# ============================================================

def validate_experiment_data(data):

    print("\n" + "=" * 70)
    print("AUTOENCODER INPUT VALIDATION")
    print("=" * 70)

    X_train_benign = data[
        "X_train_benign"
    ]

    X_validation_benign = data[
        "X_validation_benign"
    ]

    X_test = data["X_test"]

    print(
        f"AE input dimensions:       "
        f"{X_train_benign.shape[1]}"
    )

    print(
        f"AE training observations:  "
        f"{len(X_train_benign):,}"
    )

    print(
        f"Benign validation events:  "
        f"{len(X_validation_benign):,}"
    )

    print(
        f"Untouched test events:     "
        f"{len(X_test):,}"
    )

    # Expected values from frozen experimental design.

    assert X_train_benign.shape == (
        54690,
        27,
    )

    assert X_validation_benign.shape == (
        18230,
        27,
    )

    assert X_test.shape == (
        20000,
        27,
    )

    assert not np.isnan(
        X_train_benign
    ).any()

    assert not np.isinf(
        X_train_benign
    ).any()

    print(
        "\nAE training contamination "
        "check: BENIGN ONLY"
    )

    print(
        "Test-set training exposure: NONE"
    )

    print(
        "\nExperiment input validation: PASSED"
    )


# ============================================================
# 3. AUTOENCODER ARCHITECTURE
# ============================================================

def build_autoencoder(input_dim):

    inputs = Input(
        shape=(input_dim,),
        name="ae_input",
    )

    # Encoder
    x = Dense(
        16,
        activation="relu",
        name="encoder_1",
    )(inputs)

    x = Dense(
        8,
        activation="relu",
        name="encoder_2",
    )(x)

    latent = Dense(
        4,
        activation="relu",
        name="latent",
    )(x)

    # Decoder
    x = Dense(
        8,
        activation="relu",
        name="decoder_1",
    )(latent)

    x = Dense(
        16,
        activation="relu",
        name="decoder_2",
    )(x)

    outputs = Dense(
        input_dim,
        activation="sigmoid",
        name="reconstruction",
    )(x)

    autoencoder = Model(
        inputs=inputs,
        outputs=outputs,
        name="gsaas_autoencoder",
    )

    autoencoder.compile(
        optimizer="adam",
        loss="mse",
    )

    return autoencoder


# ============================================================
# 4. TRAIN AUTOENCODER
# ============================================================

def train_autoencoder(data):

    X_train_benign = data[
        "X_train_benign"
    ]

    X_validation_benign = data[
        "X_validation_benign"
    ]

    input_dim = X_train_benign.shape[1]

    print("\n" + "=" * 70)
    print("AUTOENCODER TRAINING")
    print("=" * 70)

    autoencoder = build_autoencoder(
        input_dim
    )

    autoencoder.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        min_delta=1e-6,
        verbose=1,
    )

    history = autoencoder.fit(
        X_train_benign,
        X_train_benign,
        validation_data=(
            X_validation_benign,
            X_validation_benign,
        ),
        epochs=100,
        batch_size=256,
        shuffle=True,
        callbacks=[early_stopping],
        verbose=2,
    )

    autoencoder.save(
        AE_MODEL_PATH
    )

    print(
        f"\nSaved autoencoder model: "
        f"{AE_MODEL_PATH}"
    )

    return autoencoder, history

# ============================================================
# 5. TRAINING SUMMARY
# ============================================================

def summarize_training(history):

    train_loss = history.history[
        "loss"
    ]

    validation_loss = history.history[
        "val_loss"
    ]

    best_epoch = (
        int(np.argmin(validation_loss)) + 1
    )

    best_validation_loss = float(
        np.min(validation_loss)
    )

    corresponding_train_loss = float(
        train_loss[best_epoch - 1]
    )

    print("\n" + "=" * 70)
    print("AUTOENCODER TRAINING SUMMARY")
    print("=" * 70)

    print(
        f"Epochs executed:          "
        f"{len(train_loss)}"
    )

    print(
        f"Best epoch:               "
        f"{best_epoch}"
    )

    print(
        f"Training loss at best:    "
        f"{corresponding_train_loss:.8f}"
    )

    print(
        f"Best validation loss:     "
        f"{best_validation_loss:.8f}"
    )

    print(
        "\nBest model weights restored."
    )

# ============================================================
# 6. RECONSTRUCTION ERROR
# ============================================================

def reconstruction_errors(
    autoencoder,
    X,
):

    reconstructed = autoencoder.predict(
        X,
        batch_size=256,
        verbose=0,
    )

    errors = np.mean(
        np.square(X - reconstructed),
        axis=1,
    )

    return errors


# ============================================================
# 7. VALIDATION RECONSTRUCTION-ERROR ANALYSIS
# ============================================================

def analyze_validation_errors(
    autoencoder,
    data,
):

    print("\n" + "=" * 70)
    print("VALIDATION RECONSTRUCTION-ERROR ANALYSIS")
    print("=" * 70)

    X_validation = data[
        "X_validation"
    ]

    validation_labels = data[
        "validation_labels"
    ]

    # Reconstruction error for every validation event.
    validation_errors = reconstruction_errors(
        autoencoder,
        X_validation,
    )

    y_validation = validation_labels[
        "is_anomaly"
    ]

    benign_errors = validation_errors[
        y_validation == 0
    ]

    threat_errors = validation_errors[
        y_validation == 1
    ]

    print(
        f"All validation events:     "
        f"{len(validation_errors):,}"
    )

    print(
        f"Benign validation events:  "
        f"{len(benign_errors):,}"
    )

    print(
        f"Threat validation events:  "
        f"{len(threat_errors):,}"
    )

    print("\nBenign reconstruction error:")

    print(
        f"  Mean:   "
        f"{np.mean(benign_errors):.8f}"
    )

    print(
        f"  Median: "
        f"{np.median(benign_errors):.8f}"
    )

    print(
        f"  P95:    "
        f"{np.percentile(benign_errors, 95):.8f}"
    )

    print(
        f"  P99:    "
        f"{np.percentile(benign_errors, 99):.8f}"
    )

    print("\nThreat reconstruction error:")

    print(
        f"  Mean:   "
        f"{np.mean(threat_errors):.8f}"
    )

    print(
        f"  Median: "
        f"{np.median(threat_errors):.8f}"
    )

    print(
        f"  P95:    "
        f"{np.percentile(threat_errors, 95):.8f}"
    )

    print(
        f"  P99:    "
        f"{np.percentile(threat_errors, 99):.8f}"
    )

    return (
        validation_errors,
        benign_errors,
        threat_errors,
    )

# ============================================================
# 8. THRESHOLD CALIBRATION
# ============================================================

THRESHOLD_PERCENTILES = [
    75.0,
    80.0,
    85.0,
    90.0,
    95.0,
    97.5,
    99.0,
]


def calibrate_thresholds(
    benign_validation_errors,
):

    print("\n" + "=" * 70)
    print("VALIDATION-BASED THRESHOLD CALIBRATION")
    print("=" * 70)

    thresholds = {}

    for percentile in THRESHOLD_PERCENTILES:

        threshold = float(
            np.percentile(
                benign_validation_errors,
                percentile,
            )
        )

        thresholds[percentile] = threshold

        print(
            f"P{percentile:<5g} "
            f"threshold = "
            f"{threshold:.8f}"
        )

    return thresholds


# ============================================================
# 9. TEST-SET THRESHOLD EVALUATION
# ============================================================

def evaluate_test_thresholds(
    autoencoder,
    data,
    thresholds,
):

    print("\n" + "=" * 70)
    print("TEST-SET THRESHOLD EVALUATION")
    print("=" * 70)

    X_test = data["X_test"]

    y_test = data[
        "test_labels"
    ]["is_anomaly"]

    # --------------------------------------------------------
    # Calculate reconstruction errors ONCE.
    # Every threshold is evaluated against these same scores.
    # --------------------------------------------------------

    test_errors = reconstruction_errors(
        autoencoder,
        X_test,
    )

    results = []

    for percentile, threshold in (
        thresholds.items()
    ):

        y_pred = (
            test_errors > threshold
        ).astype(np.int32)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        ).ravel()

        tpr = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        f1 = (
            2 * precision * tpr
            / (precision + tpr)
            if (precision + tpr) > 0
            else 0.0
        )

        results.append(
            {
                "percentile": percentile,
                "threshold": threshold,
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
                "TPR": tpr,
                "FPR": fpr,
                "Precision": precision,
                "F1": f1,
            }
        )

    results_df = pd.DataFrame(results)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "threshold":
                    lambda x: f"{x:.8f}",

                "TPR":
                    lambda x: f"{100*x:.2f}%",

                "FPR":
                    lambda x: f"{100*x:.2f}%",

                "Precision":
                    lambda x: f"{100*x:.2f}%",

                "F1":
                    lambda x: f"{x:.4f}",
            },
        )
    )

    return test_errors, results_df

# ============================================================
# 10. FALSE-POSITIVE ANALYSIS BY BENIGN EVENT TYPE
# ============================================================

def analyze_false_positives_by_benign_type(
    data,
    test_errors,
    thresholds,
):

    print("\n" + "=" * 70)
    print("FALSE-POSITIVE ANALYSIS BY BENIGN EVENT TYPE")
    print("=" * 70)

    test_df = data["test_df"]

    y_test = data[
        "test_labels"
    ]["is_anomaly"]

    benign_mask = (
        y_test == 0
    )

    benign_df = (
        test_df.loc[benign_mask]
        .copy()
        .reset_index(drop=True)
    )

    benign_errors = test_errors[
        benign_mask
    ]

    benign_df[
        "reconstruction_error"
    ] = benign_errors

    rows = []

    for percentile, threshold in (
        thresholds.items()
    ):

        predicted_anomaly = (
            benign_errors > threshold
        )

        # ----------------------------------------------------
        # Ordinary NORMAL traffic
        # ----------------------------------------------------

        normal_mask = (
            benign_df["episode_type"]
            == "NORMAL"
        ).to_numpy()

        normal_total = int(
            normal_mask.sum()
        )

        normal_fp = int(
            (
                predicted_anomaly
                & normal_mask
            ).sum()
        )

        normal_fpr = (
            normal_fp / normal_total
            if normal_total > 0
            else 0.0
        )

        rows.append(
            {
                "percentile": percentile,
                "benign_type": "NORMAL",
                "events": normal_total,
                "false_positives": normal_fp,
                "FPR": normal_fpr,
            }
        )

        # ----------------------------------------------------
        # BENIGN_UNUSUAL subtypes
        # ----------------------------------------------------

        unusual_mask = (
            benign_df["episode_type"]
            == "BENIGN_UNUSUAL"
        )

        unusual_types = sorted(
            benign_df.loc[
                unusual_mask,
                "benign_event_type",
            ].unique()
        )

        for benign_type in unusual_types:

            subtype_mask = (
                benign_df[
                    "benign_event_type"
                ]
                == benign_type
            ).to_numpy()

            subtype_total = int(
                subtype_mask.sum()
            )

            subtype_fp = int(
                (
                    predicted_anomaly
                    & subtype_mask
                ).sum()
            )

            subtype_fpr = (
                subtype_fp / subtype_total
                if subtype_total > 0
                else 0.0
            )

            rows.append(
                {
                    "percentile": percentile,
                    "benign_type": benign_type,
                    "events": subtype_total,
                    "false_positives": subtype_fp,
                    "FPR": subtype_fpr,
                }
            )

    fp_df = pd.DataFrame(rows)

    for percentile in thresholds:

        print(
            f"\n--- P{percentile:g} ---"
        )

        subset = fp_df[
            fp_df["percentile"]
            == percentile
        ].copy()

        print(
            subset[
                [
                    "benign_type",
                    "events",
                    "false_positives",
                    "FPR",
                ]
            ].to_string(
                index=False,
                formatters={
                    "FPR":
                        lambda x:
                        f"{100*x:.2f}%"
                },
            )
        )

    return fp_df

# ============================================================
# 11. THREAT-SPECIFIC DETECTION ANALYSIS
# ============================================================

def analyze_detection_by_threat_type(
    data,
    test_errors,
    thresholds,
):

    print("\n" + "=" * 70)
    print("THREAT-SPECIFIC DETECTION ANALYSIS")
    print("=" * 70)

    test_df = data["test_df"]

    y_test = data[
        "test_labels"
    ]["is_anomaly"]

    threat_mask = (
        y_test == 1
    )

    threat_df = (
        test_df.loc[threat_mask]
        .copy()
        .reset_index(drop=True)
    )

    threat_errors = test_errors[
        threat_mask
    ]

    threat_df[
        "reconstruction_error"
    ] = threat_errors

    threat_types = sorted(
        threat_df[
            "attack_type"
        ].unique()
    )

    rows = []

    for percentile, threshold in (
        thresholds.items()
    ):

        predicted_anomaly = (
            threat_errors > threshold
        )

        for threat_type in threat_types:

            type_mask = (
                threat_df[
                    "attack_type"
                ]
                == threat_type
            ).to_numpy()

            total = int(
                type_mask.sum()
            )

            detected = int(
                (
                    predicted_anomaly
                    & type_mask
                ).sum()
            )

            tpr = (
                detected / total
                if total > 0
                else 0.0
            )

            rows.append(
                {
                    "percentile":
                        percentile,

                    "attack_type":
                        threat_type,

                    "events":
                        total,

                    "detected":
                        detected,

                    "TPR":
                        tpr,
                }
            )

    threat_results_df = pd.DataFrame(
        rows
    )

    for percentile in thresholds:

        print(
            f"\n--- P{percentile:g} ---"
        )

        subset = threat_results_df[
            threat_results_df[
                "percentile"
            ] == percentile
        ].copy()

        print(
            subset[
                [
                    "attack_type",
                    "events",
                    "detected",
                    "TPR",
                ]
            ].to_string(
                index=False,
                formatters={
                    "TPR":
                        lambda x:
                        f"{100*x:.2f}%"
                },
            )
        )

    return threat_results_df

# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":

    data = prepare_experiment_data()

    validate_experiment_data(data)

    autoencoder, history = (
        train_autoencoder(data)
    )

    summarize_training(history)
    (
        validation_errors,
        benign_validation_errors,
        threat_validation_errors,
    ) = analyze_validation_errors(
        autoencoder,
        data,
    )
    thresholds = calibrate_thresholds(
        benign_validation_errors
    )
    (
        test_errors,
        test_results,
    ) = evaluate_test_thresholds(
        autoencoder,
        data,
        thresholds,
    )
    fp_by_benign_type = (
        analyze_false_positives_by_benign_type(
            data,
            test_errors,
            thresholds,
        )
    )
    threat_detection_results = (
        analyze_detection_by_threat_type(
            data,
            test_errors,
            thresholds,
        )
    )