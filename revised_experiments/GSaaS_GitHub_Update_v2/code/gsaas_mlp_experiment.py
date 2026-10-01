# ============================================================
# GSaaS MLP THREAT-CLASSIFICATION EXPERIMENT
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from gsaas_preprocessing import (
    load_splits,
    validate_feature_schema,
    preprocess_all_splits,
    extract_labels,
)
# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42
def reset_random_seeds(seed=SEED):

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

AE_MODEL_PATH = "gsaas_autoencoder_seed42.keras"
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = [
    "BENIGN",
    "CREDENTIAL_ABUSE",
    "MULTI_TENANT_PROBING",
    "SERVICE_FLOODING",
    "SESSION_MANIPULATION",
    "TELECOMMAND_ANOMALY",
    "TELEMETRY_EXFILTRATION",
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# ============================================================
# 3. BUILD MULTICLASS LABELS
# ============================================================

def build_multiclass_labels(labels):

    is_anomaly = np.asarray(
        labels["is_anomaly"]
    )

    attack_type = np.asarray(
        labels["attack_type"]
    )

    multiclass_labels = []

    for anomaly, attack in zip(
        is_anomaly,
        attack_type,
    ):

        if anomaly == 0:
            class_name = "BENIGN"

        else:
            class_name = str(attack)

        if class_name not in CLASS_TO_INDEX:

            raise ValueError(
                f"Unknown class encountered: "
                f"{class_name}"
            )

        multiclass_labels.append(
            CLASS_TO_INDEX[class_name]
        )

    return np.asarray(
        multiclass_labels,
        dtype=np.int32,
    )


# ============================================================
# 4. CLASS-DISTRIBUTION REPORT
# ============================================================

def report_class_distribution(
    split_name,
    y,
):

    print("\n" + "=" * 70)
    print(
        f"{split_name.upper()} CLASS DISTRIBUTION"
    )
    print("=" * 70)

    total = len(y)

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        count = int(
            np.sum(y == class_index)
        )

        percentage = (
            100.0 * count / total
        )

        print(
            f"{class_name:<25} "
            f"{count:>7,} "
            f"({percentage:6.2f}%)"
        )

    print("-" * 70)
    print(
        f"{'TOTAL':<25} "
        f"{total:>7,}"
    )


# ============================================================
# 5. COMPUTE CLASS WEIGHTS — TRAINING DATA ONLY
# ============================================================

def compute_training_class_weights(y_train):

    classes = np.arange(len(CLASS_NAMES))

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    class_weights = {
        int(class_index): float(weight)
        for class_index, weight in zip(
            classes,
            weights,
        )
    }

    print("\n" + "=" * 70)
    print("TRAINING CLASS WEIGHTS")
    print("=" * 70)

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):
        print(
            f"{class_name:<25} "
            f"{class_weights[class_index]:.4f}"
        )

    return class_weights

# ============================================================
# BUILD MLP_27 CLASSIFIER
# ============================================================

def build_mlp_27(input_dim, num_classes):

    model = Sequential(
        [
            tf.keras.Input(
                shape=(input_dim,)
            ),

            Dense(
                64,
                activation="relu",
            ),

            Dense(
                32,
                activation="relu",
            ),

            Dense(
                16,
                activation="relu",
            ),

            Dense(
                num_classes,
                activation="softmax",
            ),
        ],
        name="MLP_27",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
# ============================================================
# BUILD MLP_27+AE
# ============================================================

def build_mlp_27_ae(
    input_dim,
    num_classes,
):

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(input_dim,)
            ),

            tf.keras.layers.Dense(
                64,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                32,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                16,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                num_classes,
                activation="softmax",
            ),
        ],
        name="MLP_27_AE",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
# ============================================================
# TRAIN MLP_27+AE
# ============================================================

def train_mlp_27_ae(
    X_train,
    y_train,
    X_validation,
    y_validation,
    class_weights,
):

    print("\n" + "=" * 70)
    print("MLP_27+AE TRAINING")
    print("=" * 70)

    model = build_mlp_27_ae(
        input_dim=X_train.shape[1],
        num_classes=len(CLASS_NAMES),
    )

    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        min_delta=1e-5,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_validation,
            y_validation,
        ),
        epochs=100,
        batch_size=256,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1,
    )

    best_epoch = (
        np.argmin(
            history.history["val_loss"]
        )
        + 1
    )

    best_val_loss = np.min(
        history.history["val_loss"]
    )

    print("\n" + "=" * 70)
    print("MLP_27+AE TRAINING SUMMARY")
    print("=" * 70)

    print(
        f"Epochs executed: "
        f"{len(history.history['loss'])}"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.8f}"
    )

    print(
        "Best model weights restored: YES"
    )

    model.save(
        "gsaas_mlp27ae_seed42.keras"
    )

    print(
        "Saved model: "
        "gsaas_mlp27ae_seed42.keras"
    )

    return model, history

# ============================================================
# TRAIN MLP_27
# ============================================================

def train_mlp_27(
    X_train,
    y_train,
    X_validation,
    y_validation,
    class_weights,
):

    print("\n" + "=" * 70)
    print("MLP_27 TRAINING")
    print("=" * 70)

    model = build_mlp_27(
        input_dim=X_train.shape[1],
        num_classes=len(CLASS_NAMES),
    )

    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=1e-5,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_validation,
            y_validation,
        ),
        epochs=100,
        batch_size=256,
        class_weight=class_weights,
        callbacks=[early_stopping],
        shuffle=True,
        verbose=1,
    )

    best_epoch = (
        np.argmin(history.history["val_loss"]) + 1
    )

    best_val_loss = np.min(
        history.history["val_loss"]
    )

    print("\n" + "=" * 70)
    print("MLP_27 TRAINING SUMMARY")
    print("=" * 70)

    print(
        f"Epochs executed: "
        f"{len(history.history['loss'])}"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.8f}"
    )

    print(
        "Best model weights restored: YES"
    )

    return model, history
# ============================================================
# EVALUATE MLP_27
# ============================================================

def evaluate_mlp_27(
    model,
    X_test,
    y_test,
):

    print("\n" + "=" * 70)
    print("MLP_27 TEST-SET EVALUATION")
    print("=" * 70)

    probabilities = model.predict(
        X_test,
        batch_size=256,
        verbose=0,
    )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print(
        f"\nOverall accuracy:     "
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced accuracy:    "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Macro-F1:             "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted-F1:          "
        f"{weighted_f1:.4f}"
    )

    print("\nPER-CLASS PERFORMANCE")
    print("-" * 70)

    print(
        classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
    )

    print("CONFUSION MATRIX")
    print("Rows = Actual class | Columns = Predicted class")
    print("-" * 120)

    short_names = [
        "BENIGN",
        "CRED_ABUSE",
        "TENANT_PROBE",
        "FLOODING",
        "SESSION_MANIP",
        "TELECMD_ANOM",
        "TELEM_EXFIL",
    ]

    header = "ACTUAL \\ PRED".ljust(18) + "".join(
        f"{name:>15}"
        for name in short_names
    )

    print(header)
    print("-" * 123)

    for class_name, row in zip(
        short_names,
        cm,
    ):
        row_values = "".join(
            f"{value:>15}"
            for value in row
        )

        print(
            f"{class_name:<18}"
            f"{row_values}"
        )

    return {
        "y_pred": y_pred,
        "probabilities": probabilities,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
    }

# ============================================================
# EVALUATE MLP_27_AE
# ============================================================

def evaluate_mlp_27_ae(
    model,
    X_test,
    y_test,
):

    print("\n" + "=" * 70)
    print("MLP_27+AE TEST-SET EVALUATION")
    print("=" * 70)

    probabilities = model.predict(
        X_test,
        batch_size=256,
        verbose=0,
    )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print(
        f"\nOverall accuracy:     "
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced accuracy:    "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Macro-F1:             "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted-F1:          "
        f"{weighted_f1:.4f}"
    )

    print("\nPER-CLASS PERFORMANCE")
    print("-" * 70)

    print(
        classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
    )

    print("CONFUSION MATRIX")
    print("Rows = Actual class | Columns = Predicted class")
    print("-" * 120)

    short_names = [
        "BENIGN",
        "CRED_ABUSE",
        "TENANT_PROBE",
        "FLOODING",
        "SESSION_MANIP",
        "TELECMD_ANOM",
        "TELEM_EXFIL",
    ]

    header = "ACTUAL \\ PRED".ljust(18) + "".join(
        f"{name:>15}"
        for name in short_names
    )

    print(header)
    print("-" * 123)

    for class_name, row in zip(
        short_names,
        cm,
    ):
        row_values = "".join(
            f"{value:>15}"
            for value in row
        )

        print(
            f"{class_name:<18}"
            f"{row_values}"
        )

    return {
        "y_pred": y_pred,
        "probabilities": probabilities,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
    }

# ============================================================
# LOAD FROZEN AUTOENCODER
# ============================================================

def load_frozen_autoencoder():

    print("\n" + "=" * 70)
    print("LOADING FROZEN AUTOENCODER")
    print("=" * 70)

    autoencoder = tf.keras.models.load_model(
        AE_MODEL_PATH
    )

    print(
        f"Loaded autoencoder: "
        f"{AE_MODEL_PATH}"
    )

    print(
        f"Autoencoder input dimension: "
        f"{autoencoder.input_shape[1]}"
    )

    print(
        f"Autoencoder output dimension: "
        f"{autoencoder.output_shape[1]}"
    )

    return autoencoder

# ============================================================
# COMPUTE AUTOENCODER RECONSTRUCTION ERROR
# ============================================================

def compute_ae_reconstruction_error(
    autoencoder,
    X,
):

    reconstructed = autoencoder.predict(
        X,
        batch_size=256,
        verbose=0,
    )

    reconstruction_error = np.mean(
        np.square(
            X - reconstructed
        ),
        axis=1,
    )

    return reconstruction_error
# ============================================================
# PREPARE AE RECONSTRUCTION-ERROR FEATURE
# ============================================================

def prepare_ae_error_feature(
    autoencoder,
    X_train,
    X_validation,
    X_test,
):

    print("\n" + "=" * 70)
    print("AUTOENCODER RECONSTRUCTION-ERROR FEATURE")
    print("=" * 70)

    train_ae_error = compute_ae_reconstruction_error(
        autoencoder,
        X_train,
    )

    validation_ae_error = compute_ae_reconstruction_error(
        autoencoder,
        X_validation,
    )

    test_ae_error = compute_ae_reconstruction_error(
        autoencoder,
        X_test,
    )

    print(
        f"Training AE-error shape:   "
        f"{train_ae_error.shape}"
    )

    print(
        f"Validation AE-error shape: "
        f"{validation_ae_error.shape}"
    )

    print(
        f"Test AE-error shape:       "
        f"{test_ae_error.shape}"
    )

    print(
        f"\nTraining AE-error range:   "
        f"{train_ae_error.min():.8f} "
        f"to {train_ae_error.max():.8f}"
    )

    print(
        f"Validation AE-error range: "
        f"{validation_ae_error.min():.8f} "
        f"to {validation_ae_error.max():.8f}"
    )

    print(
        f"Test AE-error range:       "
        f"{test_ae_error.min():.8f} "
        f"to {test_ae_error.max():.8f}"
    )

    return (
        train_ae_error,
        validation_ae_error,
        test_ae_error,
    )
# ============================================================
# 6. PREPARE MLP DATA
# ============================================================

def prepare_mlp_data():

    print("\n" + "=" * 70)
    print("GSaaS MLP EXPERIMENT — DATA PREPARATION")
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

    train_labels = extract_labels(
        train_df
    )

    validation_labels = extract_labels(
        validation_df
    )

    test_labels = extract_labels(
        test_df
    )

    y_train = build_multiclass_labels(
        train_labels
    )

    y_validation = build_multiclass_labels(
        validation_labels
    )

    y_test = build_multiclass_labels(
        test_labels
    )

    return {
        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,

        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,

        "train_df": train_df,
        "validation_df": validation_df,
        "test_df": test_df,

        "encoder": encoder,
        "scaler": scaler,
        "feature_names": feature_names,
    }


# ============================================================
# 7. VALIDATE MLP INPUTS
# ============================================================

def validate_mlp_data(data):

    X_train = data["X_train"]
    X_validation = data["X_validation"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_validation = data["y_validation"]
    y_test = data["y_test"]

    print("\n" + "=" * 70)
    print("MLP INPUT VALIDATION")
    print("=" * 70)

    print(
        f"Training matrix:   "
        f"{X_train.shape}"
    )

    print(
        f"Validation matrix: "
        f"{X_validation.shape}"
    )

    print(
        f"Test matrix:       "
        f"{X_test.shape}"
    )

    assert X_train.shape == (60000, 27)
    assert X_validation.shape == (20000, 27)
    assert X_test.shape == (20000, 27)

    assert len(y_train) == 60000
    assert len(y_validation) == 20000
    assert len(y_test) == 20000

    assert not np.isnan(X_train).any()
    assert not np.isnan(X_validation).any()
    assert not np.isnan(X_test).any()

    assert not np.isinf(X_train).any()
    assert not np.isinf(X_validation).any()
    assert not np.isinf(X_test).any()

    assert set(np.unique(y_train)).issubset(
        set(range(len(CLASS_NAMES)))
    )

    print("\nFeature dimensions: 27")
    print("Classification classes: 7")
    print("MLP input validation: PASSED")


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":

    data = prepare_mlp_data()

    validate_mlp_data(data)

    report_class_distribution(
        "Train",
        data["y_train"],
    )

    report_class_distribution(
        "Validation",
        data["y_validation"],
    )

    report_class_distribution(
        "Test",
        data["y_test"],
    )
class_weights = compute_training_class_weights(
    data["y_train"]
)

frozen_autoencoder = load_frozen_autoencoder()

(
    train_ae_error,
    validation_ae_error,
    test_ae_error,
) = prepare_ae_error_feature(
    frozen_autoencoder,
    data["X_train"],
    data["X_validation"],
    data["X_test"],
)

# Create 28-feature matrices:
# original 27 features + AE reconstruction error
X_train_28 = np.column_stack(
    (
        data["X_train"],
        train_ae_error,
    )
)

X_validation_28 = np.column_stack(
    (
        data["X_validation"],
        validation_ae_error,
    )
)

X_test_28 = np.column_stack(
    (
        data["X_test"],
        test_ae_error,
    )
)

print("\n" + "=" * 70)
print("MLP_27+AE FEATURE VALIDATION")
print("=" * 70)

print(
    f"Training matrix:   "
    f"{X_train_28.shape}"
)

print(
    f"Validation matrix: "
    f"{X_validation_28.shape}"
)

print(
    f"Test matrix:       "
    f"{X_test_28.shape}"
)

assert X_train_28.shape == (60000, 28)
assert X_validation_28.shape == (20000, 28)
assert X_test_28.shape == (20000, 28)

assert not np.isnan(X_train_28).any()
assert not np.isnan(X_validation_28).any()
assert not np.isnan(X_test_28).any()

assert not np.isinf(X_train_28).any()
assert not np.isinf(X_validation_28).any()
assert not np.isinf(X_test_28).any()

print(
    "\nFeature #28: AE reconstruction error"
)

print(
    "MLP_27+AE feature validation: PASSED"
)

reset_random_seeds()
# ============================================================
# FINAL FROZEN OPERATING POINT — LAMBDA = 0.50
# ============================================================

FINAL_LAMBDA = 0.50

class_weights_lambda_050 = {
    class_id: (
        1.0
        + FINAL_LAMBDA
        * (weight - 1.0)
    )
    for class_id, weight in class_weights.items()
}

print("\n" + "=" * 70)
print("FINAL FROZEN CLASS WEIGHTS — LAMBDA = 0.50")
print("=" * 70)

for class_id, weight in class_weights_lambda_050.items():
    print(
        f"{CLASS_NAMES[class_id]:<25}"
        f"{weight:>10.4f}"
    )

# ============================================================
# FINAL MLP_27 — LAMBDA = 0.50
# ============================================================

reset_random_seeds()

model_27, history_27 = train_mlp_27(
    data["X_train"],
    data["y_train"],
    data["X_validation"],
    data["y_validation"],
    class_weights_lambda_050,
)

model_27.save(
    "gsaas_mlp27_lambda050_seed42.keras"
)

results_27 = evaluate_mlp_27(
    model_27,
    data["X_test"],
    data["y_test"],
)

# ============================================================
# FINAL MLP_27+AE — LAMBDA = 0.50
# ============================================================

reset_random_seeds()

model_27_ae, history_27_ae = train_mlp_27_ae(
    X_train_28,
    data["y_train"],
    X_validation_28,
    data["y_validation"],
    class_weights_lambda_050,
)

model_27_ae.save(
    "gsaas_mlp27_ae_lambda050_seed42.keras"
)

results_27_ae = evaluate_mlp_27_ae(
    model_27_ae,
    X_test_28,
    data["y_test"],
)
# ============================================================
# EXPORT MLP_27+AE TEST PREDICTIONS
# ============================================================

mlp_27_ae_pred = results_27_ae["y_pred"]

prediction_export = data["test_df"][
    [
        "event_id",
        "episode_id",
    ]
].copy()

prediction_export["mlp_27_ae_pred"] = mlp_27_ae_pred

prediction_export.to_csv(
    "gsaas_mlp27ae_test_predictions.csv",
    index=False,
)

print(
    "\nSaved MLP_27+AE test predictions to "
    "gsaas_mlp27ae_test_predictions.csv"
)
# ============================================================
# SIDE-BY-SIDE MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MLP_27 vs MLP_27+AE — TEST-SET COMPARISON")
print("=" * 70)

print(
    f"{'Metric':<25}"
    f"{'MLP_27':>15}"
    f"{'MLP_27+AE':>15}"
    f"{'Difference':>15}"
)

print("-" * 70)

def detection_metrics_from_cm(cm):

    benign_total = cm[0, :].sum()

    benign_fp = (
        benign_total
        - cm[0, 0]
    )

    benign_fpr = (
        benign_fp
        / benign_total
    )

    attack_total = cm[1:, :].sum()

    attacks_detected = (
        attack_total
        - cm[1:, 0].sum()
    )

    attack_recall = (
        attacks_detected
        / attack_total
    )

    return benign_fpr, attack_recall


fpr_27, attack_recall_27 = (
    detection_metrics_from_cm(
        results_27["confusion_matrix"]
    )
)

fpr_27_ae, attack_recall_27_ae = (
    detection_metrics_from_cm(
        results_27_ae["confusion_matrix"]
    )
)
comparison_metrics = [
    ("Accuracy", "accuracy"),
    ("Balanced accuracy", "balanced_accuracy"),
    ("Macro-F1", "macro_f1"),
    ("Weighted-F1", "weighted_f1"),
]

for metric_name, metric_key in comparison_metrics:

    baseline_value = results_27[
        metric_key
    ]

    ae_value = results_27_ae[
        metric_key
    ]

    difference = (
        ae_value - baseline_value
    )

    print(
        f"{metric_name:<25}"
        f"{baseline_value:>15.4f}"
        f"{ae_value:>15.4f}"
        f"{difference:>+15.4f}"
    )
print(
    f"{'Benign FPR':<25}"
    f"{fpr_27:>14.2%}"
    f"{fpr_27_ae:>14.2%}"
    f"{(fpr_27_ae - fpr_27):>+14.2%}"
)

print(
    f"{'Attack recall':<25}"
    f"{attack_recall_27:>14.2%}"
    f"{attack_recall_27_ae:>14.2%}"
    f"{(attack_recall_27_ae - attack_recall_27):>+14.2%}"
)
# ============================================================
# MLP_27 FALSE-POSITIVE ANALYSIS BY BENIGN EVENT TYPE
# ============================================================

print("\n" + "=" * 70)
print("MLP_27 FALSE-POSITIVE ANALYSIS BY BENIGN EVENT TYPE")
print("=" * 70)

# Predictions already produced by the baseline MLP_27.
y_pred_27 = results_27["y_pred"]

# Metadata for the untouched test events.
test_metadata = data["test_df"].copy()

# Add the MLP prediction to each test event.
test_metadata["mlp27_prediction"] = y_pred_27

# A false positive is:
# actual class = BENIGN
# predicted class = anything other than BENIGN
#
# BENIGN is class index 0.
test_metadata["mlp27_false_positive"] = (
    (data["y_test"] == 0)
    &
    (y_pred_27 != 0)
)

# Keep only genuinely benign test events.
benign_test = test_metadata.loc[
    data["y_test"] == 0
].copy()

# Convert the dataset metadata into an easy-to-read benign type.
#
# NORMAL episodes have benign_event_type == "NONE".
# BENIGN_UNUSUAL episodes retain their specific subtype.
benign_test["benign_type"] = np.where(
    benign_test["episode_type"] == "NORMAL",
    "NORMAL",
    benign_test["benign_event_type"],
)

# Calculate false positives for each benign operational state.
fp_by_type = (
    benign_test
    .groupby("benign_type")
    .agg(
        events=(
            "mlp27_false_positive",
            "size",
        ),
        false_positives=(
            "mlp27_false_positive",
            "sum",
        ),
    )
    .reset_index()
)

fp_by_type["FPR"] = (
    fp_by_type["false_positives"]
    /
    fp_by_type["events"]
)

print()

print(
    f"{'Benign event type':<32}"
    f"{'Events':>10}"
    f"{'False positives':>18}"
    f"{'FPR':>12}"
)

print("-" * 72)

for _, row in fp_by_type.iterrows():

    print(
        f"{row['benign_type']:<32}"
        f"{int(row['events']):>10}"
        f"{int(row['false_positives']):>18}"
        f"{row['FPR']:>11.2%}"
    )

print("-" * 72)

total_benign = len(
    benign_test
)

total_fp = int(
    benign_test[
        "mlp27_false_positive"
    ].sum()
)

overall_fpr = (
    total_fp / total_benign
)

print(
    f"{'TOTAL':<32}"
    f"{total_benign:>10}"
    f"{total_fp:>18}"
    f"{overall_fpr:>11.2%}"
)

# ============================================================
# INVESTIGATE BENIGN "NONE" EVENTS
# ============================================================

print("\n" + "=" * 70)
print('INVESTIGATION OF BENIGN "NONE" EVENTS')
print("=" * 70)

none_events = benign_test[
    benign_test["benign_event_type"] == "NONE"
]

print(
    "\nEpisode types containing benign_event_type = NONE:"
)

print(
    none_events["episode_type"]
    .value_counts()
    .to_string()
)

print(
    "\nTotal NONE events:",
    len(none_events)
)
# ============================================================
# EXPORT FINAL MLP_27+AE TEST PREDICTIONS FOR FAB ANALYSIS
# ============================================================

fab_prediction_export = pd.DataFrame({
    "event_id": data["test_df"]["event_id"].values,
    "actual_class": data["y_test"],
    "predicted_class": results_27_ae["y_pred"],
})

fab_prediction_export.to_csv(
    "gsaas_mlp27ae_test_predictions.csv",
    index=False,
)

print(
    "\nSaved event-level MLP_27+AE predictions to "
    "gsaas_mlp27ae_test_predictions.csv"
)