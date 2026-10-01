import os
import random
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from sklearn.utils.class_weight import (
    compute_class_weight,
)

from gsaas_preprocessing import (
    load_splits,
    validate_feature_schema,
    preprocess_all_splits,
    extract_labels,
)


SEED = 42

CLASS_NAMES = [
    "BENIGN",
    "CREDENTIAL_ABUSE",
    "MULTI_TENANT_PROBING",
    "SERVICE_FLOODING",
    "SESSION_MANIPULATION",
    "TELECOMMAND_ANOMALY",
    "TELEMETRY_EXFILTRATION",
]


def reset_random_seeds(seed=SEED):

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
def build_mlp(input_dim, num_classes):

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
        name="MLP_27_WEIGHT_CALIBRATION",
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
# DATA PREPARATION
# ============================================================

print("\n" + "=" * 70)
print("GSaaS CLASS-WEIGHT CALIBRATION EXPERIMENT")
print("=" * 70)

print("\nLoading frozen dataset...")

train_df, validation_df, test_df = load_splits()

validate_feature_schema(
    train_df,
    validation_df,
    test_df,
)

preprocessed = preprocess_all_splits(
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


# Extract label dictionaries.
train_labels = extract_labels(train_df)
validation_labels = extract_labels(validation_df)
test_labels = extract_labels(test_df)


# Multiclass attack labels used by the MLP.
CLASS_TO_ID = {
    "NONE": 0,
    "CREDENTIAL_ABUSE": 1,
    "MULTI_TENANT_PROBING": 2,
    "SERVICE_FLOODING": 3,
    "SESSION_MANIPULATION": 4,
    "TELECOMMAND_ANOMALY": 5,
    "TELEMETRY_EXFILTRATION": 6,
}

y_train = np.array(
    [
        CLASS_TO_ID[label]
        for label in train_labels["attack_type"]
    ],
    dtype=np.int64,
)

y_validation = np.array(
    [
        CLASS_TO_ID[label]
        for label in validation_labels["attack_type"]
    ],
    dtype=np.int64,
)

y_test = np.array(
    [
        CLASS_TO_ID[label]
        for label in test_labels["attack_type"]
    ],
    dtype=np.int64,
)

print("\nEncoded training classes:")

unique_classes, class_counts = np.unique(
    y_train,
    return_counts=True,
)

for class_id, count in zip(
    unique_classes,
    class_counts,
):
    print(
        f"{class_id} = "
        f"{CLASS_NAMES[class_id]:<25} "
        f"{count:,}"
    )

print("\n" + "=" * 70)
print("CALIBRATION DATA VALIDATION")
print("=" * 70)

print(
    f"Training matrix:   {X_train.shape}"
)

print(
    f"Validation matrix: {X_validation.shape}"
)

print(
    f"Test matrix:       {X_test.shape}"
)

print(
    f"\nTraining labels:   {y_train.shape}"
)

print(
    f"Validation labels: {y_validation.shape}"
)

print(
    f"Test labels:       {y_test.shape}"
)

# ============================================================
# ORIGINAL TRAINING-DERIVED CLASS WEIGHTS
# ============================================================

classes = np.unique(y_train)

base_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train,
)

base_class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(
        classes,
        base_weights_array,
    )
}


print("\n" + "=" * 70)
print("ORIGINAL TRAINING-DERIVED CLASS WEIGHTS")
print("=" * 70)

for class_id in classes:

    print(
        f"{CLASS_NAMES[int(class_id)]:<25}"
        f"{base_class_weights[int(class_id)]:>10.4f}"
    )
# ============================================================
# CLASS-WEIGHT CALIBRATION GRID
# ============================================================

LAMBDA_VALUES = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]


def make_calibrated_weights(
    base_weights,
    lambda_value,
):

    calibrated = {}

    for class_id, base_weight in base_weights.items():

        calibrated[class_id] = (
            1.0
            +
            lambda_value
            *
            (base_weight - 1.0)
        )

    return calibrated


print("\n" + "=" * 70)
print("CLASS-WEIGHT CALIBRATION GRID")
print("=" * 70)

for lambda_value in LAMBDA_VALUES:

    weights = make_calibrated_weights(
        base_class_weights,
        lambda_value,
    )

    print(
        f"\nlambda = {lambda_value:.2f}"
    )

    for class_id in classes:

        print(
            f"  {CLASS_NAMES[int(class_id)]:<25}"
            f"{weights[int(class_id)]:>10.4f}"
        )
# ============================================================
# TRAIN ONE CALIBRATION MODEL
# ============================================================

def train_calibration_model(
    X_train,
    y_train,
    X_validation,
    y_validation,
    class_weights,
    lambda_value,
):

    reset_random_seeds()

    print("\n" + "=" * 70)
    print(
        f"TRAINING CALIBRATION MODEL — "
        f"lambda = {lambda_value:.2f}"
    )
    print("=" * 70)

    model = build_mlp(
        input_dim=X_train.shape[1],
        num_classes=len(CLASS_NAMES),
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=1e-5,
        restore_best_weights=True,
        verbose=0,
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
        verbose=0,
    )

    best_epoch = (
        np.argmin(history.history["val_loss"])
        + 1
    )

    best_val_loss = np.min(
        history.history["val_loss"]
    )

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

    return model
# ============================================================
# VALIDATION METRICS
# ============================================================

def evaluate_on_validation(
    model,
    X_validation,
    y_validation,
):

    probabilities = model.predict(
        X_validation,
        verbose=0,
    )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    cm = confusion_matrix(
        y_validation,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
    )

# Per-class recall.
    per_class_recall = np.diag(cm) / cm.sum(axis=1)
    accuracy = accuracy_score(
        y_validation,
        y_pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_validation,
        y_pred,
    )

    macro_f1 = f1_score(
        y_validation,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # BENIGN is class 0.
    benign_total = cm[0, :].sum()
    benign_false_positives = (
        benign_total - cm[0, 0]
    )

    benign_fpr = (
        benign_false_positives
        / benign_total
    )

    # Attack recall across all six attack classes together.
    attack_total = cm[1:, :].sum()

    attack_correctly_detected = (
        attack_total
        - cm[1:, 0].sum()
    )

    attack_recall = (
        attack_correctly_detected
        / attack_total
    )

    return {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "macro_f1": macro_f1,
    "benign_fpr": benign_fpr,
    "attack_recall": attack_recall,
    "per_class_recall": per_class_recall,
}

# ============================================================
# VALIDATION-ONLY CALIBRATION SWEEP
# ============================================================

calibration_results = []
model_lambda_050 = None

print("\n" + "=" * 70)
print("VALIDATION-ONLY CLASS-WEIGHT SWEEP")
print("=" * 70)

for lambda_value in LAMBDA_VALUES:

    calibrated_weights = make_calibrated_weights(
        base_class_weights,
        lambda_value,
    )

    model = train_calibration_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        calibrated_weights,
        lambda_value,
    )
    if np.isclose(
        lambda_value,
        0.50,
    ):
        model_lambda_050 = model
    metrics = evaluate_on_validation(
        model,
        X_validation,
        y_validation,
    )

    calibration_results.append(
        {
            "lambda": lambda_value,
            **metrics,
        }
    )
# ============================================================
# VALIDATION CALIBRATION TABLE
# ============================================================

print("\n" + "=" * 85)
print("CLASS-WEIGHT CALIBRATION — VALIDATION RESULTS")
print("=" * 85)

print(
    f"{'Lambda':>8}"
    f"{'Benign FPR':>15}"
    f"{'Attack Recall':>17}"
    f"{'Balanced Acc':>17}"
    f"{'Macro-F1':>15}"
)

print("-" * 85)

for result in calibration_results:

    print(
        f"{result['lambda']:>8.2f}"
        f"{result['benign_fpr']:>14.2%}"
        f"{result['attack_recall']:>16.2%}"
        f"{result['balanced_accuracy']:>17.4f}"
        f"{result['macro_f1']:>15.4f}"
    )
# ============================================================
# PER-CLASS ATTACK RECALL BY LAMBDA
# ============================================================

print("\n" + "=" * 105)
print("PER-CLASS ATTACK RECALL BY CLASS-WEIGHT STRENGTH — VALIDATION SET")
print("=" * 105)

short_names = [
    "CRED_ABUSE",
    "TENANT_PROBE",
    "FLOODING",
    "SESSION_MANIP",
    "TELECMD_ANOM",
    "TELEM_EXFIL",
]

print(
    f"{'Lambda':>8}"
    + "".join(
        f"{name:>16}"
        for name in short_names
    )
)

print("-" * 105)

for result in calibration_results:

    recalls = result[
        "per_class_recall"
    ]

    # Index 0 is BENIGN.
    # Attack classes are indices 1 through 6.
    attack_recalls = recalls[1:]

    print(
        f"{result['lambda']:>8.2f}"
        + "".join(
            f"{recall:>15.2%}"
            for recall in attack_recalls
        )
    )
# ============================================================
# DETAILED VALIDATION REPORT — LAMBDA 0.50
# ============================================================

print("\n" + "=" * 78)
print("DETAILED CLASSIFICATION REPORT — lambda = 0.50 — VALIDATION SET")
print("=" * 78)

probabilities_050 = model_lambda_050.predict(
    X_validation,
    verbose=0,
)

y_pred_050 = np.argmax(
    probabilities_050,
    axis=1,
)

print(
    classification_report(
        y_validation,
        y_pred_050,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
)