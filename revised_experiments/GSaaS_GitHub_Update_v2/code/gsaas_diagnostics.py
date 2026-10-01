# ============================================================
# GSaaS DIAGNOSTICS
# ============================================================

import numpy as np

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
print("GSaaS DIAGNOSTICS")
print("=" * 70)

print("\nLoading frozen dataset...")

train_df, validation_df, test_df = load_splits()

validate_feature_schema(
    train_df,
    validation_df,
    test_df,
)


# ============================================================
# 2. EXTRACT TEST LABELS
# ============================================================

labels_test = extract_labels(test_df)

y_test = labels_test["is_anomaly"]


# ============================================================
# 3. INVESTIGATE BENIGN NONE EVENTS
# ============================================================

print("\n" + "=" * 70)
print('INVESTIGATION OF BENIGN "NONE" EVENTS')
print("=" * 70)

# BENIGN is class index 0.
benign_mask = (
    y_test == 0
)

benign_test = test_df.loc[
    benign_mask
].copy()

none_events = benign_test.loc[
    benign_test["benign_event_type"] == "NONE"
].copy()


print(
    "\nEpisode types containing "
    'benign_event_type = "NONE":'
)

print(
    none_events["episode_type"]
    .value_counts()
    .to_string()
)


print(
    "\nTotal benign NONE events:",
    len(none_events)
)


print(
    "\nUnique episode IDs containing "
    "benign NONE events:",
    none_events["episode_id"].nunique()
)


print(
    "\nNONE events by episode type:"
)

summary = (
    none_events
    .groupby("episode_type")
    .agg(
        events=("episode_id", "size"),
        episodes=("episode_id", "nunique"),
    )
    .sort_values(
        "events",
        ascending=False,
    )
)

print(
    summary.to_string()
)

# ============================================================
# INSPECT FROZEN DATASET FEATURES AND METADATA
# ============================================================

print("\n" + "=" * 78)
print("FROZEN DATASET — FEATURE AND METADATA INSPECTION")
print("=" * 78)

# ------------------------------------------------------------
# 1. Print the 27 detector feature names
# ------------------------------------------------------------

#print("\n27 DETECTOR FEATURES")
#print("-" * 78)

#for i, feature_name in enumerate(feature_names, start=1):
#print(f"{i:>2}. {feature_name}")

#print(f"\nTotal detector features: {len(feature_names)}")

print("\n" + "=" * 78)
print("AVAILABLE VARIABLES FOR FEATURE INSPECTION")
print("=" * 78)

for variable_name in sorted(globals().keys()):
    if not variable_name.startswith("_"):
        print(variable_name)


# ------------------------------------------------------------
# 2. Print every column available in the frozen test dataframe
# ------------------------------------------------------------

print("\n" + "=" * 78)
print("ALL TEST-DATASET COLUMNS")
print("=" * 78)

for i, column_name in enumerate(test_df.columns, start=1):
    print(f"{i:>2}. {column_name}")

print(f"\nTotal dataframe columns: {len(test_df.columns)}")


# ------------------------------------------------------------
# 3. Show metadata columns that are NOT detector features
# ------------------------------------------------------------

#metadata_columns = [
    #col
    #for col in test_df.columns
    #if col not in feature_names
#]

#print("\n" + "=" * 78)
#print("NON-DETECTOR / METADATA COLUMNS")
#print("=" * 78)

#for i, column_name in enumerate(metadata_columns, start=1):
    #print(f"{i:>2}. {column_name}")

#print(f"\nTotal non-detector columns: {len(metadata_columns)}")
# ============================================================
# INSPECT CANDIDATE VARIABLES FOR S AND C
# ============================================================

print("\n" + "=" * 78)
print("CANDIDATE VARIABLES FOR SYSTEM CRITICALITY S")
print("=" * 78)

for column in [
    "service",
    "mission_phase",
    "state",
    "protocol",
]:
    print(f"\n{column.upper()}")
    print("-" * 40)
    print(
        test_df[column]
        .value_counts()
        .to_string()
    )


print("\n" + "=" * 78)
print("CANDIDATE VARIABLES FOR CONTEXT C")
print("=" * 78)

for column in [
    "tenant_id",
    "ground_station_id",
    "satellite_id",
    "episode_type",
    "benign_event_type",
]:
    print(f"\n{column.upper()}")
    print("-" * 40)
    print(
        test_df[column]
        .value_counts()
        .to_string()
    )


print("\nRESOURCE UTILIZATION")
print("-" * 40)

print(
    test_df["resource_utilization"]
    .describe()
    .to_string()
)