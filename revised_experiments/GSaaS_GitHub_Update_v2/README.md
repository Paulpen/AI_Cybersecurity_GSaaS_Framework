# GSaaS AI-Driven Cybersecurity Framework — Reproducibility Release v2

This archive accompanies the revised manuscript **“An AI-Driven Cybersecurity Framework for GSaaS: Threat Detection in Emerging Space Ecosystems.”** It adds the GSaaS-specific synthetic dataset, trained models, revised experimental code, event-level outputs, and AC-FAB/benchmark artefacts used in the revised evaluation.

## Important scope statement

The GSaaS dataset is **synthetic**. It is a controlled, domain-grounded experimental dataset and must not be interpreted as operational traffic collected from a real GSaaS provider. The software-policy experiments do not constitute physical deployment validation.

## Dataset

The generator uses random seed **42** and creates:

- 1,000 operational episodes
- 100 events per episode
- 100,000 events total
- 3 tenants, 3 ground stations, 3 satellites
- normal benign, benign-unusual, and threat episodes
- 6 threat classes

Frozen episode-level partitions:

- `data/gsaas_train_seed42.csv` — 60,000 events / 600 episodes
- `data/gsaas_validation_seed42.csv` — 20,000 events / 200 episodes
- `data/gsaas_test_seed42.csv` — 20,000 events / 200 episodes
- `data/gsaas_events_seed42.csv` — complete 100,000-event dataset

The partitions are episode-disjoint. Preprocessing is fitted on the training partition only.

## Trained models

The `models/` directory contains the saved Keras models used during the revised experiments, including the autoencoder and MLP variants.

## Key frozen outputs

- `results/gsaas_mlp27ae_test_predictions.csv` — event-level actual/predicted classes used by downstream false-positive/FAB analysis.
- `results/gsaas_test_risk_scores.csv` — event-level composite risk results used by routing, orchestration, tenant-scope, and post-orchestration analyses.

## Suggested execution order

1. `code/gsaas_event_generator.py`
2. `code/gsaas_data_audit.py`
3. `code/gsaas_preprocessing.py` (imported by model experiments)
4. `code/gsaas_autoencoder_experiment.py`
5. `code/gsaas_mlp_experiment.py`
6. `code/gsaas_risk_score_experiment.py`
7. `code/gsaas_edge_cloud_experiment.py`
8. `code/gsaas_response_orchestration_experiment.py`
9. `code/gsaas_tenant_isolation_experiment.py`
10. `code/gsaas_post_orchestration_fab.py`
11. `code/gsaas_fab_experiment.py` for broader AC-FAB sensitivity scenarios

Additional diagnostic/sensitivity scripts are also included.

## Benchmark artefacts

`benchmark_results/` contains the archived AC-FAB threshold/burden sensitivity tables and figures. `code/benchmark_framework_simulation.py` contains the earlier benchmark pipeline for NSL-KDD and UNSW-NB15. These benchmark datasets are complementary detector-level references; they are not treated as substitutes for GSaaS-specific operational variables.

## Reproducibility note

The frozen CSVs and `.keras` files are included so that the exact experimental artefacts can be inspected without retraining. Neural-network retraining can exhibit small numerical variation across TensorFlow versions, hardware, and execution environments even when seeds are fixed. Where exact comparison with the manuscript is required, use the frozen models and event-level outputs supplied in this archive.

## Archive DOI

The revised archive is intended to be deposited as a new version of the existing Zenodo record associated with DOI **10.5281/zenodo.21883421**.
