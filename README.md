# GSaaS AI-Driven Cybersecurity Framework — Simulation Code

## Paper
**"An AI-Driven Cybersecurity Framework for GSaaS: A Case Study of Africa"**  
Paul A. Oche, Ifunanya Mafiana, Brown C. Ejike, Haruna Ocholi, Robinson T. Sibe  
National Space Research and Development Agency (NASRDA), Abuja, Nigeria

---

## Files in This Repository

| File | Description |
|---|---|
| `gsaas_framework_simulation.py` | Main simulation — full framework evaluation |
| `requirements.txt` | Python package dependencies |
| `README.md` | This file |

---

## Requirements

### Python Version
Python 3.8 or higher recommended.

### Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
```

---

## Datasets

The datasets are **not included** in this repository as they are
publicly available from their original sources.

### NSL-KDD Dataset
Download both files and place in the same folder as the script:

**Training set:**
```
https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt
```
Save as: `KDDTrain+.txt`

**Test set:**
```
https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt
```
Save as: `KDDTest+.txt`

**Citation:**
> M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani,
> "A detailed analysis of the KDD CUP 99 data set,"
> in Proc. 2nd IEEE Symp. Comput. Intell. Security Def. Appl., 2009.
> doi: 10.1109/CISDA.2009.5356528

---

### UNSW-NB15 Dataset
Download both files and place in the same folder as the script:

**Training set:**
```
https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv
```
Save as: `UNSW_NB15_training-set.csv`

**Test set:**
```
https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW_NB15_testing-set.csv
```
Save as: `UNSW_NB15_testing-set.csv`

**Citation:**
> N. Moustafa and J. Slay,
> "UNSW-NB15: A comprehensive data set for network intrusion
> detection systems,"
> in Proc. Military Commun. Inf. Syst. Conf. (MilCIS), 2015.
> doi: 10.1109/MilCIS.2015.7348942

---

## Running the Simulation

### Step 1 — Confirm your folder structure
```
your_folder/
├── gsaas_framework_simulation.py
├── requirements.txt
├── KDDTrain+.txt
├── KDDTest+.txt
├── UNSW_NB15_training-set.csv
└── UNSW_NB15_testing-set.csv
```

### Step 2 — Run
```bash
python gsaas_framework_simulation.py
```

### Step 3 — Check outputs
The script produces five CSV files:

| Output File | Contents |
|---|---|
| `results_kdd_baseline.csv` | Baseline comparison on NSL-KDD |
| `results_kdd_calibration.csv` | τ calibration study on NSL-KDD |
| `results_unsw_baseline.csv` | Baseline comparison on UNSW-NB15 |
| `results_unsw_calibration.csv` | τ calibration study on UNSW-NB15 |
| `results_resilience.csv` | Resilience index by attack type |

---

## Expected Runtime

| Machine | Approximate Time |
|---|---|
| Standard laptop (CPU only) | 20–40 minutes |
| Workstation with GPU | 5–10 minutes |

TensorFlow will automatically use a GPU if available.

---

## Reproducibility

Random seeds are fixed:
```python
np.random.seed(42)
tf.random.set_seed(42)
```

Results should be reproducible to within ±0.5% variation
due to hardware-level floating point differences.

---

## Framework Components Implemented

| Component | Paper Section | Implementation |
|---|---|---|
| System model | Section V.A | Feature vector x_i(t) ∈ ℝ^d |
| Anomaly detection | Section V.B | Autoencoder f_θ, Equations (4)–(7) |
| Threat classification | Section V.C | MLP softmax g_φ, Equation (8) |
| Risk scoring | Section V.D | R = αA_i(t) + βS + γC, Equation (9) |
| Edge-cloud decision | Section V.E | D based on A_i(t) vs τ_e, Equation (10) |
| Multi-tenant isolation | Section V.F | Corr(A_a, A_b) > δ, Equation (13) |
| Resilience metric | Section V.G | ℛ = U_attack/U_normal, Equation (14) |

---

## Contact

Paul A. Oche  
National Space Research and Development Agency (NASRDA)  
Abuja, Nigeria  
oche.paul@nasrda.gov.ng
