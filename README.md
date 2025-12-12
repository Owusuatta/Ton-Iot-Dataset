TON_IoT CTM–LSTM Baseline
Sequence-Based Intrusion Detection for Multi-Stream IoT Cybersecurity Data
Overview

This repository contains an end-to-end experimental pipeline for building baseline intrusion-detection models on the TON_IoT dataset. The workflow integrates Correlation-Tracking Method (CTM) feature selection with LSTM-based sequence modelling, enabling reproducible benchmarking for network, telemetry, and operating-system log streams.

The implementation follows a strict and reproducible research workflow, including:

Data acquisition and schema verification

Preprocessing (encoding, scaling, timestamp alignment)

CTM feature selection with export of ranked features

Sequence construction for temporal modelling

LSTM model training, validation, and evaluation

Export of metrics, artefacts, and plots

The project is designed to serve as a baseline for comparison with advanced multimodal or federated IoT intrusion detection systems.

Repository Structure
project_root/
│
├── data/
│   └── ton_iot/
│       ├── network_traffic.csv
│       ├── telemetry.csv
│       └── os_logs.csv
│
├── notebooks/
│   └── 01_ton_iot_ctm_lstm.ipynb
│
├── results/
│   ├── features/
│   │   └── selected_features.json
│   ├── metrics/
│   │   └── ton_eval.json
│   ├── plots/
│   │   ├── training_curve.png
│   │   └── confusion_matrix.png
│   └── sequences/
│       ├── X_seq.npy
│       └── y_seq.npy
│
├── models/
│   └── ton_lstm/
│       ├── checkpoint/
│       └── final_model.h5
│
└── README.md

Key Features
1. Data Loading and Audit

Each stream is loaded independently to preserve modality integrity. The pipeline performs:

Schema validation

Class imbalance inspection

Identification of categorical and numerical attributes

Detection of missing values and outliers

2. Preprocessing

The preprocessing pipeline applies:

Label encoding or one-hot encoding (depending on cardinality)

Standardization of numerical attributes

Timestamp alignment where applicable

Time-based train/validation/test splits to avoid look-ahead bias

3. CTM Feature Selection

Feature importance is computed using tree-based correlation tracking.
Outputs include:

Ranked list of features

Top-N selected features saved as selected_features.json

Feature-importance plot

Decision tree visualisations for interpretability

4. Sequence Construction

Time windows are generated for sequential modelling:

Sliding window length: 30 timesteps

Overlapping sequence construction

Shapes example:

X_seq: (39231, 30, 3)
y_seq: (39231,)
Train: (27461, 30, 3)
Validation: (5885, 30, 3)
Test: (5885, 30, 3)

5. LSTM Model Training

The baseline model consists of:

One or two LSTM layers

Dense classification layers

Dropout regularization

Checkpoint saving and early stopping

All training logs and final weights are saved.

6. Evaluation

The model is evaluated using:

Accuracy

Precision, recall, and F1

ROC-AUC

Confusion matrix

Inference latency and computational cost (optional)

Example baseline outcome:

accuracy: 1.0  
f1_score: 1.0  
roc_auc: 1.0  

7. Reproducibility

The pipeline saves:

Preprocessed datasets

Selected features

Model checkpoints

Evaluation metrics

All plots and artefacts required for replication

Requirements

Python 3.9 or above with the following libraries:

pandas

numpy

scikit-learn

tensorflow or pytorch

matplotlib

seaborn

tqdm

Install via:

pip install -r requirements.txt


(Generate a requirements file if needed.)

How to Run
Step 1: Prepare the Dataset

Download the TON_IoT CSV files and place them in:

data/ton_iot/

Step 2: Run the Notebook

Execute:

notebooks/01_ton_iot_ctm_lstm.ipynb


This automatically generates:

CTM feature list

Preprocessed sequences

LSTM model

Metrics and plots

Step 3: Review Outputs

Final artefacts will appear under:

results/
models/

Citation

If this work is used in an academic publication, please cite the TON_IoT dataset creators appropriately and reference this repository as a baseline implementation.
