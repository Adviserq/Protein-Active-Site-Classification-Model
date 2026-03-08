# Active Site Protein Classification Model

This repository builds a residue-level dataset from protein structures and trains a neural network to classify whether a residue belongs to an active site.

The project combines three stages:

1. Query UniProt for human proteins with annotated active sites and available 3D structures.
2. Download the best matching PDB/mmCIF structure and extract residue-centered structural features.
3. Train a binary classifier with Keras/TensorFlow and inspect the training run with TensorBoard.

## What The Project Does

For each selected UniProt accession, the pipeline:

- fetches active-site annotations and linked PDB entries from UniProt
- selects the best PDB entry by structure resolution
- downloads the mmCIF structure file
- computes residue-level features from C-alpha neighborhoods
- labels residues as active-site or non-active-site
- appends the result to a CSV dataset for model training

The training script then scales the numeric features, applies class weighting, trains a dense neural network, and stores both the trained model and the training history.

## Repository Layout

```text
.
├── main.py                          # End-to-end dataset generation pipeline
├── models/
│   ├── keras_model.py
│   └── train_residue_model.py       # Model training + TensorBoard logging
├── scripts/
│   ├── build_residue_dataset.py     # Converts mmCIF files into residue-level rows
│   ├── extract_pdb_proteins.py      # UniProt queries and PDB selection
│   ├── feature_extractor.py         # Structural feature extraction
│   └── read_mmCIFs.py               # mmCIF download helpers
├── data/
│   ├── raw/
│   ├── preprocessed/
│   ├── temp_pdb/
│   └── tensors/
├── test/                            # Scratch scripts and debugging utilities
├── requirements.txt
└── README.md
```

## Main Components

### Dataset Pipeline

[main.py](main.py) runs the parallel data-building workflow.

It creates an `ActiveSitePipeline` that:

- fetches candidate proteins from UniProt
- processes multiple accessions in parallel with `ThreadPoolExecutor`
- downloads a structure into a temporary folder per accession
- builds or appends rows into `data/preprocessed/final_dataset.csv`
- removes temporary structure files after each accession is processed

### Feature Extraction

[scripts/feature_extractor.py](scripts/feature_extractor.py) generates residue-centered features such as:

- amino-acid identity encoded as one-hot values
- number of neighboring residues inside a radius
- mean and standard deviation of neighbor distances
- counts of charged, polar, and hydrophobic neighbors
- distance from the protein centroid

### Model Training

[models/train_residue_model.py](models/train_residue_model.py) trains a binary classifier on the residue dataset.

Current training behavior:

- loads `data/preprocessed/protein_data_cleaned.csv`
- drops metadata columns and keeps feature columns
- performs a stratified train/test split
- standardizes features with `StandardScaler`
- computes class weights for label imbalance
- trains a feed-forward Keras model with early stopping
- saves the model as `protein_model_final.h5`
- saves metric history as `training_history.csv`
- writes TensorBoard logs to `logs/fit/<timestamp>`
- tracks AUPRC with the metric name `auprc`

## Requirements

The project currently depends on:

- Biopython
- Keras
- Matplotlib
- NumPy
- pandas
- Requests
- scikit-learn
- TensorFlow
- TensorBoard

Install everything with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### 1. Build The Dataset

Run the end-to-end structure processing pipeline:

```bash
python main.py
```

Expected output:

- downloaded temporary mmCIF files under `data/temp_pdb/`
- generated residue dataset in `data/preprocessed/final_dataset.csv`

If you already have a curated dataset file, the training script expects:

```text
data/preprocessed/protein_data_cleaned.csv
```

### 2. Train The Model

Run training from the repository root:

```bash
python -m models.train_residue_model
```

If a saved model and training history already exist, the script will reuse them instead of retraining.

To force a fresh run:

```bash
FORCE_RETRAIN=1 python -m models.train_residue_model
```

Generated artifacts:

- `protein_model_final.h5`
- `training_history.csv`
- `logs/fit/<timestamp>/`

## TensorBoard

TensorBoard is already integrated into the training script through the Keras `TensorBoard` callback.

Start a fresh training run:

```bash
FORCE_RETRAIN=1 python -m models.train_residue_model
```

Then launch TensorBoard in a second terminal:

```bash
tensorboard --logdir logs/fit
```

Open this in your browser:

```text
http://localhost:6006
```

You should see metrics such as:

- `loss`
- `val_loss`
- `accuracy`
- `precision`
- `recall`
- `auprc`
- validation versions of the same metrics

## Typical Workflow

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py
FORCE_RETRAIN=1 python -m models.train_residue_model
tensorboard --logdir logs/fit
```

## Notes And Caveats

- The current training script uses `data/preprocessed/protein_data_cleaned.csv`, while the pipeline writes `data/preprocessed/final_dataset.csv`. If you want a single end-to-end path, you should either rename the generated dataset or adjust the training script to read the final dataset directly.
- Several scripts in `test/` look like debugging or exploratory files rather than automated tests.
- Existing model artifacts and CSV histories are ignored by Git through `.gitignore`.
- TensorBoard logs are also ignored by Git under `logs/`.

## Next Improvements

Reasonable next steps for the project:

1. Unify dataset file naming so generation and training use the same CSV by default.
2. Add evaluation metrics on the held-out test set after training.
3. Save the fitted scaler together with the trained model.
4. Replace exploratory `test/` scripts with reproducible unit or integration tests.

## License

No license file is currently included in the repository.
