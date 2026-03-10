# D200 Project

## Replicating Results

### 1) Create the Python 3.10 environment
From the project root:

```bash
python3.10 -m venv d200
source d200/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### (Optional) Create a Jupyter kernel

If you want to use this environment in Jupyter:

```bash
python -m ipykernel install --user --name d200 --display-name "Python (d200)"
```

### 3) Run the code
Run the project in this order:

1. Build the dataset:
   The raw CSV files and dataset-building scripts are in `dataset_building/`.
   Running `dataset_building/dataset_building.py` produces `dataset_building/cleaned_dataset.csv`.

2. Clean and engineer features:
   Run `data_cleaning.py` to do the additional feature engineering and exploratory analysis.
   This produces the analysis dataset `final_dataset_nn.csv`.

3. Train and evaluate the main models:
   Run `analysis.py`.
   On a MacBook Air M1, this takes roughly 1.5 hours.

4. Fine-tune Chronos:
   Chronos fine-tuning was run on CSD3 rather than locally.
   The relevant scripts and outputs are in `chronos_fine_tune/`.
