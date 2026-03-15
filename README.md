# D200 Project

## Replicating Results

### 1) Create the Python 3.10 environment
From the project root:

```bash
python3.10 -m venv d200
source d200/bin/activate
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
1. To just reproduce the main results, run:

   ```bash
   python analysis.py
   ```

   This uses the cleaned analysis dataset `final_dataset.csv` and the committed Chronos predictions in `chronos_fine_tune/chronos_finetune_preds.csv`.
   On a MacBook Air M1, `analysis.py` takes roughly 1 hour. Or run a subset of it on a VS Code interactive interface that uses #%%, which was what I did.

2. To rebuild the dataset from the raw files first, run:

   ```bash
   python dataset_building/dataset_building.py
   python data_cleaning.py
   python analysis.py
   ```

   Notes:
   `dataset_building/dataset_building.py` reads the raw files in `dataset_building/` and writes `dataset_building/cleaned_dataset.csv`.
   It also downloads STI data from Yahoo Finance.
   `data_cleaning.py` performs the additional feature engineering, and exploratory analysis, and produces `final_dataset.csv`.

3. Chronos fine-tuning:
   Chronos fine-tuning was run on CSD3 rather than locally. To rerun it on CSD3:

   ```bash
   sbatch chronos_fine_tune/chronos.slurm
   ```

   The fine-tuning script `chronos_fine_tune/cronos_finetune.py` writes fresh outputs to:
   `chronos_fine_tune/chronos_finetune_preds.csv`
   `chronos_fine_tune/chronos_finetune_eval.csv`

   Note:
   `analysis.py` currently reads the committed file `chronos_fine_tune/chronos_finetune_preds.csv`.
   Might need to install the venv again on CSD3 and edit slurm a little for it to work
