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
All the csv used, the code to clean and build the dataset, and the final dataset are all in the folder dataset_building