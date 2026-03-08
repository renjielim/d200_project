#%% ===================== 0) Imports + config =====================
import os
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# %%
REPO_ROOT = Path(__file__).parent
DATA_PATH = REPO_ROOT / "dataset_building/quota_predictions.csv"

# --- forecasting setup ---
HORIZONS = [1, 2, 3, 4]          # predict +1,+2,+3,+4 quarters ahead
STEP_MONTHS = 3                 # quarter step = 3 months
SEQ_LEN = 6                  # history length (quarters). try 12, 16, 24

# --- split setup (choose ONE style) ---
TEST_PERIODS = 24               # last N rows reserved for test (time-based)
VAL_PERIODS  = 16               # last N rows before test reserved for validation

# # If you'd rather use fractions, set TEST_PERIODS=None and use these:
# TEST_FRAC = 0.20
# VAL_FRAC  = 0.15

# --- leakage rule (your requirement) ---
ALLOWED_CURRENT_COLS = {"regis_cat_a_lag_9y"}  # can be used at time t without shifting
SHIFT_UNKNOWN_CURRENT_FEATURES = True         # shift "unknown at t" cols by 1

# --- model ---
MODEL_TYPE = "LSTM"              # "LSTM" or "GRU"
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.10

# --- training ---
SEED = 7
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 400
PATIENCE = 40                   # early stopping patience
CLIP_NORM = 1.0                 # gradient clipping

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

def set_seed(seed: int = 7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


#%% ===================== 1) Load + identify date/target =====================
df_raw = pd.read_csv(DATA_PATH)

def guess_date_col(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.lower() in {"date","period","time","month","quarter","qtr"}]
    if candidates:
        return candidates[0]
    # fallback: first column that parses well as datetime
    best = None
    best_rate = -1
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            rate = parsed.notna().mean()
            if rate > best_rate and rate > 0.8:
                best, best_rate = c, rate
        except Exception:
            pass
    if best is None:
        raise ValueError("Couldn't auto-detect a date column. Set DATE_COL manually.")
    return best

DATE_COL = guess_date_col(df_raw)
TARGET_COL = "quota_per_bid"
if TARGET_COL not in df_raw.columns:
    raise ValueError(f"TARGET_COL='{TARGET_COL}' not found. Available columns:\n{df_raw.columns.tolist()}")

df = df_raw.copy()
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)

print("Date col:", DATE_COL)
print("Rows:", len(df))
print("Columns:", df.columns.tolist())


#%% ===================== 2) Build leakage-safe features =====================
# Rule:
# - keep lag features (name contains "lag") as-is (known by construction)
# - shift any non-lag feature by 1 (unknown contemporaneously), EXCEPT ALLOWED_CURRENT_COLS
# - include past target as part of the sequence (quota_per_bid history is known up to t)

def to_numeric_safely(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s, errors="coerce")

# choose base feature columns = everything except date (target handled separately)
base_cols = [c for c in df.columns if c not in {DATE_COL}]
for c in base_cols:
    df[c] = to_numeric_safely(df[c])

# drop all-null columns (after numeric coercion)
all_null = [c for c in base_cols if df[c].isna().all()]
if all_null:
    print("Dropping all-NaN columns:", all_null)
    df = df.drop(columns=all_null)
    base_cols = [c for c in base_cols if c not in all_null]

feature_cols = [c for c in df.columns if c not in {DATE_COL, TARGET_COL}]

df_feat = df[[DATE_COL, TARGET_COL] + feature_cols].copy()

if SHIFT_UNKNOWN_CURRENT_FEATURES:
    for c in feature_cols:
        c_lower = c.lower()
        is_lag = ("lag" in c_lower)
        if (c in ALLOWED_CURRENT_COLS) or is_lag:
            continue
        # shift by 1 so that at time t we only use info from t-1
        df_feat[c] = df_feat[c].shift(1)

# also create target history feature explicitly (optional, but usually helps)
# Here we keep TARGET_COL itself as part of the sequence inputs (history up to t).
# That's fine because for origin t, y_t is known.
input_cols = [TARGET_COL] + feature_cols

# drop rows with missing values created by shifting
df_feat = df_feat.dropna().reset_index(drop=True)

print("After leakage-safe shifting + dropna:")
print("Rows:", len(df_feat))
print("Input cols:", input_cols)


#%% ===================== 3) Time-based train/val/test split (no leakage) =====================
n = len(df_feat)

if TEST_PERIODS is None:
    test_n = int(math.ceil(TEST_FRAC * n))
    val_n  = int(math.ceil(VAL_FRAC * n))
else:
    test_n = int(TEST_PERIODS)
    val_n  = int(VAL_PERIODS)

if test_n + val_n + (SEQ_LEN + max(HORIZONS)) >= n:
    raise ValueError("Not enough data for chosen TEST/VAL periods + SEQ_LEN. Reduce TEST/VAL or SEQ_LEN.")

test_start = n - test_n
val_start  = test_start - val_n
train_end  = val_start - 1

print(f"Split indices (in df_feat):")
print(f"  train: [0 .. {train_end}]  ({train_end+1} rows)")
print(f"  val:   [{val_start} .. {test_start-1}] ({val_n} rows)")
print(f"  test:  [{test_start} .. {n-1}] ({test_n} rows)")

print("Date ranges:")
print("  train:", df_feat.loc[0, DATE_COL].date(), "->", df_feat.loc[train_end, DATE_COL].date())
print("  val:  ", df_feat.loc[val_start, DATE_COL].date(), "->", df_feat.loc[test_start-1, DATE_COL].date())
print("  test: ", df_feat.loc[test_start, DATE_COL].date(), "->", df_feat.loc[n-1, DATE_COL].date())


#%% ===================== 4) Create supervised sequences (direct multi-horizon) =====================
# For each origin index i, input is rows [i-SEQ_LEN+1 .. i] and target is y at [i+h] for h in HORIZONS
# To prevent leakage:
# - training origins must satisfy i+max(HORIZONS) <= train_end
# - val origins must satisfy i+max(HORIZONS) <= test_start-1
# - test origins can start at (test_start-1) so that horizons land inside test

X_all = df_feat[input_cols].to_numpy(dtype=np.float32)
y_all = df_feat[TARGET_COL].to_numpy(dtype=np.float32)
dates_all = df_feat[DATE_COL].to_numpy()

max_h = max(HORIZONS)

def make_indices(origin_start: int, origin_end: int):
    """Origins i in [origin_start, origin_end] inclusive, with enough history and enough horizon."""
    idx = []
    for i in range(origin_start, origin_end + 1):
        if i - (SEQ_LEN - 1) < 0:
            continue
        if i + max_h >= n:
            continue
        idx.append(i)
    return np.array(idx, dtype=int)

train_origins = make_indices(origin_start=0,        origin_end=train_end - max_h)
val_origins   = make_indices(origin_start=val_start, origin_end=(test_start - 1) - max_h)
# test: allow origins from (test_start-1) onwards so that horizon targets are in test
test_origins  = make_indices(origin_start=max(test_start - 1, 0), origin_end=n - 1 - max_h)

print("Num origins:")
print("  train:", len(train_origins))
print("  val:  ", len(val_origins))
print("  test: ", len(test_origins))

def build_xy(origins: np.ndarray):
    X = np.zeros((len(origins), SEQ_LEN, len(input_cols)), dtype=np.float32)
    Y = np.zeros((len(origins), len(HORIZONS)), dtype=np.float32)
    target_dates = np.empty((len(origins), len(HORIZONS)), dtype="datetime64[ns]")
    origin_dates = np.empty((len(origins),), dtype="datetime64[ns]")

    for j, i in enumerate(origins):
        X[j] = X_all[i-SEQ_LEN+1:i+1]
        for k, h in enumerate(HORIZONS):
            Y[j, k] = y_all[i + h]
            target_dates[j, k] = dates_all[i + h]
        origin_dates[j] = dates_all[i]
    return X, Y, origin_dates, target_dates

X_tr, Y_tr, d0_tr, dt_tr = build_xy(train_origins)
X_va, Y_va, d0_va, dt_va = build_xy(val_origins)
X_te, Y_te, d0_te, dt_te = build_xy(test_origins)


#%% ===================== 5) Scale using TRAIN only =====================
x_scaler = StandardScaler()
y_scaler = StandardScaler()

# Fit x_scaler on all timesteps in train
x_scaler.fit(X_tr.reshape(-1, X_tr.shape[-1]))
X_tr_s = x_scaler.transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
X_va_s = x_scaler.transform(X_va.reshape(-1, X_va.shape[-1])).reshape(X_va.shape)
X_te_s = x_scaler.transform(X_te.reshape(-1, X_te.shape[-1])).reshape(X_te.shape)

# Fit y_scaler on all horizon values in train (flatten)
y_scaler.fit(Y_tr.reshape(-1, 1))
Y_tr_s = y_scaler.transform(Y_tr.reshape(-1, 1)).reshape(Y_tr.shape)
Y_va_s = y_scaler.transform(Y_va.reshape(-1, 1)).reshape(Y_va.shape)
Y_te_s = y_scaler.transform(Y_te.reshape(-1, 1)).reshape(Y_te.shape)


#%% ===================== 6) Torch datasets/loaders =====================
class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_loader = DataLoader(SeqDataset(X_tr_s, Y_tr_s), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(SeqDataset(X_va_s, Y_va_s), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(SeqDataset(X_te_s, Y_te_s), batch_size=BATCH_SIZE, shuffle=False)


#%% ===================== 7) Model (LSTM/GRU) =====================
class RNNForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, out_size, rnn_type="GRU"):
        super().__init__()
        rnn_type = rnn_type.upper()
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
        else:
            raise ValueError("rnn_type must be 'LSTM' or 'GRU'")

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, x):
        out, _ = self.rnn(x)          # (B, T, H)
        last = out[:, -1, :]          # (B, H)
        return self.head(last)        # (B, out_size)

model = RNNForecaster(
    input_size=len(input_cols),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    out_size=len(HORIZONS),
    rnn_type=MODEL_TYPE,
).to(DEVICE)

loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print(model)


#%% ===================== 8) Train with early stopping =====================
def run_epoch(loader, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total = 0.0
    nobs = 0
    for Xb, Yb in loader:
        Xb = Xb.to(DEVICE)
        Yb = Yb.to(DEVICE)

        if train:
            opt.zero_grad(set_to_none=True)

        pred = model(Xb)
        loss = loss_fn(pred, Yb)

        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()

        bs = Xb.size(0)
        total += loss.item() * bs
        nobs += bs

    return total / max(nobs, 1)

best_val = float("inf")
best_state = None
pat = 0

train_losses, val_losses = [], []

for epoch in range(1, EPOCHS + 1):
    tr = run_epoch(train_loader, train=True)
    va = run_epoch(val_loader, train=False)

    train_losses.append(tr)
    val_losses.append(va)

    if va < best_val - 1e-6:
        best_val = va
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        pat = 0
    else:
        pat += 1

    if epoch % 25 == 0 or epoch == 1:
        print(f"epoch {epoch:4d} | train {tr:.5f} | val {va:.5f} | best {best_val:.5f} | pat {pat}/{PATIENCE}")

    if pat >= PATIENCE:
        print("Early stopping.")
        break

if best_state is not None:
    model.load_state_dict(best_state)
    model.to(DEVICE)


#%% ===================== 9) Plot train/val loss =====================
plt.figure()
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("epoch")
plt.ylabel("MSE (scaled)")
plt.title(f"{MODEL_TYPE} training curve")
plt.legend()
plt.show()


#%% ===================== 10) Predict + inverse transform =====================
def predict_array(X_scaled: np.ndarray) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_scaled), BATCH_SIZE):
            xb = torch.tensor(X_scaled[i:i+BATCH_SIZE], dtype=torch.float32, device=DEVICE)
            pb = model(xb).detach().cpu().numpy()
            preds.append(pb)
    return np.vstack(preds)

Yhat_te_s = predict_array(X_te_s)

# inverse transform (back to original quota_per_bid units)
Yhat_te = y_scaler.inverse_transform(Yhat_te_s.reshape(-1, 1)).reshape(Yhat_te_s.shape)
Ytrue_te = Y_te  # already original units from build_xy


#%% ===================== 11) Metrics per horizon =====================
def mape(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

rows = []
for k, h in enumerate(HORIZONS):
    yt = Ytrue_te[:, k]
    yp = Yhat_te[:, k]
    mae = mean_absolute_error(yt, yp)
    rmse = math.sqrt(mean_squared_error(yt, yp))
    r2 = r2_score(yt, yp)
    mp = mape(yt, yp)
    rows.append((h, mae, rmse, mp, r2))

metrics = pd.DataFrame(rows, columns=["horizon(+quarters)", "MAE", "RMSE", "MAPE_%", "R2"])
print(metrics.to_string(index=False))


#%% ===================== 12) Plots: actual vs predicted (per horizon) =====================
# Plot against the target dates for each horizon
for k, h in enumerate(HORIZONS):
    dd = pd.to_datetime(dt_te[:, k])
    order = np.argsort(dd)

    plt.figure()
    plt.plot(dd[order], Ytrue_te[:, k][order], label="actual")
    plt.plot(dd[order], Yhat_te[:, k][order], label="pred")
    plt.xlabel("date (target)")
    plt.ylabel("quota_per_bid")
    plt.title(f"{MODEL_TYPE}: forecast horizon +{h} quarter(s)")
    plt.legend()
    plt.show()


#%% ===================== 13) Baseline (optional): last value carried forward =====================
# For origin i, baseline predicts y_{i+h} ≈ y_i (random walk)
# This is a good sanity check; your RNN should beat it.

Ybase = np.zeros_like(Ytrue_te)
for j, origin_date in enumerate(d0_te):
    # origin index in df_feat: find matching row
    # (fast way: use mapping once)
    pass
# %%
