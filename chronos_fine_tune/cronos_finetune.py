from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from darts import TimeSeries, concatenate
from darts.metrics import mae, mape, rmse
from darts.models import Chronos2Model
from pytorch_lightning.callbacks import Callback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from functions.analysis_functions import evaluation


class TrainLossPrinter(Callback):
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is None:
            return

        if hasattr(train_loss, "item"):
            train_loss = train_loss.item()

        print(f"Epoch {trainer.current_epoch + 1}: train_loss={train_loss:.6f}")


def main() -> None:
    repo_root = PROJECT_ROOT
    out_dir = repo_root / "logs"
    out_dir.mkdir(exist_ok=True)

    # Dataset prep (mirrors analysis_v2.py)
    df_nn = pd.read_csv(repo_root / "final_dataset_nn.csv")
    df_nn["date"] = pd.to_datetime(df_nn["date"])
    df_nn = df_nn.sort_values("date").set_index("date")

    cols_to_fill = df_nn.columns.difference(
        ["quota_per_bid_lag_-1", "quota_per_bid_growth_rate_lag_-1"]
    )
    df_nn[cols_to_fill] = df_nn[cols_to_fill].fillna(0)
    df_nn["prev_quota_premium"] = df_nn["quota_premium"].shift(1)

    y = df_nn["quota_premium_growth_rate"].astype(np.float32)
    X = df_nn.drop(columns=["quota_premium_growth_rate", "prev_quota_premium"])
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    df_nn = df_nn.reset_index(drop=True)

    H = 96
    X_train, X_test = X.iloc[:-H].astype(np.float32), X.iloc[-H:].astype(np.float32)
    y_train, y_test = y.iloc[:-H].astype(np.float32), y.iloc[-H:].astype(np.float32)
    prev_levels_test = df_nn["prev_quota_premium"].iloc[-H:]
    actual_levels_test = df_nn["quota_premium"].iloc[-H:]

    # Chronos setup
    y_full_ts = TimeSeries.from_values(y.values.astype(np.float32), columns=["y"])
    X_full_ts = TimeSeries.from_values(X.values.astype(np.float32), columns=X.columns.tolist())
    train_end = len(y_train)
    y_train_ts = y_full_ts[:train_end]
    X_train_ts = X_full_ts[:train_end]
    y_test_ts = y_full_ts[train_end : train_end + H]

    train_loss_printer = TrainLossPrinter()

    chronos_ft_model = Chronos2Model(
        input_chunk_length=12,
        output_chunk_length=1,
        hub_model_name="autogluon/chronos-2-small",
        enable_finetuning=True,
        batch_size=32,
        loss_fn=nn.L1Loss(),
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-3},
        random_state=42,
        pl_trainer_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "gradient_clip_val": 1.0,
            "enable_progress_bar": True,
            "log_every_n_steps": 25,
            "callbacks": [train_loss_printer],
        },
    )

    # Fine-tune for exactly 20 epochs.
    chronos_ft_model.fit(
        series=y_train_ts,
        past_covariates=X_train_ts,
        epochs=3,
        verbose=True,
    )

    # 1-step walk-forward inference on test horizon.
    chronos_ft_preds = []
    for i in range(H):
        end = train_end + i
        y_hist = y_full_ts[:end]
        X_hist = X_full_ts[:end]
        chronos_ft_preds.append(
            chronos_ft_model.predict(n=1, series=y_hist, past_covariates=X_hist)
        )

    chronos_ft_pred_ts = concatenate(chronos_ft_preds, axis=0)
    chronos_ft_pred = pd.Series(
        chronos_ft_pred_ts.univariate_values().astype(np.float32),
        index=y_test.index,
        name="chronos2_ft_pred",
    )

    print("Chronos-2 full fine-tuned 1-step walk-forward metrics on growth_rate:")
    print(f"MAE  : {mae(y_test_ts, chronos_ft_pred_ts):.6f}")
    print(f"RMSE : {rmse(y_test_ts, chronos_ft_pred_ts):.6f}")

    chronos_ft_eval = evaluation(
        model_name="Chronos2_FT_full",
        y_true=y_test,
        y_pred=chronos_ft_pred,
        prev_levels_test=prev_levels_test,
        actual_levels_test=actual_levels_test,
    )

    eval_out_path = out_dir / "chronos_finetune_eval.csv"
    pred_out_path = out_dir / "chronos_finetune_preds.csv"

    chronos_ft_eval.to_csv(eval_out_path, index=False)
    pd.DataFrame(
        {
            "t": y_test.index,
            "y_true_growth": y_test.values,
            "y_pred_growth": chronos_ft_pred.values,
        }
    ).to_csv(pred_out_path, index=False)

    print(f"Saved evaluation: {eval_out_path}")
    print(f"Saved predictions: {pred_out_path}")


if __name__ == "__main__":
    main()
