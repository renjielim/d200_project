#%%
import os
import warnings
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import statsmodels.api as sm

from darts.models import (
    AutoARIMA,
    NaiveSeasonal,
    RandomForestModel,
    SKLearnModel,
    TSMixerModel,
    XGBModel,
)
from scipy.stats import wilcoxon
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.svm import SVR

from functions.analysis_functions import (
    darts_pipeline,
    diebold_mariano_test,
    error_plot,
    evaluation,
    mean_expected_profit_sigmoid,
    plot_profit_curves_sigmoid,
    plot_test,
    profit_cv_score_sigmoid,
)

# %% import
REPO_ROOT = Path(__file__).parent
df_nn = pd.read_csv(REPO_ROOT / "final_dataset_nn.csv")
df_nn["date"] = pd.to_datetime(df_nn["date"])
df_nn = df_nn.sort_values("date").set_index("date")
cols_to_fill = df_nn.columns.difference(['quota_per_bid_lag_-1', 'quota_per_bid_growth_rate_lag_-1'])
df_nn[cols_to_fill] = df_nn[cols_to_fill].fillna(0)
df_nn["prev_quota_premium"] = df_nn["quota_premium"].shift(1)

y = df_nn["quota_premium_growth_rate"].astype(np.float32)
X = df_nn.drop(columns=['quota_premium_growth_rate','prev_quota_premium'])
date_real = X.index.copy()

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
df_nn = df_nn.reset_index(drop=True)
X.index.name = "t"
y.index.name = "t"

H = 96
X_train, X_test = X.iloc[:-H].astype(np.float32), X.iloc[-H:].astype(np.float32)
y_train, y_test = y.iloc[:-H].astype(np.float32), y.iloc[-H:].astype(np.float32)
prev_levels_train = df_nn['prev_quota_premium'].iloc[:-H]
prev_levels_test = df_nn['prev_quota_premium'].iloc[-H:]
actual_levels_train = df_nn['quota_premium'].iloc[:-H]
actual_levels_test = df_nn['quota_premium'].iloc[-H:]

assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)
assert X_train.index.equals(y_train.index)
assert X_test.index.equals(y_test.index)
assert X_train.isna().sum().sum() == 0
assert y_train.isna().sum() == 0
assert y_test.isna().sum() == 0

#%%
# AutoArima

arima_base_params = dict()
arima_pred, arima_model = darts_pipeline(
    model_cls=AutoARIMA,
    base_params=arima_base_params,
    X_train=None, y_train=y_train,
    X_test=None, y_test=y_test,
    param_grid=None,
    scale_features=False,
    scale_target=False,
    scoring_metric=mean_absolute_error)

arima_eval = evaluation(
    model_name="AutoARIMA",
    y_true=y_test.loc[arima_pred.index],
    y_pred=arima_pred,
    prev_levels_test=prev_levels_test.loc[arima_pred.index],
    actual_levels_test=actual_levels_test.loc[arima_pred.index],
)

if hasattr(arima_model, "model") and hasattr(arima_model.model, "model_"):
    spec = arima_model.model.model_
    p, q, P, Q, m, d, D = spec["arma"]   # statsforecast format
    print(f"Selected: ARIMA({p},{d},{q}) x ({P},{D},{Q})[{m}]")
    print("AIC:", spec.get("aic"), "AICc:", spec.get("aicc"), "BIC:", spec.get("bic"))
    print("Coefficients:", spec.get("coef"))


#%%
# Random Walk (Darts): NaiveSeasonal(K=1)
rw_base_params = {"K": 1}
rw_pred, rw_model = darts_pipeline(
    model_cls=NaiveSeasonal,
    base_params=rw_base_params,
    X_train=None, y_train=actual_levels_train,
    X_test=None, y_test=actual_levels_test,
    param_grid=None,
    scale_features=False,
    scale_target=False,
    scoring_metric=mean_absolute_error,
)

rw_rmse = root_mean_squared_error(actual_levels_test.loc[rw_pred.index], rw_pred)
rw_mae = mean_absolute_error(actual_levels_test.loc[rw_pred.index], rw_pred)

#MAE: 3590.2917 RMSE: 4694.0878 for last 48
#MAE: 3823.1146 RMSE: 5209.9593 for all 96
#%% ===================== ENET =======================
enet_base_params = dict(
    lags=12,                 
    lags_past_covariates=12,   
    output_chunk_length=1,
    model=ElasticNet(
        alpha=0.01,          
        l1_ratio=0.5,        
        fit_intercept=True,
        max_iter=10000,
        random_state=42,
    ),
)

enet_param_grid = {
    "model__alpha": [0.1, 1, 2, 5, 10, 20, 30],
    "model__l1_ratio": [0, 0.2,0.4,0.6,0.8,1.0],
}

enet_pred, enet_model = darts_pipeline(
    model_cls=SKLearnModel,
    base_params=enet_base_params,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    param_grid=enet_param_grid,
    scale_features=True,     
    scale_target=False,
    n_splits= 3,            
    scoring_metric=mean_absolute_error,
    validation_refit_each_step=False
)

#%% xgb

xgb_base_params = dict(
    lags=12,                 
    lags_past_covariates=12,  
    output_chunk_length=1,
    n_estimators=800,
    max_depth=3,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.1,
    random_state=42,
)

xgb_param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 200, 300],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0.1, 0.3, 0.5],
}

xgb_pred, xgb_model = darts_pipeline(
    model_cls=XGBModel,
    base_params=xgb_base_params,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    param_grid=xgb_param_grid,
    scale_features=False,    # trees don't need scaling
    scale_target=False,
    n_splits=3,
    scoring_metric=mean_absolute_error,
    validation_refit_each_step=False
)


#%% random forest
rf_base_params = dict(
    lags=12,                   
    lags_past_covariates=12,  
    output_chunk_length=1,
    n_estimators=800,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
)

rf_param_grid = {
    "n_estimators": [800],
    "max_depth": [3, 5, 8],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 5],
    "max_features": ["sqrt"],
}

rf_pred, rf_model = darts_pipeline(
    model_cls=RandomForestModel,
    base_params=rf_base_params,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    param_grid=rf_param_grid,
    scale_features=False,     # trees don't need scaling
    scale_target=False,
    n_splits=3,
    scoring_metric=mean_absolute_error,
    validation_refit_each_step=False
)

#%% svr
svr_base_params = dict(
    lags=12,                   # y_{t-1}..y_{t-12}
    lags_past_covariates=12,   # X_{t-1}..X_{t-12}
    output_chunk_length=1,
    model=SVR(
        kernel="rbf",
        C=10.0,
        epsilon=0.1,
        gamma="scale",
    ),
)

svr_param_grid = {
    "model__kernel": ["rbf", "linear"],
    "model__C": [0.3, 1.0, 3.0, 10.0, 30.0],
    "model__epsilon": [0.1, 0.5, 1, 2],
    "model__gamma": ["scale"],
}

svr_pred, svr_model = darts_pipeline(
    model_cls=SKLearnModel,
    base_params=svr_base_params,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    param_grid=svr_param_grid,
    scale_features=True,      # important for SVR
    scale_target=False,
    n_splits=3,
    scoring_metric=mean_absolute_error,
    validation_refit_each_step=False
)


#%% XGB with custom sigmoid profit loss
xgb_base_params_sigmoid = dict(
    lags=12,
    lags_past_covariates=12,
    output_chunk_length=1,
    n_estimators=800,
    max_depth=3,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.1,
    random_state=42,
    objective="reg:absoluteerror",
)

xgb_pred_sigmoid, xgb_model_sigmoid = darts_pipeline(
    model_cls=XGBModel,
    base_params=xgb_base_params_sigmoid,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    param_grid=xgb_param_grid,
    scale_features=False,
    scale_target=False,
    n_splits=3,
    scoring_metric=lambda yt, yp: profit_cv_score_sigmoid(
        y_true_growth=yt,
        y_pred_growth=yp,
        prev_levels_lookup=prev_levels_train, 
        steepness=0.0005,      
        max_demand=1,        
        base_margin=2000
    ),
    validation_refit_each_step=False,
)


#%% EVALUATION AND PLOTS
enet_eval = evaluation(
    model_name="Elastic_Net",
    y_true=y_test,
    y_pred=enet_pred,
    prev_levels_test=prev_levels_test,
    actual_levels_test=actual_levels_test,
)

xgb_eval = evaluation(
    model_name="XGB",
    y_true=y_test,
    y_pred=xgb_pred,
    prev_levels_test=prev_levels_test,
    actual_levels_test=actual_levels_test,
)

rf_eval = evaluation(
    model_name="RF",
    y_true=y_test,
    y_pred=rf_pred,
    prev_levels_test=prev_levels_test,
    actual_levels_test=actual_levels_test,
)

svr_eval = evaluation(
    model_name="SVR",
    y_true=y_test,
    y_pred=svr_pred,
    prev_levels_test=prev_levels_test,
    actual_levels_test=actual_levels_test,
)


xgb_eval_sigmoid = evaluation(
    model_name="XGB Custom Loss",
    y_true=y_test,
    y_pred=xgb_pred_sigmoid,
    prev_levels_test=prev_levels_test,
    actual_levels_test=actual_levels_test,
)


eval_df = pd.concat(
    [arima_eval, enet_eval, xgb_eval, rf_eval, svr_eval, xgb_eval_sigmoid, chronos_eval],
    ignore_index=True,
)
#%%
xgb_pred_levels = prev_levels_test * (1 + xgb_pred / 100)
xgb_pred_sigmoid_levels = prev_levels_test * (1 + xgb_pred_sigmoid / 100)
arima_pred_levels = prev_levels_test * (1 + arima_pred / 100)

xgb_plot = plot_test(
    y_true=actual_levels_test,
    y_pred=xgb_pred_levels,
    y_pred_2=xgb_pred_sigmoid_levels,
    title="XGB: Actual vs Predicted Levels",
    xlabel="Time",
    ylabel="Quota Premium",
    y_pred_label="XGB",
    y_pred_2_label="XGB Custom Loss",
    date_real=date_real
)


#%%
diebold_mariano_test(actual_levels_test, xgb_pred_levels, rw_pred, loss='abs')
diebold_mariano_test(actual_levels_test, xgb_pred_levels, arima_pred_levels, loss='abs')

diebold_mariano_test(actual_levels_test, xgb_pred_sigmoid_levels, rw_pred, loss='abs')
diebold_mariano_test(actual_levels_test, xgb_pred_sigmoid_levels, arima_pred_levels, loss='abs')
#%%
error_plot(
    actual_levels_test=actual_levels_test,
    baseline_pred=arima_pred_levels,
    challenger_pred=xgb_pred_sigmoid_levels,
    date_real=date_real
)

# %%
plot_profit_curves_sigmoid(
    actual_levels=actual_levels_test,
    prev_levels=prev_levels_test,
    models_dict={
        "XGB Custom Loss": {"preds": xgb_pred_sigmoid_levels, "color": "tab:blue"},
        "XGB MAE Loss": {"preds": xgb_pred_levels, "color": "tab:orange"},
        "Random Walk": {"preds": rw_pred, "color": "tab:green"},
        "ARIMA": {"preds": arima_pred_levels, "color": "tab:red"},
    },
    base_margin=2000)
# %%
profit_xgb = mean_expected_profit_sigmoid(
    actual=actual_levels_test,
    pred=xgb_pred_levels,
    prev_levels=prev_levels_test,
    steepness=0.0005,
    base_margin=2000,
)
profit_xgb_sigmoid = mean_expected_profit_sigmoid(
    actual=actual_levels_test,
    pred=xgb_pred_sigmoid_levels,
    prev_levels=prev_levels_test,
    steepness=0.0005,
    base_margin=2000,
    full_list=True
)
profit_rw = mean_expected_profit_sigmoid(
    actual=actual_levels_test,
    pred=rw_pred,
    prev_levels=prev_levels_test,
    steepness=0.0005,
    base_margin=2000,
    full_list=True
)
profit_arima = mean_expected_profit_sigmoid(
    actual=actual_levels_test,
    pred=arima_pred_levels,
    prev_levels=prev_levels_test,
    steepness=0.0005,
    base_margin=2000,
    full_list=True
)

# %%
# 1. Calculate the difference in profit for each auction
d = np.array(profit_xgb_sigmoid) - np.array(profit_rw)
constant = np.ones(len(d))
model = sm.OLS(endog=d, exog=constant)
hac_results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})

# 4. Extract the Diebold-Mariano Statistic and P-Value
dm_stat = hac_results.tvalues[0]
p_value = hac_results.pvalues[0]

print(f"Mean Profit Difference (XGB - RW): ${d.mean():.2f}")
print(f"Diebold-Mariano Statistic: {dm_stat:.4f}")
print(f"P-Value: {p_value:.4f}")

# %% wilcoxon signed-rank test (non-parametric alternative to DM test)
# Calculate the profit differences
d = np.array(profit_xgb_sigmoid) - np.array(profit_rw)
# Perform the Wilcoxon signed-rank test
statistic, p_value = wilcoxon(d, alternative='two-sided')
print(f"Wilcoxon signed-rank test statistic: {statistic:.4f}")
print(f"P-Value: {p_value:.4f}")

# %%
# xgb_model is the fitted Darts XGBModel returned by darts_pipeline
importances = xgb_model_sigmoid.model.feature_importances_          # from XGBoost
feat_names = xgb_model_sigmoid.lagged_feature_names                # from Darts (public attr)

fi = (
    pd.DataFrame({"feature": feat_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

print(fi.head(30))


#%%
# ===================== Chronos-2 Fine-Tuning (3 epochs) + Walk-Forward Test =====================
from darts import TimeSeries
from darts.metrics import mae, mape, rmse
from darts.models import Chronos2Model
import torch.nn as nn

# Build full series once, then slice by train/test boundary.
y_full_ts = TimeSeries.from_values(y.values.astype(np.float32), columns=["y"])
X_full_ts = TimeSeries.from_values(X.values.astype(np.float32), columns=X.columns.tolist())

train_end = len(y_train)
y_train_ts = y_full_ts[:train_end]
X_train_ts = X_full_ts[:train_end]
y_test_ts = y_full_ts[train_end:train_end + H]

# Fine-tunable Chronos-2 model.
chronos_model = Chronos2Model(
    input_chunk_length=32,
    output_chunk_length=1,
    hub_model_name="autogluon/chronos-2-small",
    enable_finetuning=True,
    batch_size=64,
    loss_fn=nn.L1Loss(),
    optimizer_kwargs={"lr": 1e-4},
    random_state=42,
    pl_trainer_kwargs={
        "accelerator": "cpu",
        "devices": 1,
        "gradient_clip_val": 1.0,
        "enable_progress_bar": True,
        "log_every_n_steps": 25,
    },
)

# Fine-tune on training split for exactly 3 epochs.
chronos_model.fit(
    series=y_train_ts,
    past_covariates=X_train_ts,
    epochs=3,
    verbose=True,
)

# 1-step-ahead expanding-window walk-forward over the full test horizon.
chronos_preds = []
for i in range(H):
    end = train_end + i
    y_hist = y_full_ts[:end]  # includes info up to t-1
    X_hist = X_full_ts[:end]  # past covariates only up to t-1
    chronos_preds.append(
        chronos_model.predict(n=1, series=y_hist, past_covariates=X_hist)
    )

chronos_pred_ts = TimeSeries.concatenate(chronos_preds, axis=0)
chronos_pred = pd.Series(
    chronos_pred_ts.univariate_values().astype(np.float32),
    index=y_test.index,
    name="chronos2_pred",
)

print("Chronos-2 fine-tuned (3 epochs) 1-step walk-forward metrics on growth_rate:")
print(f"MAE  : {mae(y_test_ts, chronos_pred_ts):.6f}")
print(f"RMSE : {rmse(y_test_ts, chronos_pred_ts):.6f}")
print(f"MAPE : {mape(y_test_ts, chronos_pred_ts):.3f}%")

chronos_eval = evaluation(
    model_name="Chronos2_FT_3ep",
    y_true=y_test,
    y_pred=chronos_pred,
    prev_levels_test=prev_levels_test,
    actual_levels_test=actual_levels_test,
)








#%% TCN
tcn_base_params = dict(
    input_chunk_length=12,
    output_chunk_length=1,
    kernel_size=3,
    num_filters=8,
    dilation_base=2,
    dropout=0.1,
    n_epochs=80,
    batch_size=32,
    optimizer_kwargs={"lr": 1e-3},
    random_state=42,
    force_reset=True,
    save_checkpoints=False,
)

tcn_param_grid = {
    "kernel_size": [3, 5],
    "num_filters": [8, 16, 32],
    "num_layers": [2, 3],
    "dropout": [0.0, 0.2],
    "optimizer_kwargs": [{"lr": 1e-3}, {"lr": 3e-4}],
    "n_epochs": [80, 120],
}

tcn_pred, tcn_model = darts_pipeline(
    model_cls=TCNModel,
    base_params=tcn_base_params,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    param_grid=tcn_param_grid,
    recalibrate_every=None,
    n_splits=3,
    scale_features=True,
    scale_target=True,
    scoring_metric=mean_absolute_error,
    validation_refit_each_step=False,
)
