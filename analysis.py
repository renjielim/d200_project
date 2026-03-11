#%%
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

from darts import TimeSeries
from darts.explainability.shap_explainer import ShapExplainer
from darts.models import (
    AutoARIMA,
    NaiveSeasonal,
    RandomForestModel,
    SKLearnModel,
    XGBModel,
)
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.svm import SVR

from functions.analysis_functions import (
    darts_pipeline,
    diebold_mariano_test,
    evaluation,
    mean_expected_profit_sigmoid,
    plot_profit_curves_sigmoid,
    plot_test,
    profit_cv_score_sigmoid,
)

# %% import
REPO_ROOT = Path(__file__).parent
df_nn = pd.read_csv(REPO_ROOT / "final_dataset.csv")
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

#%%Chronos predictions (ran in CSD3)
chronos = pd.read_csv(REPO_ROOT / "chronos_fine_tune" / "chronos_finetune_preds.csv", index_col=0)
chronos_pred = chronos["y_pred_growth"].astype(np.float32)

chronos_eval = evaluation(
    model_name="Chronos",
    y_true=y_test,
    y_pred=chronos_pred,
    prev_levels_test=prev_levels_test,
    actual_levels_test=actual_levels_test,
)
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

print(f"Random Walk MAE: {rw_mae:.4f}, RMSE: {rw_rmse:.4f}")

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
    objective="reg:absoluteerror"
)

xgb_param_grid = {
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.05],
    "n_estimators": [200, 500],
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
    scale_features=False,   
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
    criterion="absolute_error"
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
    scale_features=False,    
    scale_target=False,
    n_splits=3,
    scoring_metric=mean_absolute_error,
    validation_refit_each_step=False
)

#%% svr
svr_base_params = dict(
    lags=12,                   
    lags_past_covariates=12,   
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
    scale_features=True,     
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

print(eval_df)
#%%
xgb_pred_levels = prev_levels_test * (1 + xgb_pred / 100)
xgb_pred_sigmoid_levels = prev_levels_test * (1 + xgb_pred_sigmoid / 100)
arima_pred_levels = prev_levels_test * (1 + arima_pred / 100)

xgb_plot = plot_test(
    y_true=actual_levels_test,
    y_pred=xgb_pred_levels,
    y_pred_2=rw_pred,
    title="Actual vs Predicted Levels for XGB and RW",
    xlabel="Time",
    ylabel="Quota Premium",
    y_pred_label="XGB",
    y_pred_2_label="Random Walk",
    date_real=date_real
)


#%%
print(f"DM Test XGB vs RW: p-value = {diebold_mariano_test(actual_levels_test, xgb_pred_levels, rw_pred, loss='abs')[1]:.4f}")
print(f"DM Test XGB vs ARIMA: p-value = {diebold_mariano_test(actual_levels_test, xgb_pred_levels, arima_pred_levels, loss='abs')[1]:.4f}")
print(f"DM Test XGB Custom Loss vs RW: p-value = {diebold_mariano_test(actual_levels_test, xgb_pred_sigmoid_levels, rw_pred, loss='abs')[1]:.4f}")
print(f"DM Test XGB Custom Loss vs ARIMA: p-value = {diebold_mariano_test(actual_levels_test, xgb_pred_sigmoid_levels, arima_pred_levels, loss='abs')[1]:.4f}")

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

# %% test if profit is significantly higher
profit_xgb = mean_expected_profit_sigmoid(
    actual=actual_levels_test,
    pred=xgb_pred_levels,
    prev_levels=prev_levels_test,
    steepness=0.0005,
    base_margin=2000,
    full_list=True
)

mean_profit_xgb = np.mean(profit_xgb)

profit_xgb_sigmoid = mean_expected_profit_sigmoid(
    actual=actual_levels_test,
    pred=xgb_pred_sigmoid_levels,
    prev_levels=prev_levels_test,
    steepness=0.0005,
    base_margin=2000,
    full_list=True
)
mean_profit_xgb_sigmoid = np.mean(profit_xgb_sigmoid)

profit_rw = mean_expected_profit_sigmoid(
    actual=actual_levels_test,
    pred=rw_pred,
    prev_levels=prev_levels_test,
    steepness=0.0005,
    base_margin=2000,
    full_list=True
)
mean_profit_rw = np.mean(profit_rw)

profit_arima = mean_expected_profit_sigmoid(
    actual=actual_levels_test,
    pred=arima_pred_levels,
    prev_levels=prev_levels_test,
    steepness=0.0005,
    base_margin=2000,
    full_list=True
)
mean_profit_arima = np.mean(profit_arima)

# %%
# calculate the difference in profit 
d = np.array(profit_xgb_sigmoid) - np.array(profit_rw)
constant = np.ones(len(d))
model = sm.OLS(endog=d, exog=constant)
hac_results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})
dm_stat = hac_results.tvalues[0]
p_value = hac_results.pvalues[0]

print(f"Mean Profit Difference (XGB_sigmoid - RW): ${d.mean():.2f}")
print(f"Percentage Profit Improvement: {(mean_profit_xgb_sigmoid - mean_profit_rw) / abs(mean_profit_rw) * 100:.2f}%")
print(f"Diebold-Mariano Statistic: {dm_stat:.4f}")
print(f"P-Value: {p_value:.4f}")

d = np.array(profit_xgb) - np.array(profit_rw)
constant = np.ones(len(d))
model = sm.OLS(endog=d, exog=constant)
hac_results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})
dm_stat = hac_results.tvalues[0]
p_value = hac_results.pvalues[0]

print(f"Mean Profit Difference (XGB - RW): ${d.mean():.2f}")
print(f"Percentage Profit Improvement: {(mean_profit_xgb - mean_profit_rw) / abs(mean_profit_rw) * 100:.2f}%")
print(f"Diebold-Mariano Statistic: {dm_stat:.4f}")
print(f"P-Value: {p_value:.4f}")


# %%
# Local SHAP for the final XGB sigmoid forecast
# how much past history
xgb_shap_history = max(
    abs(min(lags))
    for lags in (
        xgb_model_sigmoid._get_lags("target"),
        xgb_model_sigmoid._get_lags("past"),
        xgb_model_sigmoid._get_lags("future"),
    )
    if lags
)

# background data gives SHAP a reference distribution of lagged inputs.
xgb_shap_explainer = ShapExplainer(
    model=xgb_model_sigmoid,
    background_series=TimeSeries.from_series(y.iloc[:-1]),
    background_past_covariates=TimeSeries.from_dataframe(X.iloc[:-1]),
    background_num_samples=200,
    shap_method="permutation",
    seed=42,
    max_evals=10 * (2 * len(xgb_model_sigmoid.lagged_feature_names) + 1),
)

# explain the last forecast
xgb_shap_local = (
    xgb_shap_explainer.explain(
        foreground_series=TimeSeries.from_series(y.iloc[-xgb_shap_history - 1 : -1]),
        foreground_past_covariates=TimeSeries.from_dataframe(X.iloc[-xgb_shap_history - 1 : -1]),
        horizons=[1],
    )
    .get_explanation(horizon=1)
    .to_dataframe(copy=False)
    .iloc[0]
    .rename("shap_value")
    .reset_index()
    .rename(columns={"index": "feature"})
)

xgb_shap_local["abs_shap"] = xgb_shap_local["shap_value"].abs()
xgb_shap_local = xgb_shap_local.sort_values("abs_shap", ascending=False).reset_index(drop=True)

# group together
xgb_shap_local_grouped = (
    xgb_shap_local.assign(
        base_feature=lambda df: (
            df["feature"]
            .str.replace(r"_(target|pastcov|futcov)_lag-?\d+$", "", regex=True)
            .str.replace(r"_statcov_target_.*$", "", regex=True)
        )
    )
    .groupby("base_feature", as_index=False)
    .agg(
        shap_value=("shap_value", "sum"),
        abs_shap=("abs_shap", "sum"),
    )
    .sort_values("abs_shap", ascending=False)
    .reset_index(drop=True)
)
print("SHAP values for XGB_sigmoid/custom loss:")
print(xgb_shap_local_grouped.head(5))

# %%
