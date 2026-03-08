import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from darts import TimeSeries
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

warnings.filterwarnings("ignore", category=UserWarning)

def evaluation(
    model_name,
    y_true,
    y_pred,
    prev_levels_test = None,
    actual_levels_test = None,
):
    if (prev_levels_test is None or actual_levels_test is None):
        raise ValueError("Must provide prev_levels_test and actual_levels_test for regression evaluation.")
    
    if model_name is None:
        print("No model name provided.")

    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"RMSE in growth rates: {rmse:.4f}")
    print(f"MAE in growth rates: {mae:.4f}")
    pred_levels_test = prev_levels_test * (1 + y_pred/100)
    rmse_levels = root_mean_squared_error(actual_levels_test, pred_levels_test)
    mae_levels = mean_absolute_error(actual_levels_test, pred_levels_test)
    print(f"RMSE in absolute levels: {rmse_levels:.4f}")
    print(f"MAE in absolute levels: {mae_levels:.4f}")
    metrics = {
        "model": model_name,
        "rmse_growth": float(rmse),
        "mae_growth": float(mae),
        "rmse_levels": float(rmse_levels),
        "mae_levels": float(mae_levels),
    }

    return pd.DataFrame([metrics])

def darts_pipeline(
    model_cls,
    base_params,
    X_train, y_train,
    X_test, y_test,
    param_grid=None,
    n_splits=5,
    recalibrate_every=None,
    scale_features=True,
    scale_target=False,
    scoring_metric=mean_absolute_error,
    validation_refit_each_step=True,
):
    """
    Expanding-window walk-forward, 1-step ahead, PAST COVARIATES ONLY.

    Dataset is such that X_t is NOT known at time t. So we forecast y_t using covariates only up to t-1. 

    Hyperparameters are tuned using TimeSeriesSplit CV on the training set, with the same "past covariates only" logic applied to each fold.

    Forecasting is done in a walk-forward manner across the test set. 

    Initially, i wanted to recalibrate and do walk forward for validation too, but these are too time consuming. so now its all not used.
    """
    def _resolve_model_kwargs(params: dict) -> dict:
        """
         support sklearn-style nested params for wrapped estimators:
          - model__alpha, model__l1_ratio, ...
        by applying them to params["model"] before constructing the Darts model.
        """
        resolved = dict(params)
        nested = {
            k.split("__", 1)[1]: v
            for k, v in list(resolved.items())
            if k.startswith("model__")
        }
        for k in [k for k in resolved if k.startswith("model__")]:
            resolved.pop(k, None)

        if not nested:
            return resolved

        if "model" not in resolved or resolved["model"] is None:
            raise ValueError(
                "Received model__* params but base_params has no `model` estimator."
            )

        estimator = clone(resolved["model"])
        estimator = estimator.set_params(**nested)
        resolved["model"] = estimator
        return resolved

    def _supports_past(m) -> bool:
        return bool(getattr(m, "supports_past_covariates", False))

    def _prep_current_window(X_hist, y_hist):
        """
        Fit scalers on the currently available history only (up to t-1).
        """
        if X_hist is not None and (not X_hist.empty) and scale_features:
            xsc = StandardScaler()
            X_hist_p = pd.DataFrame(
                xsc.fit_transform(X_hist),
                index=X_hist.index,
                columns=X_hist.columns,
            )
        else:
            X_hist_p = X_hist

        if scale_target:
            ysc = StandardScaler()
            y_hist_p = pd.Series(
                ysc.fit_transform(y_hist.values.reshape(-1, 1)).ravel(),
                index=y_hist.index,
                name=y_hist.name,
            )
        else:
            ysc = None
            y_hist_p = y_hist

        return X_hist_p, y_hist_p, ysc

    def _fit_predict_past_only(m, y_tr_ts, X_tr_p, n):
        """
        Predict n steps ahead using ONLY past_covariates up to the end of training covariates.
        For tuning folds: this means covariates stop at fold train end, not extending into val.
        """
        if (X_tr_p is None) or X_tr_p.empty or (not _supports_past(m)):
            m.fit(series=y_tr_ts, verbose=False)
            return m.predict(n=n, verbose=False)

        X_tr_ts = TimeSeries.from_dataframe(X_tr_p)
        m.fit(series=y_tr_ts, past_covariates=X_tr_ts, verbose=False)
        return m.predict(n=n, past_covariates=X_tr_ts, verbose=False)

    def _score_fold_walk_forward(merged_params, X_tr, y_tr, X_va, y_va):
        """
        Score one CV fold with one-step recursive walk-forward over the validation segment.
        This avoids requiring future covariates beyond each step's train end.
        """
        curr_X = None if X_tr is None else X_tr.copy()
        curr_y = y_tr.copy()
        preds = []

        for j in range(len(y_va)):
            X_t = None if curr_X is None else X_va.iloc[j:j+1]
            y_t = y_va.iloc[j:j+1]

            curr_X_p, curr_y_p, ysc = _prep_current_window(curr_X, curr_y)
            y_ts = TimeSeries.from_series(curr_y_p)

            m = model_cls(**merged_params)
            pred_ts = _fit_predict_past_only(m, y_ts, curr_X_p, n=1)

            pred = float(pred_ts.values().ravel()[0])
            if scale_target and ysc is not None:
                pred = float(ysc.inverse_transform([[pred]])[0][0])

            preds.append(pred)

            curr_y = pd.concat([curr_y, y_t])
            if curr_X is not None:
                curr_X = pd.concat([curr_X, X_t])

        pred_series = pd.Series(np.asarray(preds), index=y_va.index, name="pred")
        return scoring_metric(y_va, pred_series)

    def _score_fold_no_refit(merged_params, X_tr, y_tr, X_va, y_va):
        """
        Score one CV fold by fitting once on fold-train and then predicting
        one-step ahead across validation without re-fitting at each step.
        """
        if X_tr is not None and (not X_tr.empty) and scale_features:
            xsc = StandardScaler()
            X_tr_p = pd.DataFrame(
                xsc.fit_transform(X_tr),
                index=X_tr.index,
                columns=X_tr.columns,
            )
            X_va_p = pd.DataFrame(
                xsc.transform(X_va),
                index=X_va.index,
                columns=X_va.columns,
            )
        else:
            X_tr_p = X_tr
            X_va_p = X_va

        if scale_target:
            ysc = StandardScaler()
            y_tr_p = pd.Series(
                ysc.fit_transform(y_tr.values.reshape(-1, 1)).ravel(),
                index=y_tr.index,
                name=y_tr.name,
            )
            y_va_p = pd.Series(
                ysc.transform(y_va.values.reshape(-1, 1)).ravel(),
                index=y_va.index,
                name=y_va.name,
            )
        else:
            ysc = None
            y_tr_p = y_tr
            y_va_p = y_va

        m = model_cls(**merged_params)
        y_tr_ts = TimeSeries.from_series(y_tr_p)
        if X_tr_p is None or X_tr_p.empty or (not _supports_past(m)):
            m.fit(series=y_tr_ts, verbose=False)
        else:
            X_tr_ts = TimeSeries.from_dataframe(X_tr_p)
            m.fit(series=y_tr_ts, past_covariates=X_tr_ts, verbose=False)

        preds = []
        for j in range(len(y_va)):
            y_hist_p = pd.concat([y_tr_p, y_va_p.iloc[:j]])
            y_hist_ts = TimeSeries.from_series(y_hist_p)

            if X_tr_p is None or X_tr_p.empty or (not _supports_past(m)):
                pred_ts = m.predict(n=1, series=y_hist_ts, verbose=False)
            else:
                X_hist_p = pd.concat([X_tr_p, X_va_p.iloc[:j]])
                X_hist_ts = TimeSeries.from_dataframe(X_hist_p)
                pred_ts = m.predict(
                    n=1,
                    series=y_hist_ts,
                    past_covariates=X_hist_ts,
                    verbose=False,
                )

            pred = float(pred_ts.values().ravel()[0])
            if scale_target and ysc is not None:
                pred = float(ysc.inverse_transform([[pred]])[0][0])
            preds.append(pred)

        pred_series = pd.Series(np.asarray(preds), index=y_va.index, name="pred")
        return scoring_metric(y_va, pred_series)

    def _tune(X_curr, y_curr):
        if not param_grid:
            return {}

        tscv = TimeSeriesSplit(n_splits=n_splits)
        grid = ParameterGrid(param_grid)

        best_score = float("inf")
        best_params = {}

        for p in grid:
            merged = _resolve_model_kwargs({**base_params, **p})
            fold_scores = []

            for tr_idx, va_idx in tscv.split(y_curr):
                y_tr, y_va = y_curr.iloc[tr_idx], y_curr.iloc[va_idx]
                X_tr = None if X_curr is None else X_curr.iloc[tr_idx]
                X_va = None if X_curr is None else X_curr.iloc[va_idx]

                if validation_refit_each_step:
                    fold_scores.append(
                        _score_fold_walk_forward(merged, X_tr, y_tr, X_va, y_va)
                    )
                else:
                    fold_scores.append(
                        _score_fold_no_refit(merged, X_tr, y_tr, X_va, y_va)
                    )

            avg = float(np.mean(fold_scores))
            if avg < best_score:
                best_score = avg
                best_params = p

        return best_params

    # ----------------- initial tuning -----------------
    print("--- initial tuning (past_covariates only) ---")
    best_params = _tune(X_train, y_train)
    print("best_params:", best_params)

    # ----------------- walk-forward -----------------
    curr_X = None if X_train is None else X_train.copy()
    curr_y = y_train.copy()
    preds = []
    last_trained_model = None

    for i in range(len(y_test)):
        if recalibrate_every and i > 0 and i % recalibrate_every == 0:
            print(f"⏳ step {i}: recalibrating hyperparameters...")
            best_params = _tune(curr_X, curr_y)
            print("new best_params:", best_params)

        # next realized point (unknown at forecast time)
        X_t = None if curr_X is None else X_test.iloc[i:i+1]   # X at time t (UNKNOWN at prediction time)
        y_t = y_test.iloc[i:i+1]                               # y at time t (to be predicted)

        # scale on expanding window using data up to t-1 ONLY
        curr_X_p, curr_y_p, ysc = _prep_current_window(curr_X, curr_y)

        y_ts = TimeSeries.from_series(curr_y_p)

        merged = _resolve_model_kwargs({**base_params, **best_params})
        m = model_cls(**merged)

        # predict 1-step using ONLY covariates up to t-1
        if curr_X_p is None or curr_X_p.empty or (not _supports_past(m)):
            m.fit(series=y_ts, verbose=False)
            last_trained_model = m
            pred_ts = m.predict(n=1, verbose=False)
        else:
            X_tr_ts = TimeSeries.from_dataframe(curr_X_p)
            m.fit(series=y_ts, past_covariates=X_tr_ts, verbose=False)
            last_trained_model = m
            pred_ts = m.predict(n=1, past_covariates=X_tr_ts, verbose=False)

        pred = float(pred_ts.values().ravel()[0])
        if scale_target and ysc is not None:
            pred = float(ysc.inverse_transform([[pred]])[0][0])

        preds.append(pred)

        # AFTER y_t is observed, we can append (X_t, y_t) to training
        curr_y = pd.concat([curr_y, y_t])
        if curr_X is not None:
            curr_X = pd.concat([curr_X, X_t])

    if last_trained_model is None:
        # No walk-forward steps were run; fit once on the current train window.
        final_params = _resolve_model_kwargs({**base_params, **best_params})
        last_trained_model = model_cls(**final_params)

        curr_X_p, curr_y_p, _ = _prep_current_window(curr_X, curr_y)
        y_ts = TimeSeries.from_series(curr_y_p)

        if curr_X_p is None or curr_X_p.empty or (not _supports_past(last_trained_model)):
            last_trained_model.fit(series=y_ts, verbose=False)
        else:
            X_tr_ts = TimeSeries.from_dataframe(curr_X_p)
            last_trained_model.fit(series=y_ts, past_covariates=X_tr_ts, verbose=False)

    return pd.Series(preds, index=y_test.index), last_trained_model

def mean_expected_profit_sigmoid(actual, pred, prev_levels, steepness=0.001, max_demand=1, base_margin=2000, full_list=False):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    prev = np.asarray(prev_levels, dtype=float) 

    n_auctions = len(pred) - 1
    if n_auctions <= 0:
        return np.nan

    expected_profits = []

    for t in range(n_auctions):
        budget = pred[t]
        price_t = actual[t]
        price_t_next = actual[t + 1]
        
        expected_market_price = prev[t]  
        inflection_point = expected_market_price # The point where demand starts to drop significantly
        
        # Sigmoid demand curve
        demand = max_demand / (1 + np.exp(steepness * (base_margin +budget - inflection_point)))
        
        if budget < price_t:
            realized_profit = (base_margin + budget - price_t_next) * demand
        else:
            realized_profit = (base_margin + budget - price_t) * demand

        expected_profits.append(realized_profit)

    if full_list:
        return expected_profits
    else:
        return float(np.mean(expected_profits))
def evaluate_expected_profit_sigmoid(actual, pred, prev_levels, model_name="Model", steepness=0.001, base_margin=2000):
    mean_profit = mean_expected_profit_sigmoid(
        actual=actual, pred=pred, prev_levels=prev_levels, 
        steepness=steepness, base_margin=base_margin
    )
    return {"Model": model_name, "Net Expected Profit (SGD)": round(mean_profit, 2)}
def profit_cv_score_sigmoid(y_true_growth, y_pred_growth, prev_levels_lookup, steepness=0.001, max_demand=1, base_margin=2000):
    """
    Scoring metric for CV: converts growth-rate predictions to absolute levels, 
    then scores expected profit using your Sigmoid demand curve.
    Returns negative mean profit because hyperparameter search algorithms minimize the score.
    """
    # 1. Convert to Pandas Series for safe alignment
    y_true_s = pd.Series(y_true_growth).astype(float)
    y_pred_s = pd.Series(y_pred_growth, index=y_true_s.index).astype(float)
    prev_s = pd.Series(prev_levels_lookup).reindex(y_true_s.index).astype(float)

    # 2. Drop any missing overlapping indices
    valid = prev_s.notna() & y_true_s.notna() & y_pred_s.notna()
    y_true_s = y_true_s[valid]
    y_pred_s = y_pred_s[valid]
    prev_s = prev_s[valid]

    # Need at least 2 points to calculate t and t+1
    if len(y_true_s) < 2:
        return float("inf")

    # 3. Reconstruct absolute SGD levels from % growth rates
    actual_levels = prev_s * (1.0 + y_true_s / 100.0)
    pred_levels = prev_s * (1.0 + y_pred_s / 100.0)

    # 4. Calculate profit using your exact Sigmoid function
    mean_profit = mean_expected_profit_sigmoid(
        actual=actual_levels.values,
        pred=pred_levels.values,
        prev_levels=prev_s.values,
        steepness=steepness,
        max_demand=max_demand,
        base_margin=base_margin
    )

    if np.isnan(mean_profit):
        return float("inf")

    # 5. Return NEGATIVE profit so grid search (minimizer) maximizes your money
    return -mean_profit

def plot_profit_curves_sigmoid(
    actual_levels, 
    prev_levels, 
    models_dict, 
    base_margin=2000, 
    num_points=100
):
    """
    Evaluates profit across varying Sigmoid steepness (elasticity) 
    """
    # 1. Convert directly to numpy arrays
    true_test_actual = np.asarray(actual_levels)
    true_test_prev = np.asarray(prev_levels)
    
    # 2. Define the realistic range for S-Curve steepness (k)
    steepness_to_test = np.linspace(0.0001, 0.0025, num_points)
    plt.figure(figsize=(12, 7))
    
    # 3. Process each model
    for model_name, config in models_dict.items():
        color = config['color']
        true_test_pred = np.asarray(config['preds'])
        
        profits = []
        
        # --- THE EVALUATION LOOP ---
        for k in steepness_to_test:
            
            # Evaluate predictions on the entire test set
            res = evaluate_expected_profit_sigmoid(
                actual=true_test_actual, 
                pred=true_test_pred, 
                prev_levels=true_test_prev, 
                model_name=model_name, 
                steepness=k, 
                base_margin=base_margin
            )
            profits.append(res["Net Expected Profit (SGD)"])
            
        # Add lines to the plot
        plt.plot(steepness_to_test, profits, label=f'{model_name}', color=color, linewidth=2.5)

    # --- Formatting the Chart ---
    plt.axhline(0, color='red', linestyle='--', linewidth=2, label='Break-Even (0 Profit)')
    
    plt.title(f'Profit Over Varying Customer Elasticity (Sigmoid S-Curve)\n(${base_margin} Base Margin, All 96 Observations)', fontsize=14)
    plt.xlabel('Customer Demand Steepness (k)\n(0.0001 = Brand Loyal/Inelastic, 0.0025 = Elastic/Price Sensitive)', fontsize=12)
    plt.ylabel('Net Expected Profit per Car (SGD)', fontsize=12)

    plt.xlim(steepness_to_test[0], steepness_to_test[-1])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def diebold_mariano_test(actual, pred1, pred2, loss='abs'):
    # 1. Calculate level errors
    e1 = actual - pred1
    e2 = actual - pred2
    
    # 2. Calculate the loss differential
    if loss == 'abs':
        d = np.abs(e1) - np.abs(e2)
    elif loss == 'sqr':
        d = (e1 ** 2) - (e2 ** 2)
    else:
        raise ValueError("Loss must be 'abs' or 'sqr'")
        
    # 3. Calculate mean and variance of the differential
    mean_d = np.mean(d)
    T = float(len(d))
    
    # Variance of the loss differential (ddof=0 for population variance)
    var_d = np.var(d, ddof=0)
    
    # 4. Calculate DM Statistic and p-value
    # Note: This is for 1-step ahead forecasts. 
    dm_stat = mean_d / np.sqrt(var_d / T)
    
    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value
def error_plot(
    actual_levels_test, 
    baseline_pred, 
    challenger_pred, 
    date_real
):
    loss_diff = np.abs(actual_levels_test - baseline_pred) - np.abs(actual_levels_test - challenger_pred)
    test_dates = date_real[-len(loss_diff):]

    plt.figure(figsize=(14, 6))
    colors = ['#3498db' if val > 0 else '#e74c3c' for val in loss_diff]
    plt.bar(test_dates, loss_diff, color=colors, width=15)
    plt.axhline(0, color='black', linewidth=1)
    plt.title('Absolute Loss Differential (Positive = Challenger Win)')
    plt.xlabel('Date')
    plt.ylabel('Error Reduction (Challenger Error - Baseline Error)')
    plt.grid(axis='y', alpha=0.3)
    plt.show()

def plot_test(
    y_true,
    y_pred,
    y_pred_2=None,
    title="Actual vs Predicted",
    xlabel="Time",
    ylabel="Quota Premium",
    y_pred_label="Predicted",
    y_pred_2_label="Predicted 2",
    date_real=None,
):
    # choose x-axis
    if date_real is not None:
        x_axis = pd.Series(date_real).loc[y_true.index].values
    else:
        x_axis = y_true.index

    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, y_true.values, label="Actual", marker="o")
    plt.plot(x_axis, y_pred.values, label=y_pred_label, marker="x")
    if y_pred_2 is not None:
        plt.plot(x_axis, y_pred_2.values, label=y_pred_2_label, marker=".")
    plt.title(title)
    plt.xlabel("Date" if date_real is not None else xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()
