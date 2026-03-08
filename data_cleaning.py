#%%
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from functions.data_cleaning_function import (
    add_lag_columns,
    plot_quota_premium_vs_column,
    plot_scatter_against_quota_premium,
)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf   
# %% import
REPO_ROOT = Path(__file__).parent
df = pd.read_csv(REPO_ROOT / "dataset_building" / "cleaned_dataset.csv")
# %% Check Data Types
df['date'] = pd.to_datetime(df['date']).dt.date
#check dtype of each column
print(df.dtypes)

#%%
#summary statistics of quota_premium_growth_rate
print(df["quota_premium_growth_rate"].describe())
print(df["quota_premium_growth_rate"].nlargest(10))
print(df["quota_premium_growth_rate"].nsmallest(10))
print(df['overdemand_growth_rate'].nsmallest(10))

#%%
#winsorise the quota_premium_growth_rate at 0.5% and 99.5% (around 3 data points on each side, all during 2008)
lower_bound = df["quota_premium_growth_rate"].quantile(0.005)
upper_bound = df["quota_premium_growth_rate"].quantile(0.995)
df["quota_premium_growth_rate"] = np.where(df["quota_premium_growth_rate"] < lower_bound, lower_bound, df["quota_premium_growth_rate"])
df["quota_premium_growth_rate"] = np.where(df["quota_premium_growth_rate"] > upper_bound, upper_bound, df["quota_premium_growth_rate"])

#apply the same to quota_premium_cat_b_growth_rate
lower_bound_cat_b = df["quota_premium_cat_b_growth_rate"].quantile(0.005)
upper_bound_cat_b = df["quota_premium_cat_b_growth_rate"].quantile(0.995)
df["quota_premium_cat_b_growth_rate"] = np.where(df["quota_premium_cat_b_growth_rate"] < lower_bound_cat_b, lower_bound_cat_b, df["quota_premium_cat_b_growth_rate"])
df["quota_premium_cat_b_growth_rate"] = np.where(df["quota_premium_cat_b_growth_rate"] > upper_bound_cat_b, upper_bound_cat_b, df["quota_premium_cat_b_growth_rate"])

#premium_cat_a divided by premium_cat_b
df["cat_a_divide_cat_b"] = df["quota_premium"] / df["quota_premium_cat_b"]

#%% ADF for every column in df except the first 3 columns
for col in df.columns[3:]:
    result = adfuller(df[col].dropna())
    print(f"Column: {col}")
    print("ADF Statistic: %f" % result[0])
    print("p-value: %f" % result[1])
    print("Used lag:", result[2])
    print("-" * 30)

#%% Align columns
TO_DROP = ["regis_cat_a","regis_cat_b","bidding","year"] #drop some things. regis idw anymore because im not doing more than 1 step ahead, so deregis data is enough
df.drop(columns=TO_DROP, inplace=True)
#lag 3 for data released 1 month late
NN_LAG_3 = [c for c in df.columns if c.startswith("cpi_")] + ["regis_cat_a_growth_rate","regis_cat_b_growth_rate","total_cars_cat_a_growth_rate","total_cars_cat_b_growth_rate","deregis_cat_a", "deregis_cat_a_growth_rate","total_cars_cat_a","total_cars_cat_b"]
NN_FORWARD_1 = ['quota_per_bid', 'quota_per_bid_growth_rate']
NN_LAG_1 = ['pqp_cat_a','pqp_cat_b','pqp_cat_a_growth_rate','pqp_cat_b_growth_rate']

df_nn = add_lag_columns(df, NN_LAG_3, lags=3)
df_nn = add_lag_columns(df_nn, NN_FORWARD_1, lags=-1)
df_nn = add_lag_columns(df_nn, NN_LAG_1, lags=1)
# %%
#drop rows older than Jan 2003
df_nn["date"] = pd.to_datetime(df_nn["date"])
df_nn = df_nn[df_nn["date"] >= pd.to_datetime("2003-01-01")]

#%%
missing_cols_nn = df_nn.columns[df_nn.isnull().any()]
missing_percent_nn = df_nn[missing_cols_nn].isnull().mean() * 100
print("Percentage of missing values in df_nn:\n", missing_percent_nn)
df_nn = df_nn.drop(columns=missing_percent_nn[missing_percent_nn > 60].index)

#%%#=================EDA=================
#align X_t-1 to Y_t so we can do correlation and scatter plots
NO_SHIFT = ['date']
SHIFT = [c for c in df_nn.columns if c not in NO_SHIFT]
df_eda = add_lag_columns(df_nn, SHIFT, lags=1, keep_original=SHIFT)
df_eda.rename(columns = {"quota_per_bid_lag_-1_lag_1": "quota_per_bid"}, inplace=True)
df_eda.rename(columns = {"quota_per_bid_growth_rate_lag_-1_lag_1": "quota_per_bid_growth_rate"}, inplace=True)

#%% sum stats
print(df_eda["quota_premium_growth_rate"].describe())
print(df_eda["quota_premium"].describe())
#%%
# plot quota_premium over time
plt.figure(figsize=(12, 6))
plt.plot(df_eda["date"], df_eda["quota_premium"], label="Quota Premium", color="tab:blue")
plt.xlabel("Date")
plt.ylabel("Quota Premium")
plt.title("Quota Premium Over Time")
plt.legend()
plt.show()
#%% correlation matrix
supply_var = ["quota_premium_growth_rate","quota_per_bid","quota_per_bid_growth_rate"]

corr_matrix = df_eda[supply_var].corr(method="pearson")
#keep only first column
corr_matrix = corr_matrix.iloc[:, 0].drop("quota_premium_growth_rate")

#%%
# find others with high correlation
target = "quota_premium_growth_rate"
corr_df = df_eda.drop(columns=["date"], errors="ignore").select_dtypes(include="number")
corr_with_target = corr_df.corr()[target].drop(target)
cols_gt_02 = corr_with_target[corr_with_target.abs() > 0.2].sort_values(key=lambda s: s.abs(), ascending=False)

#append corr_matrix below cols_gt_02
corr_matrix = pd.concat([cols_gt_02, corr_matrix], axis=0).drop_duplicates()
print(corr_matrix)
# %%
# plot ACF and PACF of quota_premium_growth_rate
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_acf(df["quota_premium_growth_rate"].dropna(), lags=30, ax=plt.gca())
plt.title("ACF of Quota Premium Growth Rate")
plt.subplot(1, 2, 2)
plot_pacf(df["quota_premium_growth_rate"].dropna(), lags=30, ax=plt.gca())
plt.title("PACF of Quota Premium Growth Rate")
plt.tight_layout()
plt.show()

#persistance is actually quite low, 12 lags is lowkey overkill
#no signs of seasonality

#%%
plot_quota_premium_vs_column(
    df_eda,
    right_col="quota_per_bid",
    title="Quota Premium vs quota per bid",
    right_label="Quota per Bid",
)

plot_quota_premium_vs_column(
    df_eda,
    left_col="quota_premium",
    right_col="quota_premium_growth_rate",
    title="Quota Premium and growth rate",
)

plot_scatter_against_quota_premium(
    df_eda,
    y_col="quota_premium_growth_rate",
    x_col="cat_a_divide_cat_b_lag_1",
    title="Quota Premium vs Cat A Divide Cat B (Lag 1)",
)

plot_scatter_against_quota_premium(
    df_eda,
    y_col="quota_premium_growth_rate",
    x_col="overdemand_growth_rate_lag_1",
    title="Quota Premium vs Overdemand Growth Rate (Lag 1)",
)

#graphs not very useful, just show correlation matrix
# %%
df_nn.to_csv(REPO_ROOT / "final_dataset_nn.csv", index=False)
# %%

# df.to_csv(REPO_ROOT / "final_dataset.csv", index=False)

#pqp_cat_a and pqp_cat_b, entire column shift up 2 rows (1 month)
# df["pqp_cat_a"] = df["pqp_cat_a"].shift(-2)
# df["pqp_cat_b"] = df["pqp_cat_b"].shift(-2)
# df["pqp_cat_a_growth_rate"] = df["pqp_cat_a_growth_rate"].shift(-2)
# df["pqp_cat_b_growth_rate"] = df["pqp_cat_b_growth_rate"].shift(-2)
#=================Detailed manipulation of columns=================
#%%
#Create 10 years to 11 years ago columns for registration, as they can be indicator for future supply of quota

#create a fixed_date 
# dates = pd.date_range(end="2026-01-15", periods=len(df), freq="SMS")  # 1st + 15th
# df["date_fixed"] = dates[::-1].to_numpy()  # top row = 2026-01-15

# n = len(df)
# skip_months = pd.PeriodIndex(["2020-04", "2020-05", "2020-06"], freq="M")

# periods = n + 24  # buffer to account for skipped dates
# while True:
#     arr = pd.date_range(end="2026-01-15", periods=periods, freq="SMS")[::-1]  # top: 2026-01-15
#     arr = arr[~arr.to_period("M").isin(skip_months)]
#     if len(arr) >= n:
#         df["date_fixed"] = arr[:n].to_numpy()
#         break
#     periods += 24

# df["date_fixed"] = pd.to_datetime(df["date_fixed"])

# # lookup: date_fixed -> regis_cat_a
# lookup = (
#     df[["date_fixed", "regis_cat_a"]]
#     .set_index("date_fixed")["regis_cat_a"]
# )

# # 10y0m to 11y0m ago (inclusive)
# for total_months in range(10 * 12, 10 * 12 + 1):
#     years, months = divmod(total_months, 12)
#     col = f"regis_cat_a_{years}_year_{months}_months_ago"

#     shifted_dates = df["date_fixed"] - pd.DateOffset(years=years, months=months)
#     df[col] = shifted_dates.map(lookup)

# # lookup: date_fixed -> regis_cat_b
# lookup_b = df.set_index("date_fixed")["regis_cat_b"]

# # 9y0m to 11y2m ago (inclusive)
# for total_months in range(9 * 12, 11 * 12 + 2 + 1):
#     years, months = divmod(total_months, 12)
#     col = f"regis_cat_b_{years}_year_{months}_months_ago"
#     shifted_dates = df["date_fixed"] - pd.DateOffset(years=years, months=months)
#     df[col] = shifted_dates.map(lookup_b)
#%% Data structure for NN
# NN_NO_LAG = ['quota_per_bid', 'quota_per_bid_growth_rate', 'pqp_cat_a_growth_rate', 'pqp_cat_b_growth_rate','date'] 
# NN_LAG_2 = ['gtrends_coe', 'gtrends_sgcarmart']
# NN_LAG_3 = [c for c in df.columns if c.startswith("cpi_")] + ["regis_cat_a_growth_rate","regis_cat_b_growth_rate","total_cars_cat_a_growth_rate","total_cars_cat_b_growth_rate"]
# NN_LAG_1 = [c for c in df.columns if c not in NN_NO_LAG and c not in NN_LAG_3 and c not in NN_LAG_2]


# df_nn = add_lag_columns(df, NN_LAG_1, lags=1)
# df_nn = add_lag_columns(df_nn, NN_LAG_2, lags=2)
# df_nn = add_lag_columns(df, NN_LAG_3, lags=3)
# df_nn = add_lag_columns(df_nn, NN_FORWARD_1, lags=-1)

# #%%
# NO_LAG = ["quota_per_bid", "quota_per_bid_growth_rate", "pqp_cat_a_growth_rate", "pqp_cat_b_growth_rate"]

# LAG_1 = [c for c in df.columns if c.startswith("policy_")]

# LAG_2_4_6 = ['gtrends_coe', 'gtrends_sgcarmart']

# LAG_1_TO_6 = ["quota","quota_premium","quota_premium_growth_rate","overdemand_growth_rate","quota_cat_b","quota_premium_cat_b_growth_rate","overdemand_cat_b_growth_rate","sora_1m","sti_close","sti_close_growth_rate","cat_a_divide_cat_b"]

# LAG_4_6 = ["regis_cat_a_growth_rate","regis_cat_b_growth_rate","total_cars_cat_a_growth_rate","total_cars_cat_b_growth_rate"] + [c for c in df.columns if c.startswith("cpi_")]


# #%%
# # Add lagged columns
# df = add_lag_columns(df, LAG_1, lags=1)
# df = add_lag_columns(df, LAG_2_4_6, lags=[2, 4, 6])
# df = add_lag_columns(df, LAG_1_TO_6, lags=(1, 6))
# df = add_lag_columns(df, LAG_4_6, lags=[4, 6])
# df = add_lag_columns(df, ["quota_per_bid","quota_per_bid_growth_rate"], lags=6, keep_original= ["quota_per_bid","quota_per_bid_growth_rate"])
# df = add_lag_columns(df, ["pqp_cat_a_growth_rate","pqp_cat_b_growth_rate"], lags=[2,4,6], keep_original=["pqp_cat_a_growth_rate","pqp_cat_b_growth_rate"])
# %%
# #columns with missing values, and percentage of missing values
# missing_cols = df.columns[df.isnull().any()]
# missing_percent = df[missing_cols].isnull().mean() * 100
# print("Percentage of missing values:\n", missing_percent)

# #drop columns with more than 50% missing values
# df = df.drop(columns=missing_percent[missing_percent > 50].index)

# %% plot quota_premium and quota_per_bid vs date
#correlation matrix table

