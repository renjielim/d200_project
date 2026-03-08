# %%
import pandas as pd
from pathlib import Path
import yfinance as yf

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import importlib
import functions.data_cleaning_function as dcf
importlib.reload(dcf)


from functions.data_cleaning_function import add_growth_rate_columns

# %load_ext autoreload
# %autoreload 2


#%%
#import files
df = pd.read_excel(REPO_ROOT / 'dataset_building/coe_prices.xlsx')
df.columns = df.columns.str.lower()

quota = pd.read_excel(REPO_ROOT / 'dataset_building/quota_pdf.xlsx')
quota.columns = quota.columns.str.lower()

registrations = pd.read_csv(REPO_ROOT / 'dataset_building/vehicle_registrations.csv')
registrations.columns = registrations.columns.str.lower()

total_population = pd.read_csv(REPO_ROOT / 'dataset_building/total_vehicle_population.csv')
total_population.columns = total_population.columns.str.lower()

# car_age = pd.read_csv(REPO_ROOT / 'dataset_building/car_age.csv')

pqp = pd.read_csv(REPO_ROOT / 'dataset_building/pqp.csv')

cpi = pd.read_csv(REPO_ROOT / 'dataset_building/cpi.csv')

sora = pd.read_csv(REPO_ROOT / 'dataset_building/sora.csv')

policy = pd.read_csv(REPO_ROOT / 'dataset_building/coe_policy_features_biweekly.csv')

trends = pd.read_csv(REPO_ROOT / 'dataset_building/gtrends_combined.csv')

deregistrations = pd.read_csv(REPO_ROOT / 'dataset_building/deregistrations.csv')

#%%
#cat B data
df_cat_b = df[df['category'].str.contains('Cat B')].reset_index(drop=True)
df_cat_b = df_cat_b.drop(columns=['category','number of successful bids'])
df_cat_b['date'] = pd.to_datetime(df_cat_b['date'], format='%Y-%m-%d').dt.date
df_cat_b = df_cat_b.sort_values(by='date', ascending=False).reset_index(drop=True)

#remove dates below 2002-05-01
df_cat_b = df_cat_b[df_cat_b['date'] >= pd.to_datetime('2002-05-01').date()].reset_index(drop=True)

df_cat_b = df_cat_b.rename(columns={'quota': 'quota_cat_b', 'quota premium': 'quota_premium_cat_b','total bids received': 'total_bids_received_cat_b'})

#create a 'overdemand' column of total_bids_received_cat_b divided by quota_cat_b
df_cat_b['overdemand_cat_b'] = df_cat_b['total_bids_received_cat_b'] / df_cat_b['quota_cat_b']
df_cat_b = df_cat_b.drop(columns=['total_bids_received_cat_b'])

#change to growth rates
df_cat_b = add_growth_rate_columns(df_cat_b, ['quota_premium_cat_b'], drop_original=False)
df_cat_b = add_growth_rate_columns(df_cat_b, ['overdemand_cat_b'], drop_original=True)


# %%
#removes rows unless category has Cat A
df = df[df['category'].str.contains('Cat A')].reset_index(drop=True)
df = df.drop(columns=['category', 'number of successful bids'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.date
df = df.sort_values(by='date', ascending=False).reset_index(drop=True)

#remove dates below 2002-05-01
df = df[df['date'] >= pd.to_datetime('2002-05-01').date()].reset_index(drop=True)

df.columns = df.columns.str.strip().str.replace(r"\s+", "_", regex=True)

df['overdemand'] = df['total_bids_received'] / df['quota']
df = df.drop(columns=['total_bids_received'])

#change to growth rates, drop original columns
df = add_growth_rate_columns(df, ['quota_premium'], drop_original=False)
df = add_growth_rate_columns(df, ['overdemand'], drop_original=False)

df = df.merge(df_cat_b[['date','quota_cat_b','quota_premium_cat_b','quota_premium_cat_b_growth_rate','overdemand_cat_b_growth_rate']], on='date', how='left')

# %%
quota["quota_per_bid"] = round(quota['quota']/(quota['number of months']
*2))

# clean quota 
quota_months = quota.loc[quota.index.repeat(quota["number of months"])].reset_index(drop=True)

quota_months.loc[:68, "date"] = pd.date_range(end="2026-04-01", periods=69, freq="MS")[::-1]
#some dates skipped as bidding paused during covid
quota_months.loc[70:, "date"] = pd.date_range(end="2020-04-01", periods=len(quota_months)-70, freq="MS")[::-1]
quota_months['date'] = pd.to_datetime(quota_months['date'], format='%Y-%m-%d').dt.date

#removed dates above 2026-01-01
quota_months = quota_months[quota_months['date'] <= pd.to_datetime('2026-01-01').date()].reset_index(drop=True)

#remove 2020-04-01 as bidding paused during covid
quota_months = quota_months[quota_months['date'] != pd.to_datetime('2020-04-01').date()].reset_index(drop=True)
quota_months.columns = ['date','no_months','quota','increase_vehicle_pop','replacement_deregistered','adjustments','quota_per_bid']

quota_months = add_growth_rate_columns(quota_months, ['quota_per_bid'], shift_period=-3, drop_original=False)

#change to bidding frequency
quota_biweekly = quota_months.loc[quota_months.index.repeat(2)].reset_index(drop=True)

#%%
#add quota_per_bid column into df, just do a merge, not on date as dates are not exactly the same, but on index, as both are sorted by date
df = df.merge(quota_biweekly[['quota_per_bid','quota_per_bid_growth_rate']], left_index=True, right_index=True, how='left')

# %%
#clean vehicle registrations data
registrations = registrations.iloc[8:].reset_index(drop=True)
registrations.columns = registrations.iloc[0]
registrations = registrations.drop(0).reset_index(drop=True)
registrations = registrations.iloc[:420].reset_index(drop=True) #keep data from 1991 jan onwards
registrations = registrations.iloc[:, [0,2,3]]
registrations.columns = ['date', 'regis_cat_a', 'regis_cat_b']
registrations["date"] = pd.to_datetime(
    registrations["date"].str.strip(),
    format="%Y %b"
).dt.date
#remove 2020-04-01, 2020-05-01 and 2020-06-01 as bidding paused during covid
registrations = registrations[~registrations['date'].isin([pd.to_datetime('2020-04-01').date(), pd.to_datetime('2020-05-01').date(), pd.to_datetime('2020-06-01').date()])].reset_index(drop=True)

registrations = add_growth_rate_columns(registrations, ['regis_cat_a', 'regis_cat_b'], drop_original=False)

registrations_biweekly = registrations.loc[registrations.index.repeat(2)].reset_index(drop=True)

rb = registrations_biweekly[["date","regis_cat_a", "regis_cat_b", "regis_cat_a_growth_rate", "regis_cat_b_growth_rate"]].reset_index(drop=True)
rb.index = rb.index + 2   # start filling from 3rd row (index 2)

df = (
    df.reset_index(drop=True)
      .merge(rb[["regis_cat_a", "regis_cat_b", "regis_cat_a_growth_rate", "regis_cat_b_growth_rate"]], left_index=True, right_index=True, how="outer")
      .sort_index()
      .reset_index(drop=True)
)

df["date"] = pd.to_datetime(df["date"], errors="coerce").combine_first(
    pd.to_datetime(rb["date"], errors="coerce")
)


# %%
#clean total vehicle population data
total_population = total_population.iloc[8:].reset_index(drop=True)
total_population.columns = total_population.iloc[0]
total_population = total_population.drop(0).reset_index(drop=True)
total_population = total_population.iloc[:420].reset_index(drop=True) #keep data from 1991 jan onwards
total_population = total_population.iloc[:, [0,2,3]]
total_population.columns = ['date', 'total_cars_cat_a', 'total_cars_cat_b']
total_population["date"] = pd.to_datetime(
    total_population["date"].str.strip(),
    format="%Y %b"
).dt.date

#remove 2020-04-01, 2020-05-01 and 2020-06-01 as bidding paused during covid
total_population = total_population[~total_population['date'].isin([pd.to_datetime('2020-04-01').date(), pd.to_datetime('2020-05-01').date(), pd.to_datetime('2020-06-01').date()])].reset_index(drop=True)

total_population = add_growth_rate_columns(total_population, ['total_cars_cat_a', 'total_cars_cat_b'], drop_original=False)

total_population_biweekly = total_population.loc[total_population.index.repeat(2)].reset_index(drop=True)

tpb = total_population_biweekly[["total_cars_cat_a", "total_cars_cat_b", "total_cars_cat_a_growth_rate", "total_cars_cat_b_growth_rate"]].reset_index(drop=True)
tpb.index = tpb.index + 2   # start filling from 3rd row (index 2)

df = (
    df.reset_index(drop=True)
      .merge(tpb[["total_cars_cat_a", "total_cars_cat_b", "total_cars_cat_a_growth_rate", "total_cars_cat_b_growth_rate"]], left_index=True, right_index=True, how="outer")
      .sort_index()
      .reset_index(drop=True)
)
# %%
#clean car age data REMOVE, yearly data no variation
# car_age = car_age.iloc[8:].reset_index(drop=True)
# car_age.columns = car_age.iloc[0]
# car_age = car_age.drop(0).reset_index(drop=True)

# car_age = car_age.iloc[:34].reset_index(drop=True) #keep data from 1991 jan onwards
# #drop column 1
# car_age = car_age.drop(car_age.columns[1], axis=1)
# car_age = car_age.rename(columns={car_age.columns[0]: 'date'})
# #set all columns except date to numeric
# car_age.iloc[:, 1:] = car_age.iloc[:, 1:].apply(pd.to_numeric)
# #sum across columns 1 to 10 to get car age younger than 10 years, and sum across columns 11 to the end to get car age older than 10 years
# car_age['car_age_younger_than_10'] = car_age.iloc[:, 1:11].sum(axis=1)
# car_age['car_age_older_than_10'] = car_age.iloc[:, 11:].sum(axis=1)
# car_age = car_age[['date', 'car_age_younger_than_10', 'car_age_older_than_10']]
# # prepare car_age (its `date` is year-only)
# car_age = car_age.rename(columns={"date": "year"})

# car_age["year"] = pd.to_numeric(car_age["year"], errors="coerce").astype("Int64")
# df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year.astype("Int64")

# df = df.merge(car_age, on="year", how="left")

#%% clean pqp data
pqp = pqp.iloc[9:].reset_index(drop=True)
pqp.columns = pqp.iloc[0]
pqp = pqp.drop(0).reset_index(drop=True)
pqp = pqp.iloc[:288].reset_index(drop=True)
pqp.columns = ['date', 'pqp_cat_a', 'pqp_cat_b']

pqp["date"] = pd.to_datetime(
    pqp["date"].str.strip(),
    format="%Y %b"
).dt.date

#remove 2020-04-01, 2020-05-01 and 2020-06-01 as bidding paused during covid
pqp = pqp[~pqp['date'].isin([pd.to_datetime('2020-04-01').date(), pd.to_datetime('2020-05-01').date(), pd.to_datetime('2020-06-01').date()])].reset_index(drop=True)

pqp = add_growth_rate_columns(pqp, ['pqp_cat_a', 'pqp_cat_b'], drop_original=False)

pqp_biweekly = pqp.loc[pqp.index.repeat(2)].reset_index(drop=True)

df = df.merge(pqp_biweekly[['pqp_cat_a', 'pqp_cat_b','pqp_cat_a_growth_rate', 'pqp_cat_b_growth_rate']], left_index=True, right_index=True, how='left')
# %%
#clean cpi data
cpi = cpi.iloc[8:].reset_index(drop=True)
cpi.columns = cpi.iloc[0]
cpi = cpi.drop(0).reset_index(drop=True)
cpi = cpi.iloc[:420].reset_index(drop=True) #keep data from 1991 jan onwards
cpi.columns = ['date', 'cpi_all', 'cpi_transport','cpi_private_transport','cpi_cars','cpi_transport_accessories','cpi_petrol','cpi_lubricants','cpi_maintenance','cpi_transport_others','cpi_transport_insurance','cpi_land_transport','cpi_bus_train_fare','cpi_ptp_transport_services','cpi_commuting_fares','cpi_other_land_transport_services']

cpi["date"] = pd.to_datetime(
    cpi["date"].str.strip(),
    format="%Y %b"
).dt.date

#remove 2020-04-01, 2020-05-01 and 2020-06-01 as bidding paused during covid
cpi = cpi[~cpi['date'].isin([pd.to_datetime('2020-04-01').date(), pd.to_datetime('2020-05-01').date(), pd.to_datetime('2020-06-01').date()])].reset_index(drop=True)

cpi = add_growth_rate_columns(cpi, ['cpi_all', 'cpi_transport','cpi_private_transport','cpi_cars','cpi_transport_accessories','cpi_petrol','cpi_lubricants','cpi_maintenance','cpi_transport_others','cpi_transport_insurance','cpi_land_transport','cpi_bus_train_fare','cpi_ptp_transport_services','cpi_commuting_fares','cpi_other_land_transport_services'], drop_original=True)

#add 2 empty rows at the beginning to align with df
cpi_biweekly = cpi.loc[cpi.index.repeat(2)].reset_index(drop=True)
cpi_biweekly.index = cpi_biweekly.index + 2   # start filling from 3rd row (index 2)

df = (
    df.reset_index(drop=True)
      .merge(cpi_biweekly[['cpi_all_growth_rate', 'cpi_transport_growth_rate','cpi_private_transport_growth_rate','cpi_cars_growth_rate','cpi_transport_accessories_growth_rate','cpi_petrol_growth_rate','cpi_lubricants_growth_rate','cpi_maintenance_growth_rate','cpi_transport_others_growth_rate','cpi_transport_insurance_growth_rate','cpi_land_transport_growth_rate','cpi_bus_train_fare_growth_rate','cpi_ptp_transport_services_growth_rate','cpi_commuting_fares_growth_rate','cpi_other_land_transport_services_growth_rate']], left_index=True, right_index=True, how="outer")
      .sort_index()
      .reset_index(drop=True)
)

df = df.replace('na', pd.NA)

#%% clean SORA
sora = sora.drop(columns=['SORA Value Date', 'Unnamed: 1', 'Unnamed: 2'])
sora.columns = ['date', 'sora_1m']

#remove rows where sora_1m is not numeric
sora = sora[pd.to_numeric(sora['sora_1m'], errors='coerce').notnull()].reset_index(drop=True)

sora["date"] = pd.to_datetime(
    sora["date"].astype(str).str.strip(),
    format="%d %b %Y"
).dt.date

#reverse sort by date
sora = sora.sort_values(by='date', ascending=False).reset_index(drop=True)
df['date'] = df['date'].dt.date

# make sure both are datetime64 and sorted
df2 = df.copy()
sora2 = sora.copy()

df2["date"] = pd.to_datetime(df2["date"])
sora2["date"] = pd.to_datetime(sora2["date"])

df2["_order"] = range(len(df2))
df2 = df2.sort_values("date")
sora2 = sora2.sort_values("date")

df = pd.merge_asof(
    df2,
    sora2,
    on="date",
    direction="forward",          # next later date if no exact match
    allow_exact_matches=True
).sort_values("_order").drop(columns="_order")

df["date"] = pd.to_datetime(df["date"])

df.loc[df["date"] < pd.Timestamp("2005-08-01"), "sora_1m"] = pd.NA


# %% STI data
sti = yf.download("^STI", start="2002-01-01", auto_adjust=True, progress=True)
sti = sti.reset_index().rename(columns={"Date": "date"})
sti = sti[["date", "Close"]].rename(columns={"Close": "sti_close"})

sti["date"] = pd.to_datetime(sti["date"]).dt.date
sti = sti.sort_values(by='date', ascending=False).reset_index(drop=True)

sti.columns = ['date', 'sti_close']

df["date"] = pd.to_datetime(df["date"])
sti["date"] = pd.to_datetime(sti["date"])


df = pd.merge_asof(
    df.sort_values("date"),
    sti.sort_values("date"),
    on="date",
    direction="backward",
    allow_exact_matches=True
).sort_values("date").reset_index(drop=True)

df = df.sort_values(by='date', ascending=False).reset_index(drop=True)

df = add_growth_rate_columns(df, ['sti_close'], drop_original=False)
#%% Policy changes
#flip order of rows
policy = policy.iloc[::-1].reset_index(drop=True)

#prefix all columns except date with "policy_"
policy = policy.rename(columns={col: f"policy_{col}" if col != "date" else col for col in policy.columns})

policy["date"] = pd.to_datetime(
    policy["date"].astype(str).str.strip(),
    format="%d/%m/%y"
).dt.date

df = df.merge(policy.iloc[:, 1:], left_index=True, right_index=True, how='left')

#%%Clean google trends data
trends = trends.rename(
    columns={
        "Time": "date",
        "Certificate of Entitlement": "gtrends_coe",
        "sgcarmart": "gtrends_sgcarmart",
    }
)

trends["date"] = pd.to_datetime(
    trends["date"].astype(str).str.strip(),
    format="%d/%m/%y"
).dt.date

trends = trends.sort_values(by="date", ascending=False).reset_index(drop=True)

#remove 2020-04-01, 2020-05-01 and 2020-06-01 as bidding paused during covid
trends = trends[~trends['date'].isin([pd.to_datetime('2020-04-01').date(), pd.to_datetime('2020-05-01').date(), pd.to_datetime('2020-06-01').date()])].reset_index(drop=True)

trends_biweekly = trends.loc[trends.index.repeat(2)].reset_index(drop=True)
df = df.merge(trends_biweekly[['gtrends_coe', 'gtrends_sgcarmart']], left_index=True, right_index=True, how='left')



#clean deregistrations data
deregistrations = deregistrations.iloc[8:].reset_index(drop=True)
deregistrations.columns = deregistrations.iloc[0]
deregistrations = deregistrations.drop(0).reset_index(drop=True)
deregistrations = deregistrations.iloc[:420].reset_index(drop=True) #keep data from 1991 jan onwards
deregistrations.columns = ['date', 'deregis_total', 'deregis_cat_a']
deregistrations.drop(columns=['deregis_total'], inplace=True)

deregistrations["date"] = pd.to_datetime(
    deregistrations["date"].str.strip(),
    format="%Y %b"
).dt.date

#remove 2020-04-01, 2020-05-01 and 2020-06-01 as bidding paused during covid
deregistrations = deregistrations[~deregistrations['date'].isin([pd.to_datetime('2020-04-01').date(), pd.to_datetime('2020-05-01').date(), pd.to_datetime('2020-06-01').date()])].reset_index(drop=True)

deregistrations = add_growth_rate_columns(deregistrations, ['deregis_cat_a'], drop_original=False)

deregistrations_biweekly = deregistrations.loc[deregistrations.index.repeat(2)].reset_index(drop=True)
deregistrations_biweekly.index = deregistrations_biweekly.index + 2   # start filling from 3rd row (index 2)

df = (
    df.reset_index(drop=True)
      .merge(deregistrations_biweekly[['deregis_cat_a','deregis_cat_a_growth_rate']], left_index=True, right_index=True, how="outer")
      .sort_index()
      .reset_index(drop=True)
)

#%% save the cleaned dataset
df.to_csv(REPO_ROOT / 'dataset_building/cleaned_dataset.csv', index=False)
