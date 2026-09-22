"""
Generate a synthetic E-Commerce Customer Churn dataset for the
AICTE | IBM SkillsBuild Data Analytics with AI Internship Project.

The dataset mimics a realistic online-retail customer base, with
behavioral, transactional and demographic features that plausibly
drive churn, and a realistic amount of noise / missingness so that
the EDA + preprocessing steps in the notebook are meaningful.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
N = 5000

customer_id = [f"CUST{100000+i}" for i in range(N)]

gender = rng.choice(["Male", "Female"], size=N, p=[0.52, 0.48])
age = rng.integers(18, 70, size=N)

# Tenure (months as a customer)
tenure_months = rng.integers(0, 72, size=N)

# City tier (1 = metro, 2 = mid-size, 3 = small town)
city_tier = rng.choice([1, 2, 3], size=N, p=[0.4, 0.35, 0.25])

marital_status = rng.choice(["Single", "Married", "Divorced"], size=N, p=[0.45, 0.45, 0.10])

preferred_login_device = rng.choice(["Mobile Phone", "Computer"], size=N, p=[0.65, 0.35])
preferred_payment_mode = rng.choice(
    ["Credit Card", "Debit Card", "UPI", "Cash on Delivery", "E-wallet"],
    size=N, p=[0.28, 0.24, 0.24, 0.14, 0.10]
)
preferred_order_cat = rng.choice(
    ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"],
    size=N, p=[0.22, 0.28, 0.25, 0.15, 0.10]
)

satisfaction_score = rng.integers(1, 6, size=N)  # 1 (low) - 5 (high)
number_of_devices_registered = rng.integers(1, 6, size=N)
number_of_addresses = rng.integers(1, 8, size=N)
complain_raised = rng.choice([0, 1], size=N, p=[0.72, 0.28])

hours_spent_on_app = np.round(rng.gamma(shape=2.0, scale=1.3, size=N), 2)
num_orders_last_month = rng.poisson(lam=3.2, size=N)
days_since_last_order = rng.integers(0, 90, size=N)
avg_cashback = np.round(rng.normal(loc=150, scale=60, size=N).clip(0), 2)
order_amount_hike_pct = np.round(rng.normal(loc=15, scale=8, size=N).clip(0), 2)
coupon_used_last_month = rng.poisson(lam=1.5, size=N)
distance_from_warehouse_km = np.round(rng.normal(loc=15, scale=8, size=N).clip(1), 1)

# ---- Build churn probability from a logistic combination of features ----
# (This creates realistic, learnable signal for the ML model.)
z = (
    -1.4
    + 0.55 * complain_raised
    + 0.35 * (satisfaction_score <= 2).astype(int)
    - 0.03 * tenure_months
    + 0.02 * days_since_last_order
    - 0.15 * num_orders_last_month
    - 0.10 * hours_spent_on_app
    + 0.25 * (city_tier == 3).astype(int)
    - 0.01 * avg_cashback / 10
    + 0.20 * (number_of_devices_registered >= 4).astype(int)
    + rng.normal(0, 0.6, size=N)  # noise
)
churn_prob = 1 / (1 + np.exp(-z))
churn = (rng.uniform(0, 1, size=N) < churn_prob).astype(int)

df = pd.DataFrame({
    "CustomerID": customer_id,
    "Gender": gender,
    "Age": age,
    "MaritalStatus": marital_status,
    "CityTier": city_tier,
    "TenureMonths": tenure_months,
    "PreferredLoginDevice": preferred_login_device,
    "PreferredPaymentMode": preferred_payment_mode,
    "PreferredOrderCategory": preferred_order_cat,
    "SatisfactionScore": satisfaction_score,
    "NumberOfDevicesRegistered": number_of_devices_registered,
    "NumberOfAddresses": number_of_addresses,
    "ComplainRaised": complain_raised,
    "HoursSpentOnApp": hours_spent_on_app,
    "NumOrdersLastMonth": num_orders_last_month,
    "DaysSinceLastOrder": days_since_last_order,
    "AvgCashbackINR": avg_cashback,
    "OrderAmountHikeFromLastYearPct": order_amount_hike_pct,
    "CouponsUsedLastMonth": coupon_used_last_month,
    "DistanceFromWarehouseKM": distance_from_warehouse_km,
    "Churn": churn,
})

# ---- Inject realistic missing values (MCAR-ish) for cleaning practice ----
for col, frac in [
    ("HoursSpentOnApp", 0.03),
    ("AvgCashbackINR", 0.025),
    ("OrderAmountHikeFromLastYearPct", 0.02),
    ("DaysSinceLastOrder", 0.015),
    ("MaritalStatus", 0.01),
]:
    idx = rng.choice(N, size=int(N * frac), replace=False)
    df.loc[idx, col] = np.nan

# ---- Inject a handful of duplicate rows (for data-cleaning demo) ----
dup_idx = rng.choice(N, size=15, replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

out_path = "data/ecommerce_customer_churn.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows x {df.shape[1]} columns -> {out_path}")
print(f"Churn rate: {df['Churn'].mean():.3f}")
print(f"Missing values per column:\n{df.isna().sum()[df.isna().sum() > 0]}")
