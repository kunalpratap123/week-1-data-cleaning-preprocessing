import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FILE = Path("AB_NYC_2019.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)
raw_df = df.copy()

print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe(include="all").T)
print("\nMissing values:\n", df.isna().sum().sort_values(ascending=False))
print("\nDuplicate rows:", df.duplicated().sum())

df.columns = (
    df.columns.str.strip().str.lower()
      .str.replace(" ", "_", regex=False)
)

text_cols = [
    "name", "host_name", "neighbourhood_group",
    "neighbourhood", "room_type"
]
for col in text_cols:
    df[col] = df[col].astype("string").str.strip()

df = df.replace(r"^\s*$", np.nan, regex=True)

df["last_review"] = pd.to_datetime(
    df["last_review"], errors="coerce"
)

numeric_cols = [
    "price", "minimum_nights", "number_of_reviews",
    "reviews_per_month", "calculated_host_listings_count",
    "availability_365", "latitude", "longitude"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.drop_duplicates()

df.loc[df["price"] < 0, "price"] = np.nan
df.loc[df["minimum_nights"] < 0, "minimum_nights"] = np.nan
df.loc[~df["latitude"].between(40.45, 40.95), "latitude"] = np.nan
df.loc[~df["longitude"].between(-74.30, -73.65), "longitude"] = np.nan

df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

def iqr_bounds(series):
    q1 = series.quantile(.25)
    q3 = series.quantile(.75)
    iqr = q3 - q1
    return q1, q3, q1 - 1.5*iqr, q3 + 1.5*iqr

outlier_rows=[]
for col in ["price","minimum_nights","availability_365"]:
    q1,q3,lower,upper=iqr_bounds(df[col].dropna())
    mask=(df[col]<lower)|(df[col]>upper)
    outlier_rows.append(
        [col,q1,q3,lower,upper,int(mask.sum())]
    )

outliers=pd.DataFrame(
    outlier_rows,
    columns=[
        "column","Q1","Q3","lower_bound",
        "upper_bound","outlier_count"
    ]
)
outliers.to_csv(
    OUTPUT_DIR/"outlier_summary.csv",index=False
)

price_cap=df["price"].quantile(.99)
df["price_capped_99"]=df["price"].clip(
    upper=price_cap
)
df["log_price"]=np.log1p(df["price"])

_,_,_,min_nights_upper=iqr_bounds(
    df["minimum_nights"].dropna()
)
df["minimum_nights_outlier"] = (
    df["minimum_nights"] > min_nights_upper
)

df["has_reviews"] = (
    df["number_of_reviews"] > 0
).astype(int)

model_df=pd.get_dummies(
    df,
    columns=["neighbourhood_group","room_type"],
    drop_first=True
)

quality=pd.DataFrame({
    "metric":[
        "rows","columns",
        "missing_cells","duplicate_rows"
    ],
    "before":[
        raw_df.shape[0],raw_df.shape[1],
        raw_df.isna().sum().sum(),
        raw_df.duplicated().sum()
    ],
    "after":[
        df.shape[0],df.shape[1],
        df.isna().sum().sum(),
        df.duplicated().sum()
    ]
})
print("\nQUALITY COMPARISON\n",quality)
quality.to_csv(
    OUTPUT_DIR/"quality_comparison.csv",
    index=False
)

df.to_csv(
    OUTPUT_DIR/"AB_NYC_2019_cleaned.csv",
    index=False
)
model_df.to_csv(
    OUTPUT_DIR/"AB_NYC_2019_model_ready.csv",
    index=False
)

missing=df.isna().sum().sort_values(ascending=False)
missing=missing[missing>0]
plt.figure(figsize=(9,5))
missing.plot(kind="bar")
plt.title("Remaining Missing Values After Cleaning")
plt.ylabel("Number of missing values")
plt.xticks(rotation=45,ha="right")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR/"missing_values_after_cleaning.png",
    dpi=180
)
plt.close()

print("\nCleaning completed.")
