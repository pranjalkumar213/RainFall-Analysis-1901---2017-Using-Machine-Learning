import sys
sys.path.insert(
    0, r"C:\Users\kpran\AppData\Local\spyder-seaborn"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


file_path = r"C:\Users\kpran\Downloads\Sub_Division_IMD_2017.csv"
df = pd.read_csv(file_path)


df.columns = df.columns.str.strip()
df["SUBDIVISION"] = df["SUBDIVISION"].astype(str).str.strip()


print("Original shape:", df.shape)
print("\nFirst five rows:")
print(df.head())


numeric_cols = [c for c in df.columns if c != "SUBDIVISION"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


df = df.drop_duplicates()
df = df.dropna(subset=["YEAR"])
df["YEAR"] = df["YEAR"].astype(int)


months = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
]


rainfall_cols = months + ["JF", "MAM", "JJAS", "OND", "ANNUAL"]


df[rainfall_cols] = df[rainfall_cols].mask(
    df[rainfall_cols] < 0
)


print("\nShape after removing duplicate rows:", df.shape)

print("\nMissing rainfall values after validation:")
print(df[rainfall_cols].isnull().sum())


print("\nSummary statistics:")
print(df[rainfall_cols].describe())



# OBJECTIVE 1: ANNUAL RAINFALL TREND ANALYSIS
# To visualize changes in average annual rainfall from 1901 to 2017


annual = df.groupby("YEAR")["ANNUAL"].mean()


plt.figure(figsize=(12, 5))
plt.plot(annual.index, annual.values)

plt.title("Average Annual Rainfall Across Subdivisions (1901-2017)")
plt.xlabel("Year")
plt.ylabel("Average Annual Rainfall (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()



# OBJECTIVE 2: MONTHLY RAINFALL ANALYSIS
# To identify months with the highest and lowest average rainfall


monthly_avg = df[months].mean()

print("\nAverage rainfall by month:")
print(monthly_avg)

plt.figure(figsize=(10, 5))
monthly_avg.plot(kind="bar")

plt.title("Average Rainfall by Month")
plt.xlabel("Month")
plt.ylabel("Average Rainfall (mm)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# OBJECTIVE 3: RAINFALL DISTRIBUTION ANALYSIS
# To examine the distribution and variability of annual rainfall


plt.figure(figsize=(9, 5))
sns.histplot(df["ANNUAL"].dropna(), bins=30, kde=True)

plt.title("Distribution of Annual Rainfall")
plt.xlabel("Annual Rainfall (mm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# OBJECTIVE 4: CORRELATION ANALYSIS
# To examine relationships between monthly rainfall and annual rainfall



corr = df[months + ["ANNUAL"]].corr()

print("\nCorrelation matrix:")
print(corr)


plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Between Monthly and Annual Rainfall")
plt.tight_layout()
plt.show()


# OBJECTIVE 5: SEASONAL RAINFALL ANALYSIS
# To compare average rainfall across the four seasons


# Define seasonal rainfall columns:
# JF   = January-February
# MAM  = March-May
# JJAS = June-September
# OND  = October-December

seasons = ["JF", "MAM", "JJAS", "OND"]

seasonal_avg = df[seasons].mean()


print("\nAverage rainfall by season:")
print(seasonal_avg)


plt.figure(figsize=(9, 5))
seasonal_avg.plot(kind="bar")

plt.title("Average Rainfall by Season")
plt.xlabel("Season")
plt.ylabel("Average Rainfall (mm)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# OBJECTIVE 6: SUBDIVISION-WISE RAINFALL COMPARISON
# To identify the ten subdivisions with the highest average
# annual rainfall


top_subdivisions = (
    df.groupby("SUBDIVISION")["ANNUAL"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)


print("\nTop 10 subdivisions by average annual rainfall:")
print(top_subdivisions)


plt.figure(figsize=(12, 6))
top_subdivisions.sort_values().plot(kind="barh")

plt.title("Top 10 Subdivisions by Average Annual Rainfall")
plt.xlabel("Average Annual Rainfall (mm)")
plt.ylabel("Subdivision")
plt.tight_layout()
plt.show()

df.to_csv("rainfall_cleaned.csv", index=False)

print("\nEDA completed successfully.")
print("Cleaned data saved as rainfall_cleaned.csv")
