# ============================================================
# USED CAR PRICE PREDICTION - EXPLORATORY DATA ANALYSIS
# ============================================================

# -----------------------------
# 1. Import Libraries
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

# Seaborn style
sns.set_theme(style="whitegrid")


# -----------------------------
# 2. Load Dataset
# -----------------------------

df = pd.read_csv("Used_Car_Price_Prediction.csv")

print("=" * 60)
print("USED CAR PRICE DATASET - EDA")
print("=" * 60)


# -----------------------------
# 3. Basic Information
# -----------------------------

print("\nShape of Dataset:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()


# -----------------------------
# 4. Statistical Summary
# -----------------------------

print("\nStatistical Summary:")
print(df.describe())

print("\nStatistical Summary Including Categorical Columns:")
print(df.describe(include="all"))


# -----------------------------
# 5. Check Missing Values
# -----------------------------

print("\nMissing Values:")
missing = df.isnull().sum().sort_values(ascending=False)

print(missing)

print("\nMissing Value Percentage:")
missing_percentage = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)

print(missing_percentage)


# Missing value visualization
plt.figure(figsize=(12, 6))

missing_percentage[missing_percentage > 0].plot(kind="bar")

plt.title("Missing Values Percentage")
plt.xlabel("Columns")
plt.ylabel("Missing Percentage (%)")
plt.xticks(rotation=75)
plt.tight_layout()
plt.show()


# -----------------------------
# 6. Duplicate Values
# -----------------------------

print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())


# -----------------------------
# 7. Data Types
# -----------------------------

print("\nData Types:")
print(df.dtypes)


# Count data types
print("\nData Type Count:")
print(df.dtypes.value_counts())


# ============================================================
# NUMERICAL DATA ANALYSIS
# ============================================================

# -----------------------------
# 8. Numerical Columns
# -----------------------------

numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

print("\nNumerical Columns:")
print(numerical_columns.tolist())


# -----------------------------
# 9. Numerical Distributions
# -----------------------------

df[numerical_columns].hist(
    figsize=(16, 12),
    bins=30
)

plt.suptitle("Distribution of Numerical Variables", fontsize=16)

plt.tight_layout()
plt.show()


# -----------------------------
# 10. Boxplots for Numerical Data
# -----------------------------

for column in numerical_columns:

    plt.figure(figsize=(10, 4))

    sns.boxplot(x=df[column])

    plt.title(f"Boxplot of {column}")

    plt.tight_layout()
    plt.show()


# ============================================================
# SALE PRICE ANALYSIS
# ============================================================

# -----------------------------
# 11. Sale Price Distribution
# -----------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    df["sale_price"],
    bins=50,
    kde=True
)

plt.title("Distribution of Used Car Sale Prices")
plt.xlabel("Sale Price")
plt.ylabel("Number of Cars")

plt.tight_layout()
plt.show()


# -----------------------------
# 12. Sale Price Boxplot
# -----------------------------

plt.figure(figsize=(10, 5))

sns.boxplot(
    x=df["sale_price"]
)

plt.title("Sale Price Boxplot")
plt.xlabel("Sale Price")

plt.tight_layout()
plt.show()


# -----------------------------
# 13. Sale Price Statistics
# -----------------------------

print("\nSale Price Statistics:")

print("Minimum Price:",
      df["sale_price"].min())

print("Maximum Price:",
      df["sale_price"].max())

print("Average Price:",
      df["sale_price"].mean())

print("Median Price:",
      df["sale_price"].median())


# ============================================================
# CATEGORICAL DATA ANALYSIS
# ============================================================

categorical_columns = df.select_dtypes(
    include="object"
).columns

print("\nCategorical Columns:")
print(categorical_columns.tolist())


# -----------------------------
# 14. Fuel Type Distribution
# -----------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="fuel_type"
)

plt.title("Number of Cars by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Number of Cars")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()


# -----------------------------
# 15. Transmission Distribution
# -----------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="transmission"
)

plt.title("Manual vs Automatic Cars")
plt.xlabel("Transmission")
plt.ylabel("Number of Cars")

plt.tight_layout()
plt.show()


# -----------------------------
# 16. Body Type Distribution
# -----------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="body_type"
)

plt.title("Cars by Body Type")
plt.xlabel("Body Type")
plt.ylabel("Number of Cars")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()


# -----------------------------
# 17. Cars by Make
# -----------------------------

plt.figure(figsize=(12, 6))

top_makes = df["make"].value_counts().head(15)

sns.barplot(
    x=top_makes.values,
    y=top_makes.index
)

plt.title("Top 15 Car Manufacturers")
plt.xlabel("Number of Cars")
plt.ylabel("Manufacturer")

plt.tight_layout()
plt.show()


# -----------------------------
# 18. Cars by City
# -----------------------------

plt.figure(figsize=(12, 6))

city_count = df["city"].value_counts()

sns.barplot(
    x=city_count.values,
    y=city_count.index
)

plt.title("Number of Cars by City")
plt.xlabel("Number of Cars")
plt.ylabel("City")

plt.tight_layout()
plt.show()


# ============================================================
# PRICE COMPARISONS
# ============================================================

# -----------------------------
# 19. Average Price by Fuel Type
# -----------------------------

fuel_price = (
    df.groupby("fuel_type")["sale_price"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Price by Fuel Type:")
print(fuel_price)

plt.figure(figsize=(9, 5))

sns.barplot(
    x=fuel_price.index,
    y=fuel_price.values
)

plt.title("Average Car Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Average Sale Price")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()


# -----------------------------
# 20. Average Price by Transmission
# -----------------------------

transmission_price = (
    df.groupby("transmission")["sale_price"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Price by Transmission:")
print(transmission_price)

plt.figure(figsize=(7, 5))

sns.barplot(
    x=transmission_price.index,
    y=transmission_price.values
)

plt.title("Average Price: Manual vs Automatic")
plt.xlabel("Transmission")
plt.ylabel("Average Sale Price")

plt.tight_layout()
plt.show()


# -----------------------------
# 21. Average Price by Body Type
# -----------------------------

body_price = (
    df.groupby("body_type")["sale_price"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Price by Body Type:")
print(body_price)

plt.figure(figsize=(9, 5))

sns.barplot(
    x=body_price.index,
    y=body_price.values
)

plt.title("Average Price by Body Type")
plt.xlabel("Body Type")
plt.ylabel("Average Sale Price")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()


# -----------------------------
# 22. Average Price by Manufacturer
# -----------------------------

make_price = (
    df.groupby("make")["sale_price"]
    .mean()
    .sort_values(ascending=False)
    .head(15)
)

print("\nTop Manufacturers by Average Price:")
print(make_price)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=make_price.values,
    y=make_price.index
)

plt.title("Top 15 Manufacturers by Average Sale Price")
plt.xlabel("Average Sale Price")
plt.ylabel("Manufacturer")

plt.tight_layout()
plt.show()


# -----------------------------
# 23. Average Price by City
# -----------------------------

city_price = (
    df.groupby("city")["sale_price"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Price by City:")
print(city_price)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=city_price.values,
    y=city_price.index
)

plt.title("Average Used Car Price by City")
plt.xlabel("Average Sale Price")
plt.ylabel("City")

plt.tight_layout()
plt.show()


# ============================================================
# RELATIONSHIP ANALYSIS
# ============================================================

# -----------------------------
# 24. Car Age vs Sale Price
# -----------------------------

current_year = 2021

df["car_age"] = current_year - df["yr_mfr"]

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="car_age",
    y="sale_price",
    alpha=0.5
)

plt.title("Car Age vs Sale Price")
plt.xlabel("Car Age")
plt.ylabel("Sale Price")

plt.tight_layout()
plt.show()


# -----------------------------
# 25. Kilometers Driven vs Price
# -----------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="kms_run",
    y="sale_price",
    alpha=0.5
)

plt.title("Kilometers Driven vs Sale Price")
plt.xlabel("Kilometers Driven")
plt.ylabel("Sale Price")

plt.tight_layout()
plt.show()


# -----------------------------
# 26. Views vs Sale Price
# -----------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="times_viewed",
    y="sale_price",
    alpha=0.5
)

plt.title("Times Viewed vs Sale Price")
plt.xlabel("Times Viewed")
plt.ylabel("Sale Price")

plt.tight_layout()
plt.show()


# -----------------------------
# 27. Broker Quote vs Sale Price
# -----------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="broker_quote",
    y="sale_price",
    alpha=0.5
)

plt.title("Broker Quote vs Sale Price")
plt.xlabel("Broker Quote")
plt.ylabel("Sale Price")

plt.tight_layout()
plt.show()


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

# -----------------------------
# 28. Correlation Matrix
# -----------------------------

numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)

correlation = numeric_df.corr()

print("\nCorrelation with Sale Price:")

print(
    correlation["sale_price"]
    .sort_values(ascending=False)
)


# -----------------------------
# 29. Correlation Heatmap
# -----------------------------

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# ============================================================
# TOP / BOTTOM CARS
# ============================================================

# -----------------------------
# 30. Most Expensive Cars
# -----------------------------

print("\nTop 10 Most Expensive Cars:")

print(
    df[
        [
            "car_name",
            "make",
            "model",
            "yr_mfr",
            "kms_run",
            "sale_price"
        ]
    ]
    .sort_values(
        by="sale_price",
        ascending=False
    )
    .head(10)
)


# -----------------------------
# 31. Cheapest Cars
# -----------------------------

print("\nTop 10 Cheapest Cars:")

print(
    df[
        [
            "car_name",
            "make",
            "model",
            "yr_mfr",
            "kms_run",
            "sale_price"
        ]
    ]
    .sort_values(
        by="sale_price"
    )
    .head(10)
)


# -----------------------------
# 32. Most Viewed Cars
# -----------------------------

print("\nTop 10 Most Viewed Cars:")

print(
    df[
        [
            "car_name",
            "make",
            "model",
            "times_viewed",
            "sale_price"
        ]
    ]
    .sort_values(
        by="times_viewed",
        ascending=False
    )
    .head(10)
)


# ============================================================
# OWNERSHIP ANALYSIS
# ============================================================

# -----------------------------
# 33. Price by Number of Owners
# -----------------------------

owner_price = (
    df.groupby("total_owners")["sale_price"]
    .mean()
)

print("\nAverage Price by Number of Owners:")
print(owner_price)

plt.figure(figsize=(8, 5))

sns.barplot(
    x=owner_price.index,
    y=owner_price.values
)

plt.title("Average Car Price by Number of Owners")
plt.xlabel("Number of Owners")
plt.ylabel("Average Sale Price")

plt.tight_layout()
plt.show()


# ============================================================
# WARRANTY / ASSURED BUY / RESERVED ANALYSIS
# ============================================================

# -----------------------------
# 34. Warranty vs Price
# -----------------------------

warranty_price = (
    df.groupby("warranty_avail")["sale_price"]
    .mean()
)

print("\nAverage Price by Warranty Availability:")
print(warranty_price)

plt.figure(figsize=(7, 5))

sns.barplot(
    x=warranty_price.index,
    y=warranty_price.values
)

plt.title("Warranty Availability vs Average Price")
plt.xlabel("Warranty Available")
plt.ylabel("Average Sale Price")

plt.tight_layout()
plt.show()


# -----------------------------
# 35. Assured Buy vs Price
# -----------------------------

assured_price = (
    df.groupby("assured_buy")["sale_price"]
    .mean()
)

print("\nAverage Price by Assured Buy:")
print(assured_price)


# ============================================================
# OUTLIER DETECTION
# ============================================================

# -----------------------------
# 36. IQR Outlier Detection
# -----------------------------

def find_outliers(column):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower) |
        (df[column] > upper)
    ]

    print(f"\n{column}")
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower Limit:", lower)
    print("Upper Limit:", upper)
    print("Number of Outliers:", len(outliers))


find_outliers("sale_price")
find_outliers("kms_run")
find_outliers("times_viewed")
find_outliers("broker_quote")


# ============================================================
# FINAL EDA SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nDataset Shape:", df.shape)

print("Number of Numerical Columns:",
      len(numerical_columns))

print("Number of Categorical Columns:",
      len(categorical_columns))

print("Total Missing Values:",
      df.isnull().sum().sum())

print("Total Duplicate Rows:",
      df.duplicated().sum())

print("\nAverage Sale Price:",
      round(df["sale_price"].mean(), 2))

print("Median Sale Price:",
      df["sale_price"].median())

print("Minimum Sale Price:",
      df["sale_price"].min())

print("Maximum Sale Price:",
      df["sale_price"].max())

print("\nEDA Finished!")