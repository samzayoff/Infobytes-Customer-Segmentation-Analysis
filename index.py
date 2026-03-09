# Retail Sales EDA Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1. Load Dataset
# ---------------------------
df = pd.read_csv("retail_sales_dataset.csv")

# Rename columns (remove spaces)
df.columns = df.columns.str.replace(" ", "_")

print(df.head())
print(df.info())

# ---------------------------
# 2. Data Cleaning
# ---------------------------
df.drop_duplicates(inplace=True)
df["Date"] = pd.to_datetime(df["Date"])

# ---------------------------
# 3. Descriptive Statistics
# ---------------------------
print("\nDescriptive Statistics:")
print(df.describe())

mean_sales = df["Total_Amount"].mean()
median_sales = df["Total_Amount"].median()
std_sales = df["Total_Amount"].std()

print("Mean Sales:", mean_sales)
print("Median Sales:", median_sales)
print("Standard Deviation:", std_sales)

# ---------------------------
# 4. Time Series Analysis
# ---------------------------
monthly_sales = (
    df.groupby(df["Date"].dt.to_period("M"))["Total_Amount"]
    .sum()
    .to_timestamp()
)

plt.figure()
plt.plot(monthly_sales)
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.show()

# ---------------------------
# 5. Product Category Analysis
# ---------------------------
category_sales = (
    df.groupby("Product_Category")["Total_Amount"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure()
category_sales.plot(kind="bar")
plt.title("Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")
plt.show()

# ---------------------------
# 6. Customer Analysis
# ---------------------------
customer_sales = df.groupby("Customer_ID")["Total_Amount"].sum()

plt.figure()
sns.histplot(customer_sales, bins=20)
plt.title("Customer Spending Distribution")
plt.xlabel("Total Spending")
plt.show()

# ---------------------------
# 7. Correlation Heatmap
# ---------------------------
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
