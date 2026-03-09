import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# 1. Data Collection
# Using the Housing Prices Dataset by Yasser H (downloaded as Housing.csv)
df = pd.read_csv("Housing.csv")

# 2. Data Exploration and Cleaning
print("Dataset Head:")
print(df.head())

# Check for missing values
print("\nMissing Values Check:")
print(df.isnull().sum())

# Visualize the distribution of the target variable 'price'
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True, color='blue')
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.savefig('price_distribution.png')
plt.close()

# Explore relationships between categorical variables and price
categorical_cols_raw = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
plt.figure(figsize=(15, 12))
for i, col in enumerate(categorical_cols_raw):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x=col, y='price', data=df)
    plt.title(f'Price vs {col}')
plt.tight_layout()
plt.savefig('categorical_boxplots.png')
plt.close()

# 3. Data Preprocessing (Feature Selection & Encoding)
# Encoding binary categorical features
le = LabelEncoder()
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-hot encoding for multi-level categorical feature 'furnishingstatus'
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# 4. Correlation Analysis
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.savefig("correlation_heatmap.png")
plt.close()

# 5. Feature Selection
# Using all relevant features for the predictive model
X = df.drop('price', axis=1)
y = df['price']

# 6. Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# 7. Model Evaluation
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation Metrics:")
print(f"Mean Squared Error: {mse:,.2f}")
print(f"Root Mean Squared Error: {rmse:,.2f}")
print(f"R-squared: {r2:.4f}")

# 8. Visualizing Actual vs Predicted
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices (with Regression Line)")
plt.savefig("actual_vs_predicted_refined.png")
plt.close()

# Residual Analysis
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='purple')
plt.title("Distribution of Residuals")
plt.xlabel("Residual (Actual - Predicted)")
plt.savefig("residuals_distribution.png")
plt.close()

print("\nAll visualizations have been saved in the project directory.")
