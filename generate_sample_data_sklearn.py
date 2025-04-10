import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of samples
n_samples = 1000

# Generate classification data
X_class, y_class = make_classification(
    n_samples=n_samples,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_repeated=0,
    n_classes=3,
    weights=[0.6, 0.3, 0.1],
    random_state=42
)

# Generate regression data
X_reg, y_reg = make_regression(
    n_samples=n_samples,
    n_features=10,
    n_informative=6,
    noise=20,
    random_state=42
)

# Create a DataFrame
df = pd.DataFrame(
    np.hstack([X_class, X_reg[:, :5]]),  # Use 5 features from regression dataset
    columns=[f'feature_{i}' for i in range(15)]  # 15 numerical features
)

# Scale the regression target to a reasonable range (e.g., price in dollars)
y_reg_scaled = 10000 + (y_reg - y_reg.min()) * 2000 / (y_reg.max() - y_reg.min())

# Add classification target (customer type)
customer_types = np.array(['standard', 'premium', 'enterprise'])[y_class]
df['customer_type'] = customer_types

# Add regression target (sales_amount)
df['sales_amount'] = y_reg_scaled

# Add a binary classification target (churn)
churn_prob = 0.2 + 0.4 * (df['feature_0'] - df['feature_0'].min()) / (df['feature_0'].max() - df['feature_0'].min())
df['churn'] = np.random.binomial(1, churn_prob)
df['churn'] = df['churn'].map({0: 'no', 1: 'yes'})

# Add categorical features
# Product category
categories = ['Electronics', 'Furniture', 'Clothing', 'Books', 'Software']
df['product_category'] = np.random.choice(categories, size=n_samples, p=[0.3, 0.2, 0.2, 0.15, 0.15])

# Region
regions = ['North', 'South', 'East', 'West', 'Central']
df['region'] = np.random.choice(regions, size=n_samples)

# Payment method
payment_methods = ['Credit Card', 'PayPal', 'Bank Transfer', 'Cash']
df['payment_method'] = np.random.choice(payment_methods, size=n_samples, p=[0.5, 0.3, 0.15, 0.05])

# Add date features
start_date = datetime(2023, 1, 1)
# Convert numpy.int64 to Python int to avoid the error with timedelta
days = [int(day) for day in np.random.randint(0, 365, size=n_samples)]
df['purchase_date'] = [start_date + timedelta(days=day) for day in days]
df['purchase_month'] = df['purchase_date'].dt.month
df['purchase_quarter'] = df['purchase_date'].dt.quarter
df['purchase_day_of_week'] = df['purchase_date'].dt.dayofweek

# Add a customer ID column
df['customer_id'] = [f'CUST-{i+1000:05d}' for i in range(n_samples)]

# Add some missing values
for col in df.columns:
    if col not in ['customer_id', 'customer_type', 'churn', 'sales_amount']:  # Don't add missing values to key columns
        missing_mask = np.random.random(size=n_samples) < 0.05  # 5% missing values
        df.loc[missing_mask, col] = np.nan

# Reorder columns to have IDs first, features in the middle, and targets at the end
column_order = ['customer_id', 'purchase_date', 'purchase_month', 'purchase_quarter', 
                'purchase_day_of_week', 'product_category', 'region', 'payment_method']

for i in range(15):
    column_order.append(f'feature_{i}')

column_order.extend(['customer_type', 'churn', 'sales_amount'])

df = df[column_order]

# Save to CSV
df.to_csv('customer_sales_data.csv', index=False)

print(f"Created dataset with {n_samples} samples and {len(df.columns)} columns")
print(f"Classification targets: 'customer_type' (3 classes) and 'churn' (binary)")
print(f"Regression target: 'sales_amount'")
print("Dataset saved to 'customer_sales_data.csv'")

# Print a sample and summary
print("\nData sample:")
print(df.head(3))

print("\nData summary:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nTarget distributions:")
print("'customer_type' distribution:")
print(df['customer_type'].value_counts())
print("\n'churn' distribution:")
print(df['churn'].value_counts())
print("\n'sales_amount' statistics:")
print(df['sales_amount'].describe())
