
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# --- Configuration --- #
RANDOM_SEED = 42
OUTPUT_DIR = "./reports"

# --- 1. Data Generation --- #
def generate_synthetic_sales_data(num_records=1000):
    """Generates synthetic sales data for demonstration."""
    np.random.seed(RANDOM_SEED)
    data = {
        'date': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.arange(num_records), unit='D'),
        'region': np.random.choice(['North', 'South', 'East', 'West'], num_records),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Home Goods', 'Books'], num_records),
        'price': np.random.uniform(10, 500, num_records).round(2),
        'quantity': np.random.randint(1, 20, num_records),
        'promotion': np.random.choice([0, 1], num_records, p=[0.7, 0.3]) # 0: no promo, 1: promo
    }
    df = pd.DataFrame(data)
    df['total_sales'] = df['price'] * df['quantity']
    
    # Introduce some trends/relationships
    df.loc[df['product_category'] == 'Electronics', 'total_sales'] *= 1.2
    df.loc[df['promotion'] == 1, 'total_sales'] *= 1.15
    
    return df

# --- 2. Exploratory Data Analysis (EDA) --- #
def perform_eda(df, output_dir):
    """Performs basic EDA and generates visualizations."""
    print("\n--- Performing Exploratory Data Analysis ---")
    os.makedirs(output_dir, exist_ok=True)

    # Basic statistics
    print("Dataset Info:")
    df.info()
    print("\nDescriptive Statistics:")
    print(df.describe())

    # Sales over time
    plt.figure(figsize=(12, 6))
    df.set_index('date')['total_sales'].resample('M').sum().plot()
    plt.title('Monthly Total Sales')
    plt.ylabel('Total Sales')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'monthly_sales.png'))
    plt.close()
    print(f"Saved monthly sales plot to {os.path.join(output_dir, 'monthly_sales.png')}")

    # Sales by product category
    plt.figure(figsize=(10, 6))
    sns.barplot(x='product_category', y='total_sales', data=df.groupby('product_category')['total_sales'].sum().reset_index())
    plt.title('Total Sales by Product Category')
    plt.ylabel('Total Sales')
    plt.savefig(os.path.join(output_dir, 'sales_by_category.png'))
    plt.close()
    print(f"Saved sales by category plot to {os.path.join(output_dir, 'sales_by_category.png')}")

    # Sales by region
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='region', y='total_sales', data=df)
    plt.title('Total Sales Distribution by Region')
    plt.ylabel('Total Sales')
    plt.savefig(os.path.join(output_dir, 'sales_by_region.png'))
    plt.close()
    print(f"Saved sales by region plot to {os.path.join(output_dir, 'sales_by_region.png')}")

# --- 3. Predictive Modeling --- #
def perform_predictive_modeling(df):
    """Builds a simple linear regression model to predict sales."""
    print("\n--- Performing Predictive Modeling ---")
    
    # Feature Engineering
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Convert categorical features to numerical using one-hot encoding
    df_encoded = pd.get_dummies(df, columns=['region', 'product_category'], drop_first=True)
    
    features = [col for col in df_encoded.columns if col not in ['date', 'price', 'quantity', 'total_sales']]
    X = df_encoded[features]
    y = df_encoded['total_sales']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    
    return model

# --- Main Execution --- #
if __name__ == "__main__":
    print("Starting AI Data Analytics Platform pipeline...")
    sales_df = generate_synthetic_sales_data(num_records=1500)
    
    perform_eda(sales_df.copy(), OUTPUT_DIR)
    
    predictive_model = perform_predictive_modeling(sales_df.copy())
    print("AI Data Analytics Platform pipeline completed.")
