import pandas as pd
import numpy as np

# --- Configuration ---
# Define the number of rows for the dataset
n_rows = 10000
file_name = 'user_transaction_data_10k.csv'

# --- Data Generation ---

# 1. user (User Identifier): Unique IDs
user_ids = np.arange(10001, 10001 + n_rows)

# 2. mercheant (Merchant/Product): Categorical data
merchants = ['TechWorld', 'FashionGo', 'GourmetEats', 'BookHive', 'AutoParts']
merchant_data = np.random.choice(merchants, n_rows)

# 3. amount (Amount / Price): Random positive floats
# Uniform distribution between $5.00 and $500.00
amount_data = np.round(np.random.uniform(5.00, 500.00, n_rows), 2)

# 4. country (Country / Region): Categorical data
countries = ['USA', 'Canada', 'UK', 'Germany', 'Australia', 'Brazil']
country_data = np.random.choice(countries, n_rows)

# 5. device (Device / Platform): Categorical data
devices = ['Mobile', 'Desktop', 'Tablet']
device_data = np.random.choice(devices, n_rows)

# 6. label (Label / Target): Binary label (0 or 1)
label_data = np.random.randint(0, 2, n_rows)

# --- Create and Save DataFrame ---

# Create the DataFrame
data = pd.DataFrame({
    'user': user_ids,
    'mercheant': merchant_data,
    'amount': amount_data,
    'country': country_data,
    'device': device_data,
    'label': label_data
})

# Save the DataFrame to a CSV file (index=False prevents writing the pandas index)
data.to_csv(file_name, index=False)

print(f"Successfully generated a DataFrame with {n_rows} rows and saved it to {file_name}")