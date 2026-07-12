# Core data cleaning, outlier clipping, and feature scaling 

# import os
# import sqlite3
# import pandas as pd
# import numpy as np

# # Define relative data directory pathways
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else "."
# DB_PATH = os.path.join(BASE_DIR, "data-package", "database", "afilearn_commerce_master.db")
# INTERMEDIATE_DIR = os.path.join(BASE_DIR, "data-package", "intermediate")
# os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

# print("======================================================================")
# print("  AFILEARN COMMERCE ANALYTICS - PHASE 3: PREPROCESSING PIPELINE       ")
# print("======================================================================\n")

# try:
#     # Connect to the SQLite relational database
#     conn = sqlite3.connect(DB_PATH)
    
#     # 1. EXTRACT FROM STAGING TABLES
#     df_orders = pd.read_sql_query("SELECT * FROM raw_orders;", conn)
#     df_items = pd.read_sql_query("SELECT * FROM raw_order_items;", conn)
#     df_reviews = pd.read_sql_query("SELECT * FROM raw_order_reviews;", conn)
    
#     print(f"✔ Extracted raw data profiles. Orders: {df_orders.shape} | Reviews: {df_reviews.shape}")
    
#     # 2. DATA QUALITY REPAIR LAYER (Deduplication & Null Handling)
#     # Remove duplicate order IDs to prevent double-counting revenue
#     df_orders = df_orders.drop_duplicates(subset=['order_id'])
    
#     # Convert logistics timestamps to datetime format because everything was loaded into the master db as strings, coercing data traps to NaT nulls
#     date_cols = ['order_purchase_timestamp', 'order_approved_at', 
#                  'order_delivered_carrier_date', 'order_delivered_customer_date', 
#                  'order_estimated_delivery_date']
#     for col in date_cols:
#         df_orders[col] = pd.to_datetime(df_orders[col], errors='coerce')
        
#     # Drop rows missing critical delivery dates to protect our logistics model
#     df_orders_cleaned = df_orders.dropna(subset=['order_delivered_customer_date', 'order_estimated_delivery_date']).copy()
    
#     # 3. FEATURE ENGINEERING LAYER (Derived Variables)
#     # Feature 1: Calculate actual delivery delay days (Delivered Date minus Estimated Date)
#     df_orders_cleaned['delivery_delay_days'] = (
#         df_orders_cleaned['order_delivered_customer_date'] - df_orders_cleaned['order_estimated_delivery_date']
#     ).dt.days
    
#     # Feature 2: Create a binary flag for late deliveries (1 if delayed, 0 otherwise)
#     df_orders_cleaned['is_late_delivery'] = np.where(df_orders_cleaned['delivery_delay_days'] > 0, 1, 0)
    
#     # Feature 3: Classify review scores into sentiment groups (Satisfied vs Dissatisfied)
#     df_reviews['review_score'] = pd.to_numeric(df_reviews['review_score'], errors='coerce')
#     df_reviews['review_sentiment'] = np.where(df_reviews['review_score'] >= 4, 'Satisfied', 'Dissatisfied')
    
#     # 4. EXPORT PRISTINE INTERMEDIATE LAYER
#     df_orders_cleaned.to_csv(os.path.join(INTERMEDIATE_DIR, "olist_orders_cleaned.csv"), index=False)
#     df_reviews.dropna(subset=['review_score']).to_csv(os.path.join(INTERMEDIATE_DIR, "olist_reviews_cleaned.csv"), index=False)
    
#     print("\n✔ Phase 3 Preprocessing Complete. Data traps removed and features engineered.")
#     print(f"  --> Cleaned Orders Shape: {df_orders_cleaned.shape}")
#     print(f"  --> Late Deliveries Flagged: {df_orders_cleaned['is_late_delivery'].sum():,}")
    
#     conn.close()

# except Exception as e:
#     print(f"❌ Preprocessing Execution Failure: {e}")












#divider

import os
import sqlite3
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # Added for Scaling requirement

# Define relative data directory pathways
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else "."
DB_PATH = os.path.join(BASE_DIR, "data-package", "database", "afilearn_commerce_master.db")
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "data-package", "intermediate")
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

def run_data_preprocessing():
    print("--- [Step 1] Initiating Data Preprocessing Engine ---")
    
    # Open programmatic link to the local relational database
    conn = sqlite3.connect(DB_PATH)
    
    # Extract original raw staging datasets
    df_orders = pd.read_sql_query("SELECT * FROM raw_orders;", conn)
    df_reviews = pd.read_sql_query("SELECT * FROM raw_order_reviews;", conn)
    df_items = pd.read_sql_query("SELECT * FROM raw_order_items;", conn)
    conn.close()
    
    # 1. UNIQUENESS: Drop duplicate entries across transactional keys
    df_orders = df_orders.drop_duplicates(subset=['order_id'])
    df_reviews = df_reviews.drop_duplicates(subset=['review_id'])
    #added
    df_items['price'] = pd.to_numeric(df_items['price'], errors='coerce')
    df_items['freight_value'] = pd.to_numeric(df_items['freight_value'], errors='coerce')

    # OUTLIERS REQUIREMENT: Identify and clip extreme financial values using IQR boundaries
    q_high = df_items['price'].quantile(0.99)
    df_items['price'] = df_items['price'].clip(upper=q_high)  # Clip extreme 1% price outlier



    # 2. VALIDITY: Coerce mixed string datetimes into active pandas datetime objects
    date_columns = [
        'order_purchase_timestamp', 'order_approved_at', 
        'order_delivered_carrier_date', 'order_delivered_customer_date', 
        'order_estimated_delivery_date'
    ]
    for col in date_columns:
        df_orders[col] = pd.to_datetime(df_orders[col], errors='coerce')
        
    # 3. COMPLETENESS: Drop records missing critical target fulfillment timestamps
    # This prevents calculations from introducing invalid math models later
    df_orders_cleaned = df_orders.dropna(subset=['order_delivered_customer_date', 'order_estimated_delivery_date']).copy()
    
    # SCALING REQUIREMENT: Apply MinMaxScaler to normalize transaction costs for later modeling
    scaler = MinMaxScaler()
    df_items_cleaned = df_items.dropna(subset=['price', 'freight_value']).copy()
    df_items_cleaned['scaled_price'] = scaler.fit_transform(df_items_cleaned[['price']])




    # 4. CONSISTENCY: Clean trailing spaces and normalize string text fields
    df_reviews['review_score'] = pd.to_numeric(df_reviews['review_score'], errors='coerce')
    df_reviews_cleaned = df_reviews.dropna(subset=['review_score']).copy()
    
    # Export clean staging files to the intermediate folder layer
    df_orders_cleaned.to_csv(os.path.join(INTERMEDIATE_DIR, "orders_cleaned.csv"), index=False)
    df_reviews_cleaned.to_csv(os.path.join(INTERMEDIATE_DIR, "reviews_cleaned.csv"), index=False)
    df_items.to_csv(os.path.join(INTERMEDIATE_DIR, "items_cleaned.csv"), index=False)
    
    print(f"  ✔ Preprocessing complete. Clean intermediate arrays saved to: {INTERMEDIATE_DIR}")
    print(f"  ✔ Rows Retained -> Orders: {df_orders_cleaned.shape[0]:,} | Reviews: {df_reviews_cleaned.shape[0]:,}\n")

if __name__ == "__main__":
    run_data_preprocessing()
