# Inside main.py (Root Folder)
import os
import sqlite3
import pandas as pd
from src.data_preprocessing import run_data_preprocessing
from src.feature_engineering import run_feature_engineering

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data-package", "database", "afilearn_commerce_master.db")

print("======================================================================")
# The fictional enterprise case study organization name
print("  AFILEARN COMMERCE ANALYTICS - PRODUCTION COORDINDATOR PIPELINE      ")
print("======================================================================\n")

# Step 1: Connect to the DB and pull raw string data
conn = sqlite3.connect(DB_PATH)
df_raw_orders = pd.read_sql_query("SELECT * FROM raw_orders;", conn)
df_raw_reviews = pd.read_sql_query("SELECT * FROM raw_order_reviews;", conn)

# Step 2: Pass data to the Cleaning Engine
df_clean_orders, df_clean_reviews = run_data_preprocessing(df_raw_orders, df_raw_reviews)

# Step 3: Pass clean data to the Feature Engineering Engine
df_final_orders, df_final_reviews = run_feature_engineering(df_clean_orders, df_clean_reviews)

# Step 4: Save the finalized clean data back to production database tables
print("--- [Step 3] Persisting Governed Production Assets to Database ---")
df_final_orders.to_sql("prod_orders_cleaned", conn, if_exists="replace", index=False)
df_final_reviews.to_sql("prod_reviews_cleaned", conn, if_exists="replace", index=False)

conn.close()
print("\n✔ SUCCESS: Entire relational pipeline executed cleanly.")
print("✔ Relational production tables are updated and ready for modeling/dashboards.")