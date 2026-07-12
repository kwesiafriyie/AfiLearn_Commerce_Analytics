# Inside main.py
import os
import sqlite3
import pandas as pd
from src.data_preprocessing import run_data_preprocessing
from src.feature_engineering import run_feature_engineering

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data-package", "database", "afilearn_commerce_master.db")
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "data-package", "intermediate")

print("======================================================================")
print("  AFILEARN COMMERCE ANALYTICS - PRODUCTION COORDINDATOR PIPELINE      ")
print("======================================================================\n")

# Run Step 1: Preprocess and save to disk
run_data_preprocessing()

# Run Step 2: Extract from disk, feature engineer, save to disk
run_feature_engineering()

# Run Step 3: Read final disk files and push everything cleanly to Database
print("--- [Step 3] Pushing Final Enriched Disk Datasets to Database ---")

df_final_orders = pd.read_csv(os.path.join(INTERMEDIATE_DIR, "orders_cleaned.csv"))
df_final_items = pd.read_csv(os.path.join(INTERMEDIATE_DIR, "items_cleaned.csv"))
df_final_reviews = pd.read_csv(os.path.join(INTERMEDIATE_DIR, "reviews_cleaned.csv"))

conn = sqlite3.connect(DB_PATH)
df_final_orders.to_sql("prod_orders_cleaned", conn, if_exists="replace", index=False)
df_final_items.to_sql("prod_items_cleaned", conn, if_exists="replace", index=False)
df_final_reviews.to_sql("prod_reviews_cleaned", conn, if_exists="replace", index=False)
conn.close()

print("\n✔ SUCCESS: Entire checkpointed data pipeline executed flawlessly.")
print("✔ CSV archives are synced on disk, and production tables are live in SQLite!")