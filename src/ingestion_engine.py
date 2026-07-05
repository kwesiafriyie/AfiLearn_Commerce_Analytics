import os
import sqlite3
import pandas as pd

# Define relative data directory pathways based on the project folder structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else "."
RAW_DATA_DIR = os.path.join(BASE_DIR, "data-package", "raw-data")
DB_DIR = os.path.join(BASE_DIR, "data-package", "database")

# Ensure database directory layers exist
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "afilearn_commerce_master.db")

print("=" * 70)
print("  AFILEARN COMMERCE ANALYTICS OFFICE - STAGE 2: INGESTION PIPELINE  ")
print("=" * 70)

# Map the 9 core relational Olist CSV datasets to their intended database staging tables
olist_files = {
    "olist_customers_dataset.csv": "raw_customers",
    "olist_orders_dataset.csv": "raw_orders",
    "olist_order_items_dataset.csv": "raw_order_items",
    "olist_order_payments_dataset.csv": "raw_order_payments",
    "olist_order_reviews_dataset.csv": "raw_order_reviews",
    "olist_products_dataset.csv": "raw_products",
    "olist_sellers_dataset.csv": "raw_sellers",
    "olist_geolocation_dataset.csv": "raw_geolocation",
    "product_category_name_translation.csv": "raw_category_translation"
}

try:
    # Establish connection loop to the local SQLite storage engine
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    print(f"✔ Programmatic link established to SQLite database at: {DB_PATH}\n")
    
    # Iterate, profile, and load each flat file into independent database staging zones
    for csv_file, table_name in olist_files.items():
        csv_path = os.path.join(RAW_DATA_DIR, csv_file)
        
        if not os.path.exists(csv_path):
            print(f"❌ Missing file error: {csv_file} not found in data/raw/. Skipping...")
            continue
            
        print(f"--- Processing source file: {csv_file} ---")
        
        # Stream into memory block chunks if necessary; load as string to prevent formatting assumptions
        df_chunk = pd.read_csv(csv_path, dtype=str)
        
        # Push raw data into the local storage layout, replacing if table exists
        df_chunk.to_sql(table_name, conn, if_exists="replace", index=False)
        
        # Verify the record numbers inside the newly initialized database layer
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cursor.fetchone()[0]
        print(f"  ✔ Successfully loaded table [{table_name}]. Ingested Rows: {row_count:,}")
        
    conn.close()
    print("\n" + "=" * 70)
    print("  PHASE 2 INGESTION COMPLETE: RAW SILOS LOADED INTO RELATIONAL ENGINE  ")
    print("=" * 70)

except Exception as e:
    print(f"❌ Structural Pipeline Failure: {e}")