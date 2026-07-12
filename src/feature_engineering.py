# Derived logic scripts (delivery-delay tags, review-sentiment vectors) 


import os
import pandas as pd
import numpy as np

# Define relative data directory pathways
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else "."
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "data-package", "intermediate")

def run_feature_engineering():
    print("--- [Step 2] Initiating Feature Engineering Engine ---")
    
    # Load clean intermediate files
    orders_path = os.path.join(INTERMEDIATE_DIR, "orders_cleaned.csv")
    reviews_path = os.path.join(INTERMEDIATE_DIR, "reviews_cleaned.csv")
    
    if not os.path.exists(orders_path) or not os.path.exists(reviews_path):
        raise FileNotFoundError("❌ Preprocessing error: Clean intermediate staging assets not found. Run preprocessing first.")
        
    df_orders = pd.read_csv(orders_path)
    df_reviews = pd.read_csv(reviews_path)
    
    # Force datetime parsing for structural calculations
    df_orders['order_delivered_customer_date'] = pd.to_datetime(df_orders['order_delivered_customer_date'])
    df_orders['order_estimated_delivery_date'] = pd.to_datetime(df_orders['order_estimated_delivery_date'])
    df_orders['order_purchase_timestamp'] = pd.to_datetime(df_orders['order_purchase_timestamp'])

    # FEATURE 1: Compute actual delivery delay days (Delivered Date minus Estimated Date)
    df_orders['delivery_delay_days'] = (
        df_orders['order_delivered_customer_date'] - df_orders['order_estimated_delivery_date']
    ).dt.days
    
    # FEATURE 2: Create a binary classification flag for logistics delays (1 if late, 0 if on-time/early)
    df_orders['is_late_delivery'] = np.where(df_orders['delivery_delay_days'] > 0, 1, 0)
    
    # FEATURE 3: Create a purchase month-year temporal index string for time-series trend tracking
    df_orders['purchase_month_period'] = df_orders['order_purchase_timestamp'].dt.to_period('M').astype(str)
    
    # FEATURE 4: Classify review scores into text sentiment groups (Satisfied vs Dissatisfied)
    df_reviews['review_sentiment'] = np.where(df_reviews['review_score'] >= 4, 'Satisfied', 'Dissatisfied')
    
    # Overwrite intermediate tracking assets with enriched feature arrays
    df_orders.to_csv(orders_path, index=False)
    df_reviews.to_csv(reviews_path, index=False)
    
    print("  ✔ Feature Engineering complete. Enriched metadata features saved successfully.")
    print(f"  ✔ Flagged Late Deliveries: {df_orders['is_late_delivery'].sum():,}")
    print(f"  ✔ Computed Sentiment Classes -> Satisfied: {df_reviews[df_reviews['review_sentiment'] == 'Satisfied'].shape[0]:,}\n")

if __name__ == "__main__":
    run_feature_engineering()