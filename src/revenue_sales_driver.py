import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else "."
RAW_DATA_DIR = os.path.join(BASE_DIR, "data-package", "raw-data")
#DB_DIR = os.path.join(BASE_DIR, "data-package", "database")

def load_data(data_dir=RAW_DATA_DIR):
    """
    Loads raw e-commerce CSV files. Adjust file names if your raw directory differs.
    """
    orders_df = pd.read_csv(os.path.join(data_dir, "olist_orders_dataset.csv"))
    items_df = pd.read_csv(os.path.join(data_dir, "olist_order_items_dataset.csv"))
    products_df = pd.read_csv(os.path.join(data_dir, "olist_products_dataset.csv"))
    
    return orders_df, items_df, products_df

def analyze_revenue_drivers(orders_df, items_df, products_df):
    # 1. Merge transactional tables
    df = items_df.merge(orders_df, on="order_id").merge(products_df, on="product_id")
    
    # 2. Calculate Gross Merchandise Value (GMV) & Category Metrics
    df['total_item_gmv'] = df['price'] + df['freight_value']
    
    category_summary = df.groupby('product_category_name').agg(
        total_revenue=('total_item_gmv', 'sum'),
        order_count=('order_id', 'nunique'),
        avg_price=('price', 'mean'),
        avg_freight=('freight_value', 'mean')
    ).sort_values(by='total_revenue', ascending=False).reset_index()

    # 3. Model Feature Importance for Order GMV Drivers
    feature_cols = ['price', 'freight_value', 'product_weight_g', 'product_photos_qty']
    # Select available numerical columns
    available_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[available_cols].fillna(df[available_cols].median())
    y = df['total_item_gmv']

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importance_df = pd.DataFrame({
        'feature': available_cols,
        'importance': rf.feature_importances_
    }).sort_values(by='importance', ascending=False)

    # 4. Ensure output directories exist
    os.makedirs('dashboards/screenshots', exist_ok=True)
    os.makedirs('data-package/intermediate', exist_ok=True)

    # 5. Export Diagnostic Visual Plot
    plt.figure(figsize=(10, 5))
    sns.barplot(
        x='total_revenue', 
        y='product_category_name', 
        data=category_summary.head(10), 
        palette='viridis'
    )
    plt.title('Top 10 Revenue-Generating Product Categories (GMV)')
    plt.xlabel('Total Revenue ($)')
    plt.ylabel('Product Category')
    plt.tight_layout()
    
    plot_path = 'dashboards/screenshots/revenue_by_category.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Chart saved to: {plot_path}")

    # 6. Export Intermediate Data Mart
    mart_path = 'data-package/intermediate/fact_revenue_analytics.csv'
    df.to_csv(mart_path, index=False)
    print(f"Data mart saved to: {mart_path}")

    return category_summary, importance_df

if __name__ == "__main__":
    print("Executing Revenue & Sales Drivers analysis...")
    
    # 1. Load Data
    try:
        orders_df, items_df, products_df = load_data()
        
        # 2. Run Pipeline
        summary, importance = analyze_revenue_drivers(orders_df, items_df, products_df)
        
        # 3. Display Terminal Output
        print("\n--- Top 5 Product Categories by Revenue ---")
        print(summary.head(5).to_string(index=False))
        
        print("\n--- Key Drivers of Order GMV ---")
        print(importance.to_string(index=False))
        
        print("\nAnalysis completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\n[Error] Missing raw input files: {e}")
        print("Please ensure raw CSV files are placed in 'data-package/raw-data/' or update load_data() paths.")