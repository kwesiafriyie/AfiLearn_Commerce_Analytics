import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Dynamically calculate BASE_DIR & RAW_DATA_DIR relative to script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else "."
RAW_DATA_DIR = os.path.join(BASE_DIR, "data-package", "raw-data")

def load_payment_data(data_dir=RAW_DATA_DIR):
    """
    Loads order payments and orders datasets from the specified raw data directory.
    """
    payments_path = os.path.join(data_dir, "olist_order_payments_dataset.csv")
    orders_path = os.path.join(data_dir, "olist_orders_dataset.csv")
    
    payments_df = pd.read_csv(payments_path)
    orders_df = pd.read_csv(orders_path)
    return payments_df, orders_df

def analyze_payment_distribution(payments_df, orders_df):
    # 1. Merge Payments with Orders
    df = payments_df.merge(orders_df, on="order_id")

    # 2. Aggregations & Summary
    payment_summary = df.groupby('payment_type').agg(
        transaction_count=('order_id', 'count'),
        total_payment_value=('payment_value', 'sum'),
        avg_order_value=('payment_value', 'mean'),
        avg_installments=('payment_installments', 'mean')
    ).reset_index()

    payment_summary['share_of_total_value'] = (
        payment_summary['total_payment_value'] / payment_summary['total_payment_value'].sum()
    ) * 100

    # 3. Absolute Output Directory Setup
    screenshots_dir = os.path.join(BASE_DIR, 'dashboards', 'screenshots')
    intermediate_dir = os.path.join(BASE_DIR, 'data-package', 'intermediate')
    
    os.makedirs(screenshots_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)

    # 4. Generate & Save Visualizations
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Pie Chart: Market Share
    ax[0].pie(
        payment_summary['total_payment_value'], 
        labels=payment_summary['payment_type'], 
        autopct='%1.1f%%', 
        colors=sns.color_palette('pastel')
    )
    ax[0].set_title('Payment Method Share by Total Value')

    # Bar Chart: AOV by Payment Type
    sns.barplot(
        x='payment_type', 
        y='avg_order_value', 
        data=payment_summary, 
        ax=ax[1], 
        palette='magma'
    )
    ax[1].set_title('Average Order Value (AOV) by Payment Type')
    ax[1].set_ylabel('Avg Order Value ($)')

    plt.tight_layout()
    plot_path = os.path.join(screenshots_dir, 'payment_method_distribution.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Chart saved to: {plot_path}")

    # 5. Export Intermediate Data Mart
    mart_path = os.path.join(intermediate_dir, 'fact_payment_analytics.csv')
    df.to_csv(mart_path, index=False)
    print(f"Data mart saved to: {mart_path}")

    return payment_summary

if __name__ == "__main__":
    print("Executing Payment Method Distribution analysis...")
    print(f"DEBUG: Script is executing from BASE_DIR -> {BASE_DIR}")
    print(f"DEBUG: Reading raw data from -> {RAW_DATA_DIR}")
    
    try:
        # Load and process data
        payments_df, orders_df = load_payment_data()
        summary = analyze_payment_distribution(payments_df, orders_df)
        
        print("\n--- Payment Method Summary ---")
        print(summary.to_string(index=False))
        
        # Verify generated files on disk
        target_chart = os.path.join(BASE_DIR, 'dashboards', 'screenshots', 'payment_method_distribution.png')
        target_csv = os.path.join(BASE_DIR, 'data-package', 'intermediate', 'fact_payment_analytics.csv')
        
        print("\n--- File Verification Check ---")
        print(f"Chart Exists ({target_chart}): {os.path.exists(target_chart)}")
        print(f"CSV Mart Exists ({target_csv}): {os.path.exists(target_csv)}")
        
        print("\nAnalysis completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\n[Error] Missing raw input files: {e}")
        print("Please ensure raw CSV files are placed in 'data-package/raw-data/'.")