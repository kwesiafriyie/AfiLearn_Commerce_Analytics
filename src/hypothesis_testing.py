import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. Base Directory and File Path Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else "."
RAW_DATA_DIR = os.path.join(BASE_DIR, "data-package", "raw-data")

def load_data(data_dir=RAW_DATA_DIR):
    """
    Loads raw orders and order reviews datasets.
    """
    orders_path = os.path.join(data_dir, "olist_orders_dataset.csv")
    reviews_path = os.path.join(data_dir, "olist_order_reviews_dataset.csv")
    
    orders_df = pd.read_csv(orders_path)
    reviews_df = pd.read_csv(reviews_path)
    return orders_df, reviews_df

def run_hypothesis_test(orders_df, reviews_df):
    # 1. Convert timestamps to datetime
    orders_df['order_delivered_customer_date'] = pd.to_datetime(orders_df['order_delivered_customer_date'])
    orders_df['order_estimated_delivery_date'] = pd.to_datetime(orders_df['order_estimated_delivery_date'])

    # 2. Filter delivered orders and create binary delay target
    delivered_df = orders_df[orders_df['order_status'] == 'delivered'].copy()
    delivered_df['is_delayed'] = (
        delivered_df['order_delivered_customer_date'] > delivered_df['order_estimated_delivery_date']
    ).astype(int)

    # 3. Merge orders with customer review ratings
    merged_df = delivered_df.merge(reviews_df[['order_id', 'review_score']], on='order_id', how='inner')
    merged_df = merged_df.dropna(subset=['review_score', 'is_delayed'])

    # 4. Separate samples for hypothesis testing
    delayed_scores = merged_df[merged_df['is_delayed'] == 1]['review_score']
    ontime_scores = merged_df[merged_df['is_delayed'] == 0]['review_score']

    # 5. Perform Mann-Whitney U Test (One-Sided: delayed < ontime)
    u_stat, p_value = stats.mannwhitneyu(delayed_scores, ontime_scores, alternative='less')

    # 6. Calculate Rank-Biserial Correlation (Effect Size r)
    n1, n2 = len(delayed_scores), len(ontime_scores)
    rank_biserial_r = 1 - (2 * u_stat) / (n1 * n2)

    alpha = 0.05
    verdict = (
        "REJECT H0: Delivery delays have a statistically significant negative impact on review scores."
        if p_value < alpha else
        "FAIL TO REJECT H0: No statistically significant difference detected."
    )

    # 7. Summary Results Table
    results_df = pd.DataFrame({
        'Hypothesis Parameter': [
            'Null Hypothesis (H0)',
            'Alternative Hypothesis (H1)',
            'Significance Level (alpha)',
            'Sample Size (On-Time Orders)',
            'Sample Size (Delayed Orders)',
            'On-Time Orders Median CSAT',
            'Delayed Orders Median CSAT',
            'On-Time Orders Mean CSAT',
            'Delayed Orders Mean CSAT',
            'Mann-Whitney U Statistic',
            'p-value',
            'Effect Size (Rank-Biserial r)',
            'Test Verdict'
        ],
        'Value / Result': [
            'Delays have no impact on review scores',
            'Delays decrease review scores',
            '0.05',
            f"{n2:,}",
            f"{n1:,}",
            f"{ontime_scores.median():.1f} Stars",
            f"{delayed_scores.median():.1f} Stars",
            f"{ontime_scores.mean():.2f} / 5.0",
            f"{delayed_scores.mean():.2f} / 5.0",
            f"{u_stat:,.2f}",
            f"{p_value:.5e}" if p_value > 0 else "< 0.0001",
            f"{rank_biserial_r:.4f}",
            verdict
        ]
    })

    # 8. Output Directory Setup
    screenshots_dir = os.path.join(BASE_DIR, 'dashboards', 'screenshots')
    intermediate_dir = os.path.join(BASE_DIR, 'data-package', 'intermediate')
    os.makedirs(screenshots_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)

    # 9. Plot and Save CSAT Distribution
    plt.figure(figsize=(9, 5))
    sns.boxplot(
        x='is_delayed', 
        y='review_score', 
        data=merged_df, 
        palette=['#10b981', '#ef4444'],
        showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"}
    )
    plt.title('Mann-Whitney U Test: CSAT Score Distribution by Delivery SLA Status', fontsize=12, fontweight='bold')
    plt.xticks([0, 1], ['On-Time Orders', 'Delayed Orders'], fontsize=10)
    plt.xlabel('Delivery SLA Status', fontsize=10)
    plt.ylabel('Customer Review Score (1-5 Stars)', fontsize=10)
    plt.ylim(0.5, 5.5)
    plt.tight_layout()

    chart_path = os.path.join(screenshots_dir, 'mann_whitney_csat_distribution.png')
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Chart saved to: {chart_path}")

    # 10. Save CSV Data Mart
    csv_path = os.path.join(intermediate_dir, 'fact_hypothesis_testing.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")

    return results_df

if __name__ == "__main__":
    print("Executing Statistical Hypothesis Test...")
    try:
        orders_df, reviews_df = load_data()
        results_df = run_hypothesis_test(orders_df, reviews_df)
        print("\n--- Hypothesis Test Summary Table ---")
        print(results_df.to_string(index=False))
        print("\nAnalysis completed successfully!")
    except FileNotFoundError as e:
        print(f"\n[Error] Missing raw input files: {e}")