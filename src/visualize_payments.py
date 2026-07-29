import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# 1. Load summary data (or pass from your dataframe)
data = {
    'payment_type': ['Credit Card', 'Boleto', 'Voucher', 'Debit Card'],
    'transaction_count': [76795, 19784, 5775, 1529],
    'total_payment_value': [12542084.19, 2869361.27, 379436.87, 217989.79],
    'avg_order_value': [163.32, 145.03, 65.70, 142.57],
    'avg_installments': [3.51, 1.00, 1.00, 1.00],
    'share_of_total_value': [78.34, 17.92, 2.37, 1.36]
}

df = pd.DataFrame(data)

# 2. Create Multi-Panel Figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle('E-Commerce Payment Method Analytics & Behavioral Breakdown', fontsize=16, fontweight='bold', y=1.02)

# Panel 1: Donut Chart - Share of Total GMV
colors = ['#4f46e5', '#06b6d4', '#f59e0b', '#10b981']
wedges, texts, autotexts = axes[0].pie(
    df['share_of_total_value'], 
    labels=df['payment_type'], 
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    pctdistance=0.75,
    textprops=dict(color="black", weight="bold")
)
# Make it a Donut Chart
centre_circle = plt.Circle((0,0), 0.55, fc='white')
axes[0].add_artist(centre_circle)
axes[0].set_title('GMV Market Share (%)', fontsize=12, fontweight='bold')

# Panel 2: Bar Chart - Average Order Value (AOV)
barplot1 = sns.barplot(
    x='payment_type', 
    y='avg_order_value', 
    data=df, 
    ax=axes[1], 
    palette=colors
)
axes[1].set_title('Average Order Value (AOV in $)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('AOV ($)')
axes[1].set_ylim(0, 200)

# Add value labels above bars
for p in barplot1.patches:
    axes[1].annotate(f"${p.get_height():.2f}", 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')

# Panel 3: Bar Chart - Average Installments
barplot2 = sns.barplot(
    x='payment_type', 
    y='avg_installments', 
    data=df, 
    ax=axes[2], 
    palette=colors
)
axes[2].set_title('Avg. Selected Installments', fontsize=12, fontweight='bold')
axes[2].set_xlabel('')
axes[2].set_ylabel('Installment Count')
axes[2].set_ylim(0, 5)

# Add value labels above bars
for p in barplot2.patches:
    axes[2].annotate(f"{p.get_height():.2f}x", 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')

plt.tight_layout()

# 3. Ensure Output Directories Exist and Save Chart
output_dir = os.path.join("dashboards", "screenshots")
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "payment_method_distribution.png")

plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Visualization saved successfully to: {plot_path}")