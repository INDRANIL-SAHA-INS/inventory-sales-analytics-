import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta

# Set up directories
OUTPUT_DIR = 'outputs'
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')

# Ensure directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Data loading
def load_data(file_path):
    print(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

# 1. DATA VALIDATION
def validate_data(df):
    print("Validating data...")
    
    # Check required columns
    required_columns = [
        'Product_ID', 'Catagory', 'Stock_Quantity', 'Reorder_Level',
        'Unit_Price', 'Sales_Volume', 'Expiration_Date'
    ]
    
    # Map dataset columns to required columns for analysis
    column_mapping = {
        'Catagory': 'Product_Category',
        'Sales_Volume': 'Quantity_Sold',
        'Stock_Quantity': 'Stock_On_Hand',
        'Expiration_Date': 'Expiry_Date'
    }
    
    # Rename columns for analysis
    df = df.rename(columns=column_mapping)
    
    # Convert price columns from string to float (remove $ and convert)
    if 'Unit_Price' in df.columns:
        df['Unit_Price'] = df['Unit_Price'].str.replace('$', '').str.strip().astype(float)
    
    # Convert dates to datetime
    if 'Expiry_Date' in df.columns:
        df['Expiry_Date'] = pd.to_datetime(df['Expiry_Date'], errors='coerce')
    
    # Calculate profit based on sales and price if not available
    if 'Profit' not in df.columns:
        # Assuming 20% profit margin for simplicity
        df['Profit'] = df['Quantity_Sold'] * df['Unit_Price'] * 0.2
    
    # Add a Discount column if not present (set to 0)
    if 'Discount_%' not in df.columns:
        df['Discount_%'] = 0
    
    return df

# 2. DATA PREPROCESSING
def preprocess_data(df):
    print("Preprocessing data...")
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype in [np.int64, np.float64]:
            # Fill numeric columns with median
            df[col] = df[col].fillna(df[col].median())
        else:
            # Fill categorical columns with "Unknown" or most frequent
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    
    # Outlier treatment for numeric columns
    for col in ['Quantity_Sold', 'Stock_On_Hand']:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            # Cap values above 95th percentile
            df[col] = np.where(df[col] > df[col].quantile(0.95), df[col].quantile(0.95), df[col])
    
    # Create derived columns
    df['Inventory_Turnover_Ratio'] = df['Quantity_Sold'] / df['Stock_On_Hand']
    df['Overstock_Flag'] = df['Stock_On_Hand'] > 2 * df['Reorder_Level']
    
    # If Reorder_Level is missing, assume it as Avg_Monthly_Sales * 1.5
    # (using Quantity_Sold as a proxy for monthly sales)
    if 'Reorder_Level' in df.columns and df['Reorder_Level'].isna().any():
        df['Reorder_Level'] = df['Reorder_Level'].fillna(df['Quantity_Sold'] * 1.5)
    
    # Calculate total revenue
    df['Total_Revenue'] = df['Quantity_Sold'] * df['Unit_Price']
    
    return df

# 3. ANALYSIS TASKS
def analyze_overstocked_products(df):
    print("[PRIORITY] Analyzing overstocked products...")
    
    # Filter overstocked products
    overstocked = df[df['Overstock_Flag'] == True].sort_values(by='Stock_On_Hand', ascending=False)
    
    # Create report
    overstocked_report = overstocked[['Product_ID', 'Product_Name', 'Product_Category', 
                                     'Stock_On_Hand', 'Reorder_Level', 'Quantity_Sold', 
                                     'Inventory_Turnover_Ratio']]
    
    # Save report
    overstocked_report.to_csv(os.path.join(REPORTS_DIR, 'overstocked_products.csv'), index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    sns.barplot(data=overstocked.head(15), x='Product_Name', y='Stock_On_Hand')
    plt.xticks(rotation=90)
    plt.title('Top 15 Overstocked Products')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'overstocked_products.png'))
    
    return overstocked

def analyze_top_products(df):
    print("[PRIORITY] Analyzing top products by quantity sold and revenue...")
    
    # Top products by quantity sold
    top_by_quantity = df.sort_values(by='Quantity_Sold', ascending=False).head(10)
    
    # Top products by revenue
    top_by_revenue = df.sort_values(by='Total_Revenue', ascending=False).head(10)
    
    # Save reports
    top_by_quantity[['Product_ID', 'Product_Name', 'Product_Category', 'Quantity_Sold', 
                    'Unit_Price', 'Total_Revenue']].to_csv(
                    os.path.join(REPORTS_DIR, 'top_products_by_quantity.csv'), index=False)
    
    top_by_revenue[['Product_ID', 'Product_Name', 'Product_Category', 'Quantity_Sold', 
                   'Unit_Price', 'Total_Revenue']].to_csv(
                   os.path.join(REPORTS_DIR, 'top_products_by_revenue.csv'), index=False)
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_by_quantity, x='Product_Name', y='Quantity_Sold')
    plt.xticks(rotation=90)
    plt.title('Top 10 Products by Quantity Sold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'top_products_by_quantity.png'))
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_by_revenue, x='Product_Name', y='Total_Revenue')
    plt.xticks(rotation=90)
    plt.title('Top 10 Products by Revenue')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'top_products_by_revenue.png'))
    
    return top_by_quantity, top_by_revenue

def analyze_category_revenue(df):
    print("[PRIORITY] Analyzing category-wise revenue...")
    
    # Aggregate by category
    category_revenue = df.groupby('Product_Category').agg({
        'Total_Revenue': 'sum',
        'Quantity_Sold': 'sum',
        'Product_ID': 'count'  # count of products
    }).reset_index().sort_values(by='Total_Revenue', ascending=False)
    
    category_revenue = category_revenue.rename(columns={'Product_ID': 'Product_Count'})
    
    # Save report
    category_revenue.to_csv(os.path.join(REPORTS_DIR, 'category_revenue.csv'), index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    sns.barplot(data=category_revenue, x='Product_Category', y='Total_Revenue')
    plt.xticks(rotation=45)
    plt.title('Total Revenue by Product Category')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'category_revenue.png'))
    
    return category_revenue

def analyze_expiry_risk(df):
    print("[PRIORITY] Analyzing expiry risk...")
    
    if 'Expiry_Date' not in df.columns:
        print("Expiry_Date column not available. Skipping expiry risk analysis.")
        return None
    
    # Calculate days until expiry
    today = datetime.now()
    df['Days_To_Expiry'] = (df['Expiry_Date'] - today).dt.days
    
    # Create risk categories
    df['Expiry_Risk'] = 'Green'
    df.loc[df['Days_To_Expiry'] <= 60, 'Expiry_Risk'] = 'Yellow'
    df.loc[df['Days_To_Expiry'] <= 30, 'Expiry_Risk'] = 'Red'
    
    # Filter products with expiry risk
    at_risk = df[df['Expiry_Risk'] != 'Green'].sort_values(by='Days_To_Expiry')
    
    # Focus on products with stock
    at_risk_with_stock = at_risk[at_risk['Stock_On_Hand'] > 0]
    
    # Save report
    at_risk_with_stock[['Product_ID', 'Product_Name', 'Product_Category', 
                       'Stock_On_Hand', 'Days_To_Expiry', 'Expiry_Risk', 'Unit_Price']].to_csv(
                       os.path.join(REPORTS_DIR, 'expiry_risk.csv'), index=False)
    
    # Create visualization - count of products by risk category
    risk_counts = at_risk_with_stock['Expiry_Risk'].value_counts().reset_index()
    risk_counts.columns = ['Risk_Category', 'Count']
    
    plt.figure(figsize=(10, 6))
    colors = {'Red': 'red', 'Yellow': 'yellow', 'Green': 'green'}
    ax = sns.barplot(data=risk_counts, x='Risk_Category', y='Count', palette=colors)
    plt.title('Product Count by Expiry Risk Category')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'expiry_risk.png'))
    
    return at_risk_with_stock

# Optional analysis
def analyze_discount_impact(df):
    print("[OPTIONAL] Analyzing discount impact...")
    
    if 'Discount_%' not in df.columns or df['Discount_%'].sum() == 0:
        print("Discount data not available or all values are 0. Skipping discount impact analysis.")
        return None
    
    # Create scatter plot of Discount vs Quantity Sold
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Discount_%', y='Quantity_Sold')
    plt.title('Impact of Discount on Quantity Sold')
    plt.xlabel('Discount %')
    plt.ylabel('Quantity Sold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'discount_impact.png'))
    
    # Correlation analysis
    correlation = df['Discount_%'].corr(df['Quantity_Sold'])
    
    with open(os.path.join(REPORTS_DIR, 'discount_correlation.txt'), 'w') as f:
        f.write(f"Correlation between Discount and Quantity Sold: {correlation:.4f}")
    
    return correlation

# 4. Generate executive summary
def generate_executive_summary(df, overstocked, top_revenue, category_revenue, expiry_risk):
    print("Generating executive summary...")
    
    insights = []
    
    # Insight 1: Overstocked products
    if overstocked is not None and not overstocked.empty:
        overstock_value = (overstocked['Stock_On_Hand'] * overstocked['Unit_Price']).sum()
        insights.append(f"[PRIORITY] There are {len(overstocked)} overstocked products with a total value of ${overstock_value:.2f}. Consider running promotions or reducing reorder quantities.")
    
    # Insight 2: Top revenue generators
    if top_revenue is not None and not top_revenue.empty:
        top_category = category_revenue.iloc[0]['Product_Category']
        top_revenue_pct = (category_revenue.iloc[0]['Total_Revenue'] / category_revenue['Total_Revenue'].sum()) * 100
        insights.append(f"[PRIORITY] The {top_category} category generates ${category_revenue.iloc[0]['Total_Revenue']:.2f} in revenue ({top_revenue_pct:.1f}% of total). Consider expanding this product line.")
    
    # Insight 3: Expiry risk
    if expiry_risk is not None and not expiry_risk.empty:
        red_risk = expiry_risk[expiry_risk['Expiry_Risk'] == 'Red']
        if not red_risk.empty:
            red_risk_value = (red_risk['Stock_On_Hand'] * red_risk['Unit_Price']).sum()
            insights.append(f"[PRIORITY] {len(red_risk)} products worth ${red_risk_value:.2f} will expire within 30 days. Immediate action required with promotions or discounts.")
    
    # Insight 4: Inventory turnover
    low_turnover = df[df['Inventory_Turnover_Ratio'] < 1].sort_values(by='Inventory_Turnover_Ratio')
    if not low_turnover.empty:
        low_turnover_value = (low_turnover['Stock_On_Hand'] * low_turnover['Unit_Price']).sum()
        insights.append(f"[PRIORITY] {len(low_turnover)} products have low turnover rates (<1), tying up ${low_turnover_value:.2f} in inventory. Review pricing strategy and promotions.")
    
    # Insight 5: Stock-outs risk
    potential_stockouts = df[df['Stock_On_Hand'] < df['Reorder_Level']]
    if not potential_stockouts.empty:
        high_revenue_at_risk = potential_stockouts.sort_values(by='Total_Revenue', ascending=False).head(5)
        at_risk_revenue = high_revenue_at_risk['Total_Revenue'].sum()
        insights.append(f"[PRIORITY] {len(potential_stockouts)} products are below reorder levels, putting ${at_risk_revenue:.2f} in revenue at risk. Expedite restocking of top sellers.")
    
    # Write to file
    with open(os.path.join(REPORTS_DIR, 'executive_summary.md'), 'w') as f:
        f.write("# Executive Summary: Grocery Inventory and Sales Analysis\n\n")
        f.write("## Key Insights and Action Items\n\n")
        for i, insight in enumerate(insights[:5], 1):
            f.write(f"{i}. {insight}\n\n")
    
    return insights

# Main function
def main():
    # Load data
    data_path = 'data/Grocery_Inventory_and_Sales_Dataset.csv'
    df = load_data(data_path)
    
    # Validate and preprocess
    df = validate_data(df)
    df = preprocess_data(df)
    
    # Save cleaned dataset
    df.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_dataset.csv'), index=False)
    
    # Perform analyses
    overstocked = analyze_overstocked_products(df)
    top_quantity, top_revenue = analyze_top_products(df)
    category_revenue = analyze_category_revenue(df)
    expiry_risk = analyze_expiry_risk(df)
    
    # Optional analyses
    discount_impact = analyze_discount_impact(df)
    
    # Generate executive summary
    insights = generate_executive_summary(df, overstocked, top_revenue, category_revenue, expiry_risk)
    
    print("\nAnalysis complete! Results saved to:")
    print(f"- Cleaned dataset: {os.path.join(OUTPUT_DIR, 'cleaned_dataset.csv')}")
    print(f"- Reports: {REPORTS_DIR}")
    print(f"- Visualizations: {PLOTS_DIR}")
    print(f"- Executive summary: {os.path.join(REPORTS_DIR, 'executive_summary.md')}")

if __name__ == "__main__":
    main()
