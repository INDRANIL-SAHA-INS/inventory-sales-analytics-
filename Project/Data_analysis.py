import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import plotly.express as px
import matplotlib.patches as mpatches
# import geopandas as gpd  # Uncomment if geopandas is available in your environment

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
    
    # Interactive Plot: Top 15 Overstocked Products
    top_15 = overstocked.head(15)
    fig = px.bar(
        top_15,
        x='Product_Name',
        y='Stock_On_Hand',
        color='Product_Category',
        hover_data=['Product_ID', 'Reorder_Level', 'Quantity_Sold', 'Inventory_Turnover_Ratio'],
        title='Top 15 Overstocked Products',
        labels={'Product_Name': 'Product', 'Stock_On_Hand': 'Stock Quantity'},
        template='plotly_white'
    )

    # Make layout pretty
    fig.update_layout(
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=80, b=120),
        font=dict(size=12),
        title_font_size=20
    )

    fig.show()
    
    # Also create a matplotlib visualization for non-interactive environments
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
    
    # Plot: Top by Quantity Sold
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_by_quantity, x='Quantity_Sold', y='Product_Name', palette='crest')
    plt.title('🌟 Top 10 Products by Quantity Sold', fontsize=16, fontweight='bold')
    plt.xlabel('Quantity Sold', fontsize=12)
    plt.ylabel('Product Name', fontsize=12)
    for i, value in enumerate(top_by_quantity['Quantity_Sold']):
        plt.text(value, i, f'{value:,}', va='center', ha='left', fontsize=10)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'top_products_by_quantity.png'), dpi=300)
    
    # Plot: Top by Revenue
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_by_revenue, x='Total_Revenue', y='Product_Name', palette='mako')
    plt.title('💰 Top 10 Products by Total Revenue', fontsize=16, fontweight='bold')
    plt.xlabel('Total Revenue ($)', fontsize=12)
    plt.ylabel('Product Name', fontsize=12)
    for i, value in enumerate(top_by_revenue['Total_Revenue']):
        plt.text(value, i, f'${value:,.2f}', va='center', ha='left', fontsize=10)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'top_products_by_revenue.png'), dpi=300)
    
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

def analyze_2025_expiry_risk(df):
    print("[PRIORITY] Analyzing expiry risk for products expiring in 2025...")
    
    if 'Expiry_Date' not in df.columns:
        print("Expiry_Date column not available. Skipping 2025 expiry risk analysis.")
        return None
    
    # Filter products with expiry in 2025
    df_2025 = df[df['Expiry_Date'].dt.year == 2025].copy()
    
    if df_2025.empty:
        print("No products with expiry in 2025 found.")
        return None

    # Calculate days until expiry
    today = datetime.now()
    df_2025['Days_To_Expiry'] = (df_2025['Expiry_Date'] - today).dt.days
    
    # Create risk categories
    df_2025['Expiry_Risk'] = 'Green'
    df_2025.loc[df_2025['Days_To_Expiry'] <= 60, 'Expiry_Risk'] = 'Yellow'
    df_2025.loc[df_2025['Days_To_Expiry'] <= 30, 'Expiry_Risk'] = 'Red'
    
    # Filter products with expiry risk
    at_risk_2025 = df_2025[df_2025['Expiry_Risk'] != 'Green'].sort_values(by='Days_To_Expiry')
    
    # Focus on products with stock
    at_risk_2025_with_stock = at_risk_2025[at_risk_2025['Stock_On_Hand'] > 0]
    
    # Save CSV report
    at_risk_2025_with_stock[['Product_ID', 'Product_Name', 'Product_Category', 
                             'Stock_On_Hand', 'Days_To_Expiry', 'Expiry_Risk', 'Unit_Price']].to_csv(
                             os.path.join(REPORTS_DIR, 'expiry_risk_2025.csv'), index=False)
    
    # Create visualization - count of 2025 at-risk products
    risk_counts = at_risk_2025_with_stock['Expiry_Risk'].value_counts().reset_index()
    risk_counts.columns = ['Risk_Category', 'Count']
    
    plt.figure(figsize=(10, 6))
    colors = {'Red': 'red', 'Yellow': 'yellow', 'Green': 'green'}
    sns.barplot(data=risk_counts, x='Risk_Category', y='Count', palette=colors)
    plt.title('2025 Expiry Risk: Product Count by Risk Category')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'expiry_risk_2025.png'))
    
    return at_risk_2025_with_stock

def analyze_supplier_sales(df):
    print("[PRIORITY] Analyzing supplier-wise sales...")
    
    # Aggregate sales data by supplier and product category
    supplier_sales = df.groupby(['Supplier_Name', 'Product_Category']).agg({
        'Total_Revenue': 'sum',
        'Quantity_Sold': 'sum'
    }).reset_index()
    
    # Calculate total revenue per supplier
    supplier_total_revenue = supplier_sales.groupby('Supplier_Name')['Total_Revenue'].sum().reset_index()
    
    # Get the top 10 suppliers by total revenue
    top_suppliers = supplier_total_revenue.nlargest(10, 'Total_Revenue')['Supplier_Name']
    
    # Filter the original data for only the top 10 suppliers
    top_supplier_sales = supplier_sales[supplier_sales['Supplier_Name'].isin(top_suppliers)]
    
    # Pivot the data for a stacked bar chart
    supplier_sales_pivot = top_supplier_sales.pivot(index='Supplier_Name', 
                                                   columns='Product_Category', 
                                                   values='Total_Revenue').fillna(0)
    
    # Sort suppliers by total revenue for better visualization
    supplier_sales_pivot['Total'] = supplier_sales_pivot.sum(axis=1)
    supplier_sales_pivot = supplier_sales_pivot.sort_values(by='Total', ascending=False).drop(columns='Total')
    
    # Reset index for plotly
    supplier_sales_pivot_reset = supplier_sales_pivot.reset_index()
    
    # Save report
    top_supplier_sales.to_csv(os.path.join(REPORTS_DIR, 'top_supplier_sales.csv'), index=False)
    
    # Create the interactive stacked bar chart using Plotly
    fig = px.bar(supplier_sales_pivot_reset,
                 x=supplier_sales_pivot_reset.columns[1:],  # All categories
                 y='Supplier_Name',
                 title="Top 10 Suppliers - Supplier-wise Sales Analysis",
                 labels={'value': 'Total Revenue', 'Supplier_Name': 'Supplier'},
                 color_discrete_sequence=px.colors.sequential.Viridis,
                 height=600)
    
    # Show the plot
    fig.update_layout(barmode='stack',
                      xaxis_title="Total Revenue",
                      yaxis_title="Supplier Name",
                      legend_title="Product Category",
                      xaxis={'showgrid': False},
                      yaxis={'showgrid': False})
    fig.show()
    
    # Also create a matplotlib visualization for non-interactive environments
    plt.figure(figsize=(12, 8))
    top_suppliers_data = supplier_total_revenue[supplier_total_revenue['Supplier_Name'].isin(top_suppliers)]
    sns.barplot(data=top_suppliers_data, y='Supplier_Name', x='Total_Revenue', palette='viridis')
    plt.title('Top 10 Suppliers by Total Revenue', fontsize=16)
    plt.xlabel('Total Revenue ($)', fontsize=12)
    plt.ylabel('Supplier Name', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'top_suppliers_revenue.png'))
    
    return top_supplier_sales

def analyze_warehouse_distribution(df):
    print("[PRIORITY] Analyzing warehouse stock distribution...")
    
    # Aggregate stock by warehouse
    warehouse_stock = df.groupby('Warehouse_Location')['Stock_On_Hand'].sum().reset_index()
    
    # Sort the data by stock in descending order and select the top 10
    top_warehouses = warehouse_stock.sort_values(by='Stock_On_Hand', ascending=False).head(10)
    
    # Save report
    top_warehouses.to_csv(os.path.join(REPORTS_DIR, 'top_warehouses.csv'), index=False)
    
    # Create a Plotly treemap for the top 10 warehouses
    fig = px.treemap(top_warehouses, 
                    path=['Warehouse_Location'], 
                    values='Stock_On_Hand', 
                    color='Stock_On_Hand',
                    color_continuous_scale='Viridis',
                    title='Top 10 Warehouses by Stock Distribution')
    
    # Show the interactive treemap
    fig.update_layout(margin={"t": 40, "l": 40, "r": 40, "b": 40})
    fig.show()
    
    # Create a matplotlib visualization for non-interactive environments
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_warehouses, y='Warehouse_Location', x='Stock_On_Hand', palette='viridis')
    plt.title('Top 10 Warehouses by Stock Distribution', fontsize=16)
    plt.xlabel('Total Stock On Hand', fontsize=12)
    plt.ylabel('Warehouse Location', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'top_warehouses.png'))
    
    return top_warehouses

def analyze_supplier_product_count(df):
    print("[PRIORITY] Analyzing supplier product counts...")
    
    # Count products by supplier
    supplier_products = df.groupby('Supplier_Name')['Product_ID'].nunique().reset_index()
    supplier_products.columns = ['Supplier_Name', 'Product_Count']
    
    # Sort and get top 10
    top_suppliers = supplier_products.sort_values(by='Product_Count', ascending=False).head(10)
    
    # Save report
    top_suppliers.to_csv(os.path.join(REPORTS_DIR, 'top_suppliers_by_product_count.csv'), index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    top_suppliers_list = top_suppliers['Supplier_Name'].tolist()
    sns.countplot(data=df[df['Supplier_Name'].isin(top_suppliers_list)], 
                  y='Supplier_Name', 
                  palette='viridis', 
                  order=top_suppliers_list)
    plt.title('Count of Products by Top 10 Suppliers', fontsize=16)
    plt.xlabel('Count of Products', fontsize=12)
    plt.ylabel('Supplier Name', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'supplier_product_count.png'))
    
    return top_suppliers

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

def analyze_product_price_distribution(df):
    print("Analyzing product price distribution...")
    
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Unit_Price'], bins=30, kde=True)
    plt.title('Distribution of Product Prices')
    plt.xlabel('Unit Price ($)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'price_distribution.png'))
    
    # Create a boxplot to show outliers
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df['Unit_Price'])
    plt.title('Boxplot of Product Prices')
    plt.xlabel('Unit Price ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'price_boxplot.png'))
    
    return df['Unit_Price'].describe()

def analyze_monthly_sales_trends(df):
    print("Analyzing monthly sales trends...")
    
    # Check if Sale_Date exists
    if 'Sale_Date' not in df.columns:
        if 'Timestamp' in df.columns:
            df['Sale_Date'] = pd.to_datetime(df['Timestamp'])
        else:
            print("No date column available for monthly trends analysis.")
            return None
    
    # Ensure date format
    if not pd.api.types.is_datetime64_any_dtype(df['Sale_Date']):
        df['Sale_Date'] = pd.to_datetime(df['Sale_Date'])
    
    # Extract month and year
    df['Month_Year'] = df['Sale_Date'].dt.to_period('M')
    
    # Group by month and calculate total sales
    monthly_sales = df.groupby('Month_Year').agg({
        'Total_Revenue': 'sum',
        'Quantity_Sold': 'sum'
    }).reset_index()
    
    # Convert Period to string for plotting
    monthly_sales['Month_Year'] = monthly_sales['Month_Year'].astype(str)
    
    # Create line plot for monthly revenue
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly_sales, x='Month_Year', y='Total_Revenue', marker='o')
    plt.title('Monthly Sales Revenue Trends')
    plt.xlabel('Month')
    plt.ylabel('Total Revenue ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'monthly_revenue_trends.png'))
    
    # Create line plot for monthly quantity sold
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly_sales, x='Month_Year', y='Quantity_Sold', marker='o')
    plt.title('Monthly Quantity Sold Trends')
    plt.xlabel('Month')
    plt.ylabel('Quantity Sold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'monthly_quantity_trends.png'))
    
    return monthly_sales

def analyze_inventory_vs_sales(df):
    print("Analyzing inventory vs sales relationship...")
    
    # Create scatter plot of inventory vs sales
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Stock_On_Hand', y='Quantity_Sold', hue='Product_Category', alpha=0.7)
    plt.title('Inventory vs Sales by Product Category')
    plt.xlabel('Current Stock On Hand')
    plt.ylabel('Quantity Sold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'inventory_vs_sales.png'))
    
    # Calculate correlation between inventory and sales
    correlation = df['Stock_On_Hand'].corr(df['Quantity_Sold'])
    
    # Create a hexbin plot for dense data
    plt.figure(figsize=(10, 8))
    plt.hexbin(df['Stock_On_Hand'], df['Quantity_Sold'], gridsize=30, cmap='viridis')
    plt.colorbar(label='Count')
    plt.title('Density Plot: Inventory vs Sales')
    plt.xlabel('Current Stock On Hand')
    plt.ylabel('Quantity Sold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'inventory_vs_sales_density.png'))
    
    return correlation

def analyze_cost_vs_profit(df):
    print("Analyzing cost vs profit relationship...")
    
    # Check if cost data is available
    if 'Unit_Cost' not in df.columns:
        print("Unit_Cost column not available. Skipping cost vs profit analysis.")
        return None
    
    # Calculate profit per product
    df['Profit'] = df['Total_Revenue'] - (df['Unit_Cost'] * df['Quantity_Sold'])
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Unit_Cost', y='Profit', hue='Product_Category', size='Quantity_Sold', 
                    sizes=(20, 200), alpha=0.7)
    plt.title('Cost vs Profit by Product Category')
    plt.xlabel('Unit Cost ($)')
    plt.ylabel('Profit ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cost_vs_profit.png'))
    
    # Regression plot to show trend
    plt.figure(figsize=(10, 8))
    sns.regplot(data=df, x='Unit_Cost', y='Profit', scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title('Cost vs Profit Relationship with Trend')
    plt.xlabel('Unit Cost ($)')
    plt.ylabel('Profit ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cost_vs_profit_trend.png'))
    
    return df[['Product_ID', 'Product_Name', 'Unit_Cost', 'Profit']].sort_values(by='Profit', ascending=False)

def analyze_outlier_detection(df):
    print("Performing outlier detection analysis...")
    
    # Create a copy to avoid modifying original DataFrame
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    
    # Remove ID columns and other non-meaningful columns for outlier detection
    cols_to_exclude = [col for col in df_numeric.columns if 'ID' in col or 'Code' in col]
    df_numeric = df_numeric.drop(columns=cols_to_exclude, errors='ignore')
    
    # Z-score calculation for each numeric column
    z_scores = df_numeric.apply(lambda x: (x - x.mean()) / x.std())
    outliers = (abs(z_scores) > 3).any(axis=1)
    
    outlier_data = df[outliers].copy()
    
    # Create visualizations
    for column in ['Unit_Price', 'Stock_On_Hand', 'Quantity_Sold', 'Total_Revenue']:
        if column in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(y=df[column])
            plt.title(f'Outlier Detection: {column}')
            plt.ylabel(column)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f'outliers_{column.lower()}.png'))
    
    # Save outlier report
    if not outlier_data.empty:
        outlier_data.to_csv(os.path.join(REPORTS_DIR, 'outliers.csv'), index=False)
    
    return outlier_data

def analyze_profit_contribution(df):
    print("Analyzing profit contribution...")
    
    # Check if we have cost data to calculate profit
    if 'Unit_Cost' not in df.columns:
        print("Unit_Cost column not available. Trying to use profit margin if available.")
        if 'Profit_Margin_%' in df.columns:
            df['Profit'] = df['Total_Revenue'] * (df['Profit_Margin_%'] / 100)
        else:
            print("No profit data available. Skipping profit contribution analysis.")
            return None
    else:
        # Calculate profit
        df['Profit'] = df['Total_Revenue'] - (df['Unit_Cost'] * df['Quantity_Sold'])
    
    # Group by product category
    category_profit = df.groupby('Product_Category')['Profit'].sum().reset_index()
    total_profit = category_profit['Profit'].sum()
    category_profit['Contribution_%'] = (category_profit['Profit'] / total_profit) * 100
    category_profit = category_profit.sort_values(by='Contribution_%', ascending=False)
    
    # Create pie chart
    plt.figure(figsize=(12, 8))
    plt.pie(category_profit['Profit'], labels=category_profit['Product_Category'], 
            autopct='%1.1f%%', startangle=90, shadow=True)
    plt.axis('equal')
    plt.title('Profit Contribution by Product Category')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'profit_contribution_pie.png'))
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    sns.barplot(data=category_profit, x='Product_Category', y='Profit')
    plt.title('Profit by Product Category')
    plt.xlabel('Product Category')
    plt.ylabel('Profit ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'profit_by_category.png'))
    
    return category_profit

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
    expiry_risk_2025 = analyze_2025_expiry_risk(df)
    
    # Additional analyses found in notebook1
    supplier_sales = analyze_supplier_sales(df)
    warehouse_distribution = analyze_warehouse_distribution(df)
    supplier_products = analyze_supplier_product_count(df)
    
    # New analyses 
    price_stats = analyze_product_price_distribution(df)
    monthly_trends = analyze_monthly_sales_trends(df)
    inventory_sales_corr = analyze_inventory_vs_sales(df)
    cost_profit_analysis = analyze_cost_vs_profit(df)
    outliers = analyze_outlier_detection(df)
    profit_contribution = analyze_profit_contribution(df)
    
    # Optional analyses
    discount_impact = analyze_discount_impact(df)
    
    # Calculate inventory efficiency metrics
    inventory_value = (df['Stock_On_Hand'] * df['Unit_Price']).sum()
    total_revenue = df['Total_Revenue'].sum()
    avg_inventory_turnover = df['Inventory_Turnover_Ratio'].mean()
    
    print(f"Total Inventory Value: ${inventory_value:.2f}")
    print(f"Total Revenue: ${total_revenue:.2f}")
    print(f"Average Inventory Turnover Ratio: {avg_inventory_turnover:.2f}")
    
    # Generate executive summary
    insights = generate_executive_summary(df, overstocked, top_revenue, category_revenue, expiry_risk)
    
    print("\nAnalysis complete! Results saved to:")
    print(f"- Cleaned dataset: {os.path.join(OUTPUT_DIR, 'cleaned_dataset.csv')}")
    print(f"- Reports: {REPORTS_DIR}")
    print(f"- Visualizations: {PLOTS_DIR}")
    print(f"- Executive summary: {os.path.join(REPORTS_DIR, 'executive_summary.md')}")

if __name__ == "__main__":
    main()
