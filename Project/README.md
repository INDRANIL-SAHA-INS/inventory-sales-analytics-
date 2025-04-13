# Grocery Inventory and Sales Analysis

This project analyzes a grocery inventory and sales dataset to generate insights that optimize inventory management, sales performance, and profit margins.

## Project Structure

```
Project/
├── data/
│   └── Grocery_Inventory_and_Sales_Dataset.csv
├── outputs/
│   ├── plots/              # Visualization outputs
│   ├── reports/            # Analysis reports
│   └── cleaned_dataset.csv # Processed dataset
├── Data_analysis.py        # Main analysis script
├── analysis_notebook.ipynb # Jupyter notebook with analysis
└── README.md               # This file
```

## Requirements

Required packages are listed in `requirement.txt` in the root directory:
- pandas
- numpy
- matplotlib
- seaborn
- jupyter (optional, for running the notebook)

To install the requirements:
```
pip install -r requirement.txt
```

## How to Run

### Option 1: Run the Python Script

```
cd Project
python Data_analysis.py
```

### Option 2: Run the Jupyter Notebook

```
cd Project
jupyter notebook analysis_notebook.ipynb
```

## Analysis Tasks

The analysis covers the following key aspects:

1. **Data Validation and Preprocessing**
   - Handle missing values
   - Correct data types
   - Remove outliers

2. **Priority Analysis**
   - Identify overstocked products
   - Find top products by quantity sold and revenue
   - Analyze category-wise revenue
   - Evaluate expiry risk

3. **Optional Analysis**
   - Discount impact analysis

## Output

The analysis generates the following outputs:

1. **Cleaned Dataset**
   - `outputs/cleaned_dataset.csv`: Processed and validated dataset

2. **Reports**
   - `reports/overstocked_products.csv`: List of overstocked products
   - `reports/top_products_by_quantity.csv`: Top products by quantity sold
   - `reports/top_products_by_revenue.csv`: Top products by revenue
   - `reports/category_revenue.csv`: Revenue analysis by category
   - `reports/expiry_risk.csv`: Products at risk of expiring soon
   - `reports/executive_summary.md`: Key business insights and recommendations

3. **Visualizations**
   - Bar charts, pie charts, and other visualizations saved in `outputs/plots/` 