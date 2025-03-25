import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from typing import List, Dict, Any

class PersonalFinanceTracker:
    def __init__(self, db_path: str = 'finance_tracker.db'):
        """
        Initialize the finance tracker with database connection and setup.
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary database tables if they don't exist."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY,
                date TEXT,
                category TEXT,
                amount REAL,
                description TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS budget (
                category TEXT PRIMARY KEY,
                monthly_limit REAL
            )
        ''')
        self.conn.commit()
    
    def add_transaction(self, date: str, category: str, amount: float, description: str = ''):
        """
        Add a new financial transaction to the database.
        
        Args:
            date (str): Transaction date (YYYY-MM-DD)
            category (str): Transaction category
            amount (float): Transaction amount
            description (str): Optional transaction description
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO transactions (date, category, amount, description)
            VALUES (?, ?, ?, ?)
        ''', (date, category, amount, description))
        self.conn.commit()
    
    def set_budget(self, category: str, monthly_limit: float):
        """
        Set a monthly budget for a specific category.
        
        Args:
            category (str): Budget category
            monthly_limit (float): Maximum allowed spending
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO budget (category, monthly_limit)
            VALUES (?, ?)
        ''', (category, monthly_limit))
        self.conn.commit()
    
    def get_monthly_spending(self, year: int, month: int) -> pd.DataFrame:
        """
        Retrieve monthly spending data.
        
        Args:
            year (int): Year of spending
            month (int): Month of spending
        
        Returns:
            pd.DataFrame: Monthly spending by category
        """
        query = f'''
            SELECT category, SUM(amount) as total_spent
            FROM transactions
            WHERE strftime('%Y', date) = '{year}' AND strftime('%m', date) = '{month:02d}'
            GROUP BY category
        '''
        return pd.read_sql_query(query, self.conn)
    
    def analyze_spending_trends(self) -> Dict[str, Any]:
        """
        Analyze overall spending trends and provide insights.
        
        Returns:
            Dict containing spending analysis results
        """
        df = pd.read_sql_query('SELECT * FROM transactions', self.conn)
        df['date'] = pd.to_datetime(df['date'])
        
        monthly_trends = df.groupby([df['date'].dt.to_period('M'), 'category'])['amount'].sum().unstack()
        
        insights = {
            'total_spending': df['amount'].sum(),
            'average_monthly_spending': df.groupby(df['date'].dt.to_period('M'))['amount'].sum().mean(),
            'top_categories': df.groupby('category')['amount'].sum().nlargest(3),
            'monthly_trends': monthly_trends
        }
        return insights
    
    def predict_future_expenses(self, category: str, months_ahead: int = 3) -> List[float]:
        """
        Use machine learning to predict future expenses for a category.
        
        Args:
            category (str): Expense category to predict
            months_ahead (int): Number of months to predict
        
        Returns:
            List of predicted expense amounts
        """
        df = pd.read_sql_query('SELECT * FROM transactions', self.conn)
        df['date'] = pd.to_datetime(df['date'])
        
        category_df = df[df['category'] == category].copy()
        category_df['month'] = category_df['date'].dt.to_period('M')
        monthly_expenses = category_df.groupby('month')['amount'].sum().reset_index()
        
        monthly_expenses['month_num'] = range(len(monthly_expenses))
        
        X = monthly_expenses[['month_num']]
        y = monthly_expenses['amount']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        
        last_month = monthly_expenses['month_num'].max()
        future_months = np.array([[last_month + i + 1] for i in range(months_ahead)])
        future_months_scaled = scaler.transform(future_months)
        
        predictions = rf.predict(future_months_scaled)
        return list(predictions)
    
    def visualize_spending(self):
        """Create a comprehensive visualization of spending patterns."""
        insights = self.analyze_spending_trends()
        
        plt.figure(figsize=(15, 10))
        
        # Spending by Category Pie Chart
        plt.subplot(2, 2, 1)
        insights['top_categories'].plot(kind='pie', autopct='%1.1f%%')
        plt.title('Spending by Category')
        
        # Monthly Trends Line Plot
        plt.subplot(2, 2, 2)
        insights['monthly_trends'].plot(kind='line', marker='o')
        plt.title('Monthly Spending Trends')
        plt.xticks(rotation=45)
        
        # Budget vs Actual Comparison
        plt.subplot(2, 2, 3)
        budget_df = pd.read_sql_query('SELECT * FROM budget', self.conn)
        monthly_spending = self.get_monthly_spending(datetime.now().year, datetime.now().month)
        merged = pd.merge(budget_df, monthly_spending, on='category', how='left')
        merged['total_spent'] = merged['total_spent'].fillna(0)
        merged.plot(x='category', y=['monthly_limit', 'total_spent'], kind='bar')
        plt.title('Budget vs Actual Spending')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('spending_analysis.png')
        plt.close()
    
    def export_report(self, output_file: str = 'finance_report.csv'):
        """
        Export a comprehensive financial report.
        
        Args:
            output_file (str): Path to export CSV report
        """
        df = pd.read_sql_query('SELECT * FROM transactions', self.conn)
        df.to_csv(output_file, index=False)
    
    def close(self):
        """Close database connection."""
        self.conn.close()

def main():
    # Example usage demonstrating full functionality
    tracker = PersonalFinanceTracker()
    
    # Add sample transactions
    transactions = [
        ('2024-01-15', 'Groceries', 150.50, 'Monthly grocery shopping'),
        ('2024-01-20', 'Dining', 75.25, 'Restaurant dinner'),
        ('2024-01-22', 'Transportation', 50.00, 'Monthly transit pass'),
        ('2024-02-05', 'Groceries', 165.75, 'Monthly grocery shopping'),
        ('2024-02-10', 'Entertainment', 100.00, 'Movie and dinner'),
    ]
    
    for transaction in transactions:
        tracker.add_transaction(*transaction)
    
    # Set budget categories
    budget_categories = {
        'Groceries': 200.00,
        'Dining': 150.00,
        'Transportation': 100.00,
        'Entertainment': 200.00
    }
    
    for category, limit in budget_categories.items():
        tracker.set_budget(category, limit)
    
    # Analyze and visualize spending
    insights = tracker.analyze_spending_trends()
    print("Spending Insights:")
    for key, value in insights.items():
        print(f"{key}: {value}")
    
    # Predict future expenses for a category
    predictions = tracker.predict_future_expenses('Groceries')
    print("\nGrocery Expense Predictions:", predictions)
    
    # Generate visualization
    tracker.visualize_spending()
    
    # Export report
    tracker.export_report()
    
    tracker.close()

if __name__ == '__main__':
    main()
