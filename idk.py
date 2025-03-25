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
        """Initialize the finance tracker with database connection and setup."""
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    category TEXT NOT NULL,
                    amount REAL NOT NULL,
                    description TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS budget (
                    category TEXT PRIMARY KEY,
                    monthly_limit REAL NOT NULL
                )
            ''')
            conn.commit()

    def add_transaction(self, date: str, category: str, amount: float, description: str = ''):
        """Add a new financial transaction to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO transactions (date, category, amount, description)
                VALUES (?, ?, ?, ?)
            ''', (date, category, amount, description))
            conn.commit()
    
    def set_budget(self, category: str, monthly_limit: float):
        """Set a monthly budget for a specific category."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO budget (category, monthly_limit)
                VALUES (?, ?)
            ''', (category, monthly_limit))
            conn.commit()

    def get_monthly_spending(self, year: int, month: int) -> pd.DataFrame:
        """Retrieve monthly spending data."""
        query = '''
            SELECT category, COALESCE(SUM(amount), 0) AS total_spent
            FROM transactions
            WHERE strftime('%Y', date) = ? AND strftime('%m', date) = ?
            GROUP BY category
        '''
        return pd.read_sql_query(query, sqlite3.connect(self.db_path), params=(str(year), f"{month:02d}"))

    def analyze_spending_trends(self) -> Dict[str, Any]:
        """Analyze spending trends and return insights."""
        df = pd.read_sql_query('SELECT * FROM transactions', sqlite3.connect(self.db_path))
        if df.empty:
            return {"message": "No transactions available for analysis."}
        
        df['date'] = pd.to_datetime(df['date'])
        monthly_trends = df.groupby([df['date'].dt.to_period('M'), 'category'])['amount'].sum().unstack()
        
        return {
            'total_spending': df['amount'].sum(),
            'average_monthly_spending': df.groupby(df['date'].dt.to_period('M'))['amount'].sum().mean(),
            'top_categories': df.groupby('category')['amount'].sum().nlargest(3),
            'monthly_trends': monthly_trends
        }

    def predict_future_expenses(self, category: str, months_ahead: int = 3) -> List[float]:
        """Predict future expenses using machine learning."""
        df = pd.read_sql_query('SELECT * FROM transactions', sqlite3.connect(self.db_path))
        df['date'] = pd.to_datetime(df['date'])
        
        category_df = df[df['category'] == category].copy()
        if category_df.empty:
            return [0] * months_ahead
        
        category_df['month'] = category_df['date'].dt.to_period('M')
        monthly_expenses = category_df.groupby('month')['amount'].sum().reset_index()
        monthly_expenses['month_num'] = range(len(monthly_expenses))
        
        X = monthly_expenses[['month_num']]
        y = monthly_expenses['amount']

        if len(X) < 2:  
            return [y.mean() if not y.empty else 0] * months_ahead

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train_scaled, X_test_scaled = train_test_split(X_scaled, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        last_month = monthly_expenses['month_num'].max()
        future_months = np.array([[last_month + i + 1] for i in range(months_ahead)])
        future_months_scaled = scaler.transform(future_months)

        return list(model.predict(future_months_scaled))

    def visualize_spending(self):
        """Create and save visualizations of spending patterns."""
        insights = self.analyze_spending_trends()
        if 'message' in insights:
            print(insights['message'])
            return
        
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        insights['top_categories'].plot(kind='pie', autopct='%1.1f%%')
        plt.title('Spending by Category')

        plt.subplot(2, 2, 2)
        insights['monthly_trends'].plot(kind='line', marker='o')
        plt.title('Monthly Spending Trends')
        plt.xticks(rotation=45)

        plt.subplot(2, 2, 3)
        budget_df = pd.read_sql_query('SELECT * FROM budget', sqlite3.connect(self.db_path))
        monthly_spending = self.get_monthly_spending(datetime.now().year, datetime.now().month)
        merged = pd.merge(budget_df, monthly_spending, on='category', how='left').fillna(0)
        merged.plot(x='category', y=['monthly_limit', 'total_spent'], kind='bar')
        plt.title('Budget vs Actual Spending')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig('spending_analysis.png')
        plt.close()

    def export_report(self, output_file: str = 'finance_report.csv'):
        """Export financial transactions as a CSV report."""
        df = pd.read_sql_query('SELECT * FROM transactions', sqlite3.connect(self.db_path))
        df.to_csv(output_file, index=False)

    def close(self):
        """Close the database connection."""
        self.conn.close()
    
    def __del__(self):
        """Ensure the database connection is closed on deletion."""
        self.close()

def main():
    tracker = PersonalFinanceTracker()

    transactions = [
        ('2024-01-15', 'Groceries', 150.50, 'Monthly grocery shopping'),
        ('2024-01-20', 'Dining', 75.25, 'Restaurant dinner'),
        ('2024-01-22', 'Transportation', 50.00, 'Monthly transit pass'),
        ('2024-02-05', 'Groceries', 165.75, 'Monthly grocery shopping'),
        ('2024-02-10', 'Entertainment', 100.00, 'Movie and dinner'),
    ]
    for transaction in transactions:
        tracker.add_transaction(*transaction)

    budget = {'Groceries': 200, 'Dining': 150, 'Transportation': 100, 'Entertainment': 200}
    for category, limit in budget.items():
        tracker.set_budget(category, limit)

    print("Spending Insights:", tracker.analyze_spending_trends())
    print("\nPredicted Future Expenses:", tracker.predict_future_expenses('Groceries'))
    
    tracker.visualize_spending()
    tracker.export_report()
    tracker.close()

if __name__ == '__main__':
    main()
