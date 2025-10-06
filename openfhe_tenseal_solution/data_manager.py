"""
Data Manager Module
Handles data loading, validation, generation, and preprocessing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random
import io

from config import FINANCIAL_DATA_SCHEMA, CURRENCY_OPTIONS, TRANSACTION_CATEGORIES, SAMPLE_DATA_CONFIG


class DataManager:
    """Manages financial data loading, generation, and validation"""

    def __init__(self):
        self.user_details = None
        self.account_details = None
        self.transaction_details = None
        self.raw_data = {}
        self.validated = False

    def generate_sample_data(self, num_users: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Generate sample financial data for demonstration

        Args:
            num_users: Number of users to generate

        Returns:
            Dictionary containing three dataframes: users, accounts, transactions
        """
        # Generate User Details
        users_data = []
        for i in range(num_users):
            users_data.append({
                'user_id': f'USR_{i:05d}',
                'name': f'User {i}',
                'email': f'user{i}@example.com',
                'phone': f'+1-555-{random.randint(1000, 9999)}',
                'address': f'{random.randint(1, 9999)} {random.choice(["Main St", "Oak Ave", "Pine Rd", "Elm Dr"])}',
                'country': random.choice(['USA', 'UK', 'Germany', 'France', 'Canada']),
                'date_of_birth': (datetime.now() - timedelta(days=random.randint(7000, 25000))).date()
            })

        self.user_details = pd.DataFrame(users_data)

        # Generate Account Details
        accounts_data = []
        account_id = 0
        for user_id in self.user_details['user_id']:
            num_accounts = random.randint(1, 3)
            for _ in range(num_accounts):
                accounts_data.append({
                    'account_id': f'ACC_{account_id:06d}',
                    'user_id': user_id,
                    'account_number': f'{random.randint(100000000, 999999999)}',
                    'account_type': random.choice(['Savings', 'Checking', 'Credit Card', 'Investment']),
                    'balance': round(random.uniform(100, 50000), 2),
                    'opening_date': (datetime.now() - timedelta(days=random.randint(30, 1800))).date(),
                    'status': random.choice(['Active', 'Active', 'Active', 'Frozen']),
                    'branch': f'Branch-{random.randint(1, 50)}'
                })
                account_id += 1

        self.account_details = pd.DataFrame(accounts_data)

        # Generate Transaction Details
        transactions_data = []
        transaction_id = 0
        for _, account in self.account_details.iterrows():
            num_transactions = random.randint(10, 50)
            for _ in range(num_transactions):
                trans_date = datetime.now() - timedelta(days=random.randint(0, 365))
                transactions_data.append({
                    'transaction_id': f'TXN_{transaction_id:08d}',
                    'user_id': account['user_id'],
                    'account_id': account['account_id'],
                    'account_number': account['account_number'],
                    'amount': round(random.uniform(10, 5000), 2),
                    'currency': random.choice(CURRENCY_OPTIONS[:5]),  # Top 5 currencies
                    'transaction_date': trans_date.date(),
                    'transaction_time': trans_date.time(),
                    'merchant': f'Merchant-{random.randint(1, 200)}',
                    'category': random.choice(TRANSACTION_CATEGORIES),
                    'status': random.choice(['Completed', 'Completed', 'Completed', 'Pending', 'Failed']),
                    'transaction_type': random.choice(['Debit', 'Debit', 'Credit'])
                })
                transaction_id += 1

        self.transaction_details = pd.DataFrame(transactions_data)

        self.validated = True

        return {
            'users': self.user_details,
            'accounts': self.account_details,
            'transactions': self.transaction_details
        }

    def load_csv_files(self, files: Dict[str, any]) -> Dict[str, pd.DataFrame]:
        """
        Load CSV files uploaded by user

        Args:
            files: Dictionary with file types as keys and file objects as values

        Returns:
            Dictionary of loaded dataframes
        """
        loaded_data = {}

        for file_type, file_obj in files.items():
            try:
                df = pd.read_csv(file_obj)
                loaded_data[file_type] = df

                # Store in instance variables
                if file_type == 'users':
                    self.user_details = df
                elif file_type == 'accounts':
                    self.account_details = df
                elif file_type == 'transactions':
                    self.transaction_details = df

            except Exception as e:
                print(f"Error loading {file_type}: {e}")

        return loaded_data

    def validate_data(self) -> Dict[str, any]:
        """
        Validate loaded data against schema

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # Validate user details
        if self.user_details is not None:
            required_cols = FINANCIAL_DATA_SCHEMA['user_details']['required_columns']
            missing_cols = [col for col in required_cols if col not in self.user_details.columns]
            if missing_cols:
                validation_results['errors'].append(f"User details missing columns: {missing_cols}")
                validation_results['valid'] = False

        # Validate account details
        if self.account_details is not None:
            required_cols = FINANCIAL_DATA_SCHEMA['account_details']['required_columns']
            missing_cols = [col for col in required_cols if col not in self.account_details.columns]
            if missing_cols:
                validation_results['errors'].append(f"Account details missing columns: {missing_cols}")
                validation_results['valid'] = False

        # Validate transaction details
        if self.transaction_details is not None:
            required_cols = FINANCIAL_DATA_SCHEMA['transaction_details']['required_columns']
            missing_cols = [col for col in required_cols if col not in self.transaction_details.columns]
            if missing_cols:
                validation_results['errors'].append(f"Transaction details missing columns: {missing_cols}")
                validation_results['valid'] = False

        # Check referential integrity
        if self.user_details is not None and self.transaction_details is not None:
            invalid_users = set(self.transaction_details['user_id']) - set(self.user_details['user_id'])
            if invalid_users:
                validation_results['warnings'].append(f"Found {len(invalid_users)} transactions with invalid user_ids")

        self.validated = validation_results['valid']
        return validation_results

    def get_data_summary(self) -> Dict[str, any]:
        """Get summary statistics of loaded data"""
        summary = {}

        if self.user_details is not None:
            summary['users'] = {
                'count': len(self.user_details),
                'columns': list(self.user_details.columns),
                'countries': self.user_details['country'].nunique() if 'country' in self.user_details.columns else 0
            }

        if self.account_details is not None:
            summary['accounts'] = {
                'count': len(self.account_details),
                'columns': list(self.account_details.columns),
                'account_types': self.account_details[
                    'account_type'].nunique() if 'account_type' in self.account_details.columns else 0,
                'total_balance': self.account_details[
                    'balance'].sum() if 'balance' in self.account_details.columns else 0
            }

        if self.transaction_details is not None:
            summary['transactions'] = {
                'count': len(self.transaction_details),
                'columns': list(self.transaction_details.columns),
                'total_amount': self.transaction_details[
                    'amount'].sum() if 'amount' in self.transaction_details.columns else 0,
                'currencies': self.transaction_details[
                    'currency'].nunique() if 'currency' in self.transaction_details.columns else 0,
                'date_range': {
                    'start': str(self.transaction_details[
                                     'transaction_date'].min()) if 'transaction_date' in self.transaction_details.columns else None,
                    'end': str(self.transaction_details[
                                   'transaction_date'].max()) if 'transaction_date' in self.transaction_details.columns else None
                }
            }

        return summary

    def filter_transactions_by_date(self, start_date, end_date) -> pd.DataFrame:
        """Filter transactions within date range"""
        if self.transaction_details is None:
            return pd.DataFrame()

        df = self.transaction_details.copy()
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])

        mask = (df['transaction_date'] >= pd.to_datetime(start_date)) & \
               (df['transaction_date'] <= pd.to_datetime(end_date))

        return df[mask]

    def get_user_transactions(self, user_id: str) -> pd.DataFrame:
        """Get all transactions for a specific user"""
        if self.transaction_details is None:
            return pd.DataFrame()

        return self.transaction_details[self.transaction_details['user_id'] == user_id]

    def aggregate_by_currency(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Aggregate transactions by currency"""
        if df is None:
            df = self.transaction_details

        if df is None or df.empty:
            return pd.DataFrame()

        return df.groupby('currency').agg({
            'amount': ['sum', 'mean', 'count'],
            'transaction_id': 'count'
        }).reset_index()

    def get_sensitive_columns(self) -> Dict[str, List[str]]:
        """Identify sensitive columns that should be encrypted"""
        sensitive = {}

        if self.user_details is not None:
            user_sensitive = [col for col in FINANCIAL_DATA_SCHEMA['user_details']['sensitive_columns']
                              if col in self.user_details.columns]
            sensitive['users'] = user_sensitive

        if self.account_details is not None:
            account_sensitive = [col for col in FINANCIAL_DATA_SCHEMA['account_details']['sensitive_columns']
                                 if col in self.account_details.columns]
            sensitive['accounts'] = account_sensitive

        if self.transaction_details is not None:
            trans_sensitive = [col for col in FINANCIAL_DATA_SCHEMA['transaction_details']['sensitive_columns']
                               if col in self.transaction_details.columns]
            sensitive['transactions'] = trans_sensitive

        return sensitive

    def export_to_csv(self, data_type: str, filename: str = None) -> str:
        """Export data to CSV string"""
        df = None

        if data_type == 'users':
            df = self.user_details
        elif data_type == 'accounts':
            df = self.account_details
        elif data_type == 'transactions':
            df = self.transaction_details

        if df is None:
            return ""

        return df.to_csv(index=False)

    def denormalize_data(self) -> pd.DataFrame:
        """
        Combine all three tables into a denormalized view

        Returns:
            Denormalized DataFrame
        """
        if self.transaction_details is None:
            return pd.DataFrame()

        result = self.transaction_details.copy()

        # Join with accounts
        if self.account_details is not None:
            result = result.merge(
                self.account_details[['user_id', 'account_id', 'account_type', 'balance']],
                on=['user_id', 'account_id'],
                how='left',
                suffixes=('', '_account')
            )

        # Join with users
        if self.user_details is not None:
            result = result.merge(
                self.user_details[['user_id', 'name', 'email', 'country']],
                on='user_id',
                how='left',
                suffixes=('', '_user')
            )

        return result