import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


class FinancialDataGenerator:
    """Generate synthetic financial data for testing"""

    @staticmethod
    def generate_user_data(num_users=100):
        """Generate user details"""
        users = []
        for i in range(1, num_users + 1):
            user = {
                'user_id': f'USR{i:05d}',
                'name': f'User {i}',
                'email': f'user{i}@example.com',
                'address': f'{random.randint(100, 999)} Main St, City {i % 10}',
                'phone': f'+1{random.randint(1000000000, 9999999999)}',
                'age': random.randint(18, 80),
                'registration_date': (datetime.now() - timedelta(days=random.randint(0, 3650))).strftime('%Y-%m-%d')
            }
            users.append(user)
        return pd.DataFrame(users)

    @staticmethod
    def generate_account_data(user_ids, accounts_per_user=2):
        """Generate account details"""
        accounts = []
        account_types = ['Savings', 'Checking', 'Credit', 'Investment']

        account_id = 1
        for user_id in user_ids:
            num_accounts = random.randint(1, accounts_per_user)
            for _ in range(num_accounts):
                account = {
                    'account_id': f'ACC{account_id:07d}',
                    'user_id': user_id,
                    'account_type': random.choice(account_types),
                    'account_number': f'{random.randint(1000000000, 9999999999)}',
                    'balance': round(random.uniform(100, 50000), 2),
                    'currency': random.choice(['USD', 'EUR', 'GBP', 'JPY']),
                    'opening_date': (datetime.now() - timedelta(days=random.randint(0, 2000))).strftime('%Y-%m-%d'),
                    'status': random.choice(['Active', 'Active', 'Active', 'Frozen'])
                }
                accounts.append(account)
                account_id += 1
        return pd.DataFrame(accounts)

    @staticmethod
    def generate_transaction_data(account_data, transactions_per_account=50):
        """Generate transaction details"""
        transactions = []
        transaction_types = ['Deposit', 'Withdrawal', 'Transfer', 'Payment', 'Purchase']

        transaction_id = 1
        for _, account in account_data.iterrows():
            num_transactions = random.randint(10, transactions_per_account)
            for _ in range(num_transactions):
                transaction = {
                    'transaction_id': f'TXN{transaction_id:010d}',
                    'user_id': account['user_id'],
                    'account_id': account['account_id'],
                    'transaction_type': random.choice(transaction_types),
                    'amount': round(random.uniform(10, 5000), 2),
                    'currency': account['currency'],
                    'date': (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d'),
                    'time': f'{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}',
                    'description': f'Transaction {transaction_id}',
                    'status': random.choice(['Completed', 'Completed', 'Completed', 'Pending', 'Failed'])
                }
                transactions.append(transaction)
                transaction_id += 1
        return pd.DataFrame(transactions)

    @staticmethod
    def generate_complete_dataset(num_users=100, accounts_per_user=2, transactions_per_account=50):
        """Generate complete financial dataset"""
        user_data = FinancialDataGenerator.generate_user_data(num_users)
        account_data = FinancialDataGenerator.generate_account_data(user_data['user_id'].tolist(), accounts_per_user)
        transaction_data = FinancialDataGenerator.generate_transaction_data(account_data, transactions_per_account)

        return user_data, account_data, transaction_data