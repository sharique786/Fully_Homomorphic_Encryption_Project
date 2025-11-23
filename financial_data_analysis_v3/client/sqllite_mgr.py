import sqlite3
import threading
from threading import Lock

class ThreadSafeSQLiteManager:
    """Thread-safe SQLite operations for parallel encryption"""
    
    def __init__(self, conn):
        self.conn = conn
        self.write_lock = threading.Lock()
    
    def insert_encrypted_record(self, record_data: dict):
        """Thread-safe insert of encrypted record"""
        with self.write_lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO encrypted_data 
                (id, party_id, email_id, account_id, transaction_id, column_name, 
                 encrypted_value, original_value, data_type, transaction_date, batch_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record_data['id'],
                record_data['party_id'],
                record_data['email_id'],
                record_data['account_id'],
                record_data['transaction_id'],
                record_data['column_name'],
                record_data['encrypted_value'],
                record_data['original_value'],
                record_data['data_type'],
                record_data['transaction_date'],
                record_data['batch_id']
            ))
            self.conn.commit()
    
    def insert_batch_records(self, records: list):
        """Batch insert for better performance"""
        with self.write_lock:
            cursor = self.conn.cursor()
            cursor.executemany('''
                INSERT INTO encrypted_data 
                (id, party_id, email_id, account_id, transaction_id, column_name, 
                 encrypted_value, original_value, data_type, transaction_date, batch_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)
            self.conn.commit()