import sqlite3
from datetime import datetime

DB_NAME = "attendance.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_attendance(name):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO attendance (name, timestamp) VALUES (?, ?)', (name, timestamp))
    conn.commit()
    conn.close()
    print(f"Logged {name} at {timestamp}")