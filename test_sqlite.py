#!/usr/bin/env python3
import sqlite3

# Connect to the database (creates the file if it doesn't exist)
conn = sqlite3.connect('test_sql.db')

# Create a cursor object to execute SQL commands
cursor = conn.cursor()

# Create a table (for example, a simple users table)
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL,
    age INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

# Commit the changes
conn.commit()

# Example of inserting some data
try:
    # Insert sample users
    cursor.executemany('''
    INSERT INTO users (username, email, age) VALUES (?, ?, ?)
    ''', [
        ('johndoe', 'john@example.com', 30),
        ('janedoe', 'jane@example.com', 25)
    ])
    
    # Commit the insertions
    conn.commit()
    
    print("Database and table created successfully!")
    
    # Demonstrate reading the data
    print("\nUsers in the database:")
    cursor.execute('SELECT * FROM users')
    for row in cursor.fetchall():
        print(row)

except sqlite3.Error as e:
    print(f"An error occurred: {e}")

finally:
    # Always close the connection
    conn.close()
