# db_utils.py
import sqlite3
from datetime import datetime

# Initialize the inventory and logs tables
def init_db():
    conn = sqlite3.connect("inventory.db")
    c = conn.cursor()

    # Product table
    c.execute('''CREATE TABLE IF NOT EXISTS products (
        product_id TEXT PRIMARY KEY,
        name TEXT,
        stock INTEGER,
        category TEXT
    )''')

    # Logs table
    c.execute('''CREATE TABLE IF NOT EXISTS logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id TEXT,
        in_time TEXT,
        out_time TEXT
    )''')

    conn.commit()
    conn.close()


# Get current stock for a product
def get_stock(product_name):
    conn = sqlite3.connect("inventory.db")
    c = conn.cursor()
    c.execute("SELECT stock FROM products WHERE name = ?", (product_name,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else 0


# Add or remove item from inventory, and log the time
def update_inventory(product_name, direction):
    conn = sqlite3.connect("inventory.db")
    c = conn.cursor()

    # Get product_id using product name
    c.execute("SELECT product_id FROM products WHERE name = ?", (product_name,))
    row = c.fetchone()

    if row:
        product_id = row[0]
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if direction == "in":
            c.execute("UPDATE products SET stock = stock + 1 WHERE product_id = ?", (product_id,))
            c.execute("INSERT INTO logs (product_id, in_time) VALUES (?, ?)", (product_id, now))
        else:
            c.execute("UPDATE products SET stock = stock - 1 WHERE product_id = ?", (product_id,))
            c.execute("INSERT INTO logs (product_id, out_time) VALUES (?, ?)", (product_id, now))

        conn.commit()

    conn.close()


# Get all products and stock
def get_all_products():
    conn = sqlite3.connect("inventory.db")
    c = conn.cursor()
    c.execute("SELECT * FROM products")
    products = c.fetchall()
    conn.close()
    return products


# Get product logs
def get_logs():
    conn = sqlite3.connect("inventory.db")
    c = conn.cursor()
    c.execute("SELECT * FROM logs ORDER BY log_id DESC")
    logs = c.fetchall()
    conn.close()
    return logs
