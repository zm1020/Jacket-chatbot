# This file craetes the database and tables for the project, including the keywords table and procduct table
import sqlite3

conn = sqlite3.connect("data/canada_goose.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    brand TEXT,
    name TEXT,
    gender TEXT,
    price REAL,
    currency TEXT,
    availability TEXT,
    sku TEXT UNIQUE,
    description TEXT,
    url TEXT,
    image_url TEXT
);
""")

# keyword table + indexes
cur.executescript("""
CREATE TABLE IF NOT EXISTS product_keywords (
    product_id INTEGER NOT NULL,
    keyword TEXT NOT NULL,
    PRIMARY KEY (product_id, keyword),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

CREATE INDEX IF NOT EXISTS idx_keyword ON product_keywords(keyword);
CREATE INDEX IF NOT EXISTS idx_product_id ON product_keywords(product_id);
""")
conn.commit()
conn.close()

'''
conn = sqlite3.connect("data/canada_goose.db")
cur = conn.cursor()

try:
    cur.execute("ALTER TABLE products ADD COLUMN tei_level INTEGER;")
    conn.commit()
    print("✅ Added column: products.tei_level")
except sqlite3.OperationalError as e:
    # This happens if the column already exists
    print("ℹ️ Could not add column (maybe it already exists):", e)

conn.close()
'''