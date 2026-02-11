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

conn.commit()
conn.close()