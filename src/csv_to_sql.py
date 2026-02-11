# this file convert csv file to sql file
import csv
import sqlite3

import csv
import sqlite3

conn = sqlite3.connect("data/canada_goose.db")
cur = conn.cursor()

with open("data/extracted/products.csv", newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cur.execute("""
            INSERT OR IGNORE INTO products
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["id"],
            row["brand"],
            row["name"],
            row["gender"],
            float(row["price"]),
            row["currency"],
            row["availability"],
            row["sku"],
            row["description"],
            row["url"],
            row["image_url"]
        ))

conn.commit()
conn.close()

