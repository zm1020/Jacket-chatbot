import csv
import html
import json
import os
import re
import time

import requests
from bs4 import BeautifulSoup

URLS_PATH = "data/extracted/product_urls.txt"
OUT_CSV = "data/extracted/products.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

DELAY_SECONDS = 2

FIELDNAMES = [
    "id",
    "brand",
    "name",
    "gender",
    "price",
    "currency",
    "availability",
    "sku",
    "description",
    "url",
    "image_url",
]


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def infer_gender(name: str, url: str) -> str:
    s = f"{name or ''} {url or ''}".lower()
    if "women" in s or "womens" in s or "women's" in s:
        return "Women"
    if "men" in s or "mens" in s or "men's" in s:
        return "Men"
    return "Unknown"


def parse_product_from_html(page_html: str, url: str) -> dict | None:
    soup = BeautifulSoup(page_html, "lxml")

    # Find ld+json Product block
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = (tag.string or "").strip()
        if not raw:
            continue

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if isinstance(data, dict) and data.get("@type") == "Product":
            offers = data.get("offers") or {}
            name = data.get("name")

            brand_obj = data.get("brand") or {}
            brand = brand_obj.get("name") if isinstance(brand_obj, dict) else str(brand_obj)

            return {
                "brand": brand,
                "name": name,
                "gender": infer_gender(name, url),
                "price": offers.get("price"),
                "currency": offers.get("priceCurrency"),
                "availability": offers.get("availability"),
                "sku": data.get("sku"),
                "description": clean_text(data.get("description")),
                "url": url,
                "image_url": data.get("image"),
            }

    return None


def load_urls(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_rows_csv(rows: list[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    urls = load_urls(URLS_PATH)
    print(f"Loaded {len(urls)} URLs")

    session = requests.Session()
    session.headers.update(HEADERS)

    rows = []
    pid = 1

    for i, url in enumerate(urls, 1):
        try:
            print(f"[{i}/{len(urls)}] fetching: {url}")
            r = session.get(url, timeout=30)
            r.raise_for_status()

            product = parse_product_from_html(r.text, url)
            if not product:
                print("  -> no Product ld+json found, skipping")
            else:
                product["id"] = pid
                rows.append(product)
                print(f"  -> saved: {product['name']}")
                pid += 1

        except Exception as e:
            print("  -> failed:", repr(e))

        time.sleep(DELAY_SECONDS)

    write_rows_csv(rows, OUT_CSV)
    print(f"\nDONE. Wrote {len(rows)} products to {OUT_CSV}")


if __name__ == "__main__":
    main()
