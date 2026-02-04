import json
import re
import html
import requests
from bs4 import BeautifulSoup

URL = "https://www.escapeoutdoors.com/canada-goose-womens-cypress-jacket/?sku=2236L-63-S&color=Navy"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def clean_text(s: str) -> str:
    # decode HTML entities (&rsquo; etc) and collapse whitespace
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    r = requests.get(URL, headers=HEADERS, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")

    product_json = None

    # There can be multiple ld+json scripts. Find the one with "@type": "Product".
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = (tag.string or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        # Sometimes it's a list; sometimes it's a dict
        candidates = data if isinstance(data, list) else [data]
        for obj in candidates:
            if isinstance(obj, dict) and obj.get("@type") == "Product":
                product_json = obj
                break
        if product_json:
            break

    if not product_json:
        print("No Product ld+json found.")
        return

    name = product_json.get("name")
    sku = product_json.get("sku")
    brand = (product_json.get("brand") or {}).get("name")
    url = product_json.get("url")
    description = product_json.get("description")

    offers = product_json.get("offers") or {}
    price = offers.get("price")
    currency = offers.get("priceCurrency")
    availability = offers.get("availability")

    # Print extracted fields (first success milestone)
    print("name:", name)
    print("brand:", brand)
    print("sku:", sku)
    print("price:", price, currency)
    print("availability:", availability)
    print("url:", url)
    print("description (first 120 chars):", clean_text(description)[:120] if description else None)

if __name__ == "__main__":
    main()
