import os
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

import re
from urllib.parse import urlparse

BASE_URL = "https://www.escapeoutdoors.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
DELAY_SECONDS = 2
OUT_PATH = "data/extracted/product_urls.txt"


def same_domain(url: str) -> bool:
    return urlparse(url).netloc == urlparse(BASE_URL).netloc

PRODUCT_PATH_RE = re.compile(r"^/canada-goose-[^/]+/$", re.IGNORECASE)

def is_valid_product_url(url: str) -> bool:
    p = urlparse(url)

    if p.netloc != urlparse(BASE_URL).netloc:
        return False

    if not PRODUCT_PATH_RE.match(p.path):
        return False

    if p.query:
        return False

    if p.path.lower().startswith("/brands/"):
        return False

    return True

def extract_product_urls(html: str) -> set[str]:
    soup = BeautifulSoup(html, "lxml")
    urls: set[str] = set()

    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue

        full = urljoin(BASE_URL, href)
        if not same_domain(full):
            continue

        if "/canada-goose-" in full or "canada-goose" in full:
            if is_valid_product_url(full):
                urls.add(full)

    return urls


def get_all_product_urls(session: requests.Session, max_pages: int = 18) -> list[str]:
    all_urls: set[str] = set()

    for page in range(1, max_pages + 1):
        if page == 1:
            url = "https://www.escapeoutdoors.com/canada-goose/?sort=bestselling"
        else:
            url = f"https://www.escapeoutdoors.com/canada-goose/?sort=bestselling&page={page}"

        print(f"Fetching page {page}: {url}")
        r = session.get(url, timeout=30)
        r.raise_for_status()

        urls = extract_product_urls(r.text)
        print(f"  found {len(urls)} urls")

        if not urls:
            print("No products found, stopping early.")
            break

        all_urls |= urls
        time.sleep(DELAY_SECONDS)

    return sorted(all_urls)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    urls = get_all_product_urls(session, max_pages=8)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for u in urls:
            f.write(u + "\n")

    print(f"\nDONE. Saved {len(urls)} URLs to {OUT_PATH}")


if __name__ == "__main__":
    main()

