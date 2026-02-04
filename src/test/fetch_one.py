import os
import requests

URL = "https://www.escapeoutdoors.com/canada-goose-womens-cypress-jacket/?sku=2236L-63-S&color=Navy"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def main():
    os.makedirs("data/raw", exist_ok=True)

    r = requests.get(URL, headers=HEADERS, timeout=30)
    print("status:", r.status_code)
    print("final url:", r.url)
    print("bytes:", len(r.text))

    out_path = "data/raw/escapeoutdoors_one.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(r.text)
    print("saved:", out_path)

if __name__ == "__main__":
    main()
