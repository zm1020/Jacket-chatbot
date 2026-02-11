# this file generate keywords for each product in the database
import re
import sqlite3
from typing import Optional, Set, Tuple, List

DB_PATH = "data/canada_goose.db"

# -------------------------
# Tuned keyword config
# -------------------------
STOPWORDS = {
    # common English
    "the","a","an","and","or","but","if","then","else","to","of","in","on","for","with","without",
    "is","are","was","were","be","been","being","this","that","these","those","it","its","they",
    "you","your","from","at","as","by","into","over","under","between","across","about","also",

    # section headers / metadata noise
    "features","feature","origin","disc","trim","fit","length","materials","care",

    # marketing filler
    "introducing","perfect","ideal","versatile","essential","effortless","designed","offers",
    "provides","providing","helps","help","keep","keeping","added","adds","add","allow",
    "allows","allowing","make","makes","today","read","learn","more","proudly",
    "collection","selection","style","silhouette","updated","update","new","most",

    # repetitive apparel noise
    "interior","exterior","front","back","upper","lower","left","right",
    "collar","cuffs","hem","panel","panels","sleeves","underarm","gussets","brim",
}

# normalize variants -> canonical token
NORMALIZE = {
    "water-resistant": "water_resistant",
    "waterresistant": "water_resistant",
    "waterproof": "waterproof",
    "water-repellent": "water_repellent",
    "waterrepellent": "water_repellent",

    "wind-resistant": "wind_resistant",
    "windresistant": "wind_resistant",
    "windproof": "windproof",

    "down-filled": "down_filled",
    "downfilled": "down_filled",
    "downfill": "down_filled",
    "down": "down",

    "insulation": "insulated",
    "insulated": "insulated",

    "breathable": "breathable",

    "zip": "zipper",
    "zipper": "zipper",
    "pocket": "pockets",
    "pockets": "pockets",

    "hood": "hood",
    "hooded": "hood",

    "cap": "cap",
    "hat": "cap",
    "beanie": "beanie",

    "cordura": "cordura",
    "merino": "merino",
    "wool": "wool",
    "ripstop": "ripstop",
    "nylon": "nylon",
}

# Keep keywords domain-focused to reduce noise.
DOMAIN_KEYWORDS = {
    # protection / weather
    "waterproof","water_resistant","water_repellent","wind_resistant","windproof",
    "rain","storm","snow","cold","winter","breathable","reflective","seam_sealed","aquaguard",

    # warmth cues
    "down","down_filled","insulated","thermal","warmth","arctic","expedition","extreme_cold",

    # materials / tech
    "cordura","merino","wool","ripstop","nylon","polartec","tricot",

    # use-case
    "hiking","camping","kayaking","outdoor","city","travel","packable",

    # trims / details people ask
    "fur","coyote_fur","fur_ruff","snorkel_hood",

    # product types
    "parka","jacket","vest","hoody","hoodie","shell","cap","beanie","accessory",
}

# phrase patterns -> keyword tokens
PHRASES = {
    # protection
    r"\bfully seam[- ]sealed\b": "fully_seam_sealed",
    r"\bseam[- ]sealed\b": "seam_sealed",
    r"\baquaguard\b": "aquaguard",
    r"\bstorm flap\b": "storm_flap",
    r"\bmesh venting\b": "mesh_venting",
    r"\bwind guard\b|\bwindguard\b": "wind_guard",

    # warmth / insulation
    r"\bdown[- ]filled\b": "down_filled",
    r"\bthermal mapping\b": "thermal_mapping",
    r"\bextreme cold\b": "extreme_cold",
    r"\barctic\b": "arctic",
    r"\bexpedition\b": "expedition",

    # portability
    r"\bpackable into\b|\bpacks into\b": "packable",
    r"\bbackpack straps\b": "backpack_straps",

    # trims / details
    r"\bfur ruff\b": "fur_ruff",
    r"\bcoyote fur\b": "coyote_fur",
    r"\bsnorkel hood\b": "snorkel_hood",
    r"\brib[- ]knit cuffs?\b": "rib_knit_cuffs",
    r"\bmerino wool\b": "merino_wool",
    r"\bcordura\b": "cordura",
}

TEI_REGEX = re.compile(r"\bTEI\s*([1-5])\b", re.IGNORECASE)

# DB helpers
def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

# Extraction
def extract_tei(text: str) -> Optional[int]:
    if not text:
        return None
    m = TEI_REGEX.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None

def extract_keywords(text: str) -> Set[str]:
    text = (text or "").lower()
    kws: Set[str] = set()

    # phrase hits first
    for pattern, key in PHRASES.items():
        if re.search(pattern, text):
            kws.add(key)

    # tokenization
    cleaned = re.sub(r"[^a-z0-9\s\-]", " ", text)
    cleaned = cleaned.replace("-", " ")
    tokens = cleaned.split()

    for t in tokens:
        if len(t) < 3 or t.isdigit():
            continue
        if t in STOPWORDS:
            continue

        t = NORMALIZE.get(t, t)

        if t in DOMAIN_KEYWORDS:
            kws.add(t)

    return kws

# -------------------------
# Pipeline steps
# -------------------------
def rebuild_keywords(conn: sqlite3.Connection, clear_existing: bool = True) -> None:
    cur = conn.cursor()

    if clear_existing:
        cur.execute("DELETE FROM product_keywords")

    cur.execute("SELECT id, name, description FROM products")
    rows = cur.fetchall()

    for pid, name, desc in rows:
        combined = f"{name or ''} {desc or ''}"
        tei = extract_tei(combined)

        # store TEI on products (best)
        if tei is not None:
            cur.execute("UPDATE products SET tei_level=? WHERE id=?", (tei, pid))
        else:
            cur.execute("UPDATE products SET tei_level=NULL WHERE id=?", (pid,))

        kws = extract_keywords(combined)

        # insert each keyword row; TEI_level repeated per keyword (your schema)
        for kw in kws:
            cur.execute(
                "INSERT OR IGNORE INTO product_keywords(product_id, keyword) VALUES (?, ?)",
                (pid, kw)
            )

    conn.commit()

def print_sanity(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    product_count = cur.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    kw_rows = cur.execute("SELECT COUNT(*) FROM product_keywords").fetchone()[0]
    distinct_kw = cur.execute("SELECT COUNT(DISTINCT keyword) FROM product_keywords").fetchone()[0]
    tei_count = cur.execute("SELECT COUNT(*) FROM products WHERE tei_level IS NOT NULL").fetchone()[0]

    print("Products:", product_count)
    print("Keyword rows:", kw_rows)
    print("Distinct keywords:", distinct_kw)
    print("Products with TEI:", tei_count)
    print("\nTop keywords:")
    print(cur.execute("""
        SELECT keyword, COUNT(*) c
        FROM product_keywords
        GROUP BY keyword
        ORDER BY c DESC
        LIMIT 20
    """).fetchall())

def main() -> None:
    conn = connect_db()

    rebuild_keywords(conn, clear_existing=True)
    print_sanity(conn)

    conn.close()

if __name__ == "__main__":
    main()
