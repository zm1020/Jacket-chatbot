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

    # =========================
    # WEATHER / PROTECTION
    # =========================
    "waterproof", "water_resistant", "water_repellent",
    "windproof", "wind_resistant",
    "snow", "rain", "storm", "downpour",
    "cold", "winter", "extreme_cold",
    "breathable", "venting", "ventilation",
    "seam_sealed", "fully_seam_sealed",
    "storm_flap", "wind_guard",
    "aquaguard", "reflective",

    # =========================
    # INSULATION / WARMTH
    # =========================
    "down", "down_filled", "insulated",
    "thermal", "thermal_mapping",
    "warmth", "arctic", "expedition",
    "tei_1", "tei_2", "tei_3", "tei_4", "tei_5",

    # =========================
    # FABRIC / MATERIAL TECH
    # =========================
    "arctic_tech", "tri_durance", "cordura",
    "merino", "merino_wool",
    "ripstop", "nylon", "cotton",
    "polartec", "power_stretch",
    "tricot", "sueded_tricot",
    "fur", "fur_ruff", "coyote_fur",

    # =========================
    # STRUCTURAL DESIGN
    # =========================
    "snorkel_hood", "helmet_compatible",
    "adjustable_hood", "rib_knit_cuffs",
    "two_way_zipper", "double_zipper",
    "zipper", "drawcord",
    "backpack_straps", "packable",
    "recessed_cuffs",

    # =========================
    # USE CASE / ACTIVITY
    # =========================
    "hiking", "camping", "outdoor",
    "city", "urban", "travel",
    "commute", "performance",
    "active", "everyday",

    # =========================
    # PRODUCT TYPES
    # =========================
    "parka", "jacket", "vest",
    "hoody", "hoodie", "shell",
    "bomber", "coat",
    "cap", "beanie", "accessory"
}

# phrase patterns -> keyword tokens
PHRASES = {

    # WEATHER
    r"\bfully seam[- ]sealed\b": "fully_seam_sealed",
    r"\bwater[- ]repellent\b": "water_repellent",
    r"\bwater[- ]resistant\b": "water_resistant",
    r"\bwind[- ]resistant\b": "wind_resistant",
    r"\bstorm flap\b": "storm_flap",
    r"\bwind guard\b": "wind_guard",

    # INSULATION
    r"\bdown[- ]filled\b": "down_filled",
    r"\bthermal mapping\b": "thermal_mapping",
    r"\bthermal experience index\b": "tei",
    r"\bextreme cold\b": "extreme_cold",

    # MATERIAL
    r"\barctic tech\b": "arctic_tech",
    r"\btri[- ]durance\b": "tri_durance",
    r"\bpower stretch\b": "power_stretch",
    r"\bsueded tricot\b": "sueded_tricot",
    r"\bmerino wool\b": "merino_wool",

    # STRUCTURE
    r"\bsnorkel hood\b": "snorkel_hood",
    r"\bhelmet[- ]compatible\b": "helmet_compatible",
    r"\bbackpack straps\b": "backpack_straps",
    r"\btwo[- ]way zipper\b": "two_way_zipper",
    r"\bdouble[- ]zipper\b": "double_zipper",
    r"\brib[- ]knit cuffs?\b": "rib_knit_cuffs",
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
