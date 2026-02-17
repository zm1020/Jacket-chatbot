import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from embedder import KeywordEmbedder, DOMAIN_KEYWORDS  # same folder import

DB_PATH = "data/canada_goose.db"

# -------------------------
# Simple filter parsing
# -------------------------
PRICE_MAX = re.compile(r"(?:under|below|less than|<)\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
PRICE_MIN = re.compile(r"(?:over|above|more than|>)\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
GENDER_RE = re.compile(r"\b(men|mens|women|womens|unisex|male|female)\b", re.IGNORECASE)

def parse_filters(q: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    pmin = pmax = None
    gender = None

    m = PRICE_MAX.search(q)
    if m:
        pmax = float(m.group(1))
    m = PRICE_MIN.search(q)
    if m:
        pmin = float(m.group(1))

    m = GENDER_RE.search(q)
    if m:
        g = m.group(1).lower()
        if g in {"men", "mens", "male"}:
            gender = "men"
        elif g in {"women", "womens", "female"}:
            gender = "women"
        elif g == "unisex":
            gender = "unisex"

    return pmin, pmax, gender

# -------------------------
# Keyword canonicalization helpers
# -------------------------
def to_canonical_kw(s: str) -> str:
    # DB might store "storm flap" while embedder returns "storm_flap"
    return (s or "").strip().lower().replace(" ", "_")

def to_db_variants(canonical_kw: str) -> List[str]:
    # Query DB for both underscore and space form to be robust
    k = canonical_kw.strip().lower()
    return list(dict.fromkeys([k, k.replace("_", " ")]))

# -------------------------
# Retrieval + Ranking
# -------------------------
@dataclass
class ScoredProduct:
    id: int
    score: float
    name: str
    price: float
    currency: str
    url: str
    gender: str

def retrieve_and_rank(
    conn: sqlite3.Connection,
    embedder: KeywordEmbedder,
    user_query: str,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    gender: Optional[str] = None,
    # knobs
    top_keywords: int = 12,         # how many high-sim keywords to consider
    kw_threshold: float = 0.42,     # ignore weak matches
    top_per_product: int = 4,       # sum top N keyword scores per product
    candidate_limit: int = 300,     # limit candidate products before filter
    return_k: int = 5,
) -> Tuple[List[ScoredProduct], List[Tuple[str, float]]]:
    q = (user_query or "").strip().lower()
    if not q:
        return [], []

    # 1) semantic keyword matches
    matches = embedder.match(q, top_k=top_keywords, threshold=kw_threshold)
    if not matches:
        return [], []

    # canonical keyword scores (underscore)
    kw_scores: Dict[str, float] = {to_canonical_kw(m.token): float(m.score) for m in matches}

    # 2) fetch candidate (product_id, keyword) rows for any matched keyword (robust to spaces/underscores)
    query_kws_db: List[str] = []
    for ck in kw_scores.keys():
        query_kws_db.extend(to_db_variants(ck))
    query_kws_db = list(dict.fromkeys(query_kws_db))

    placeholders = ",".join(["?"] * len(query_kws_db))
    kw_rows = conn.execute(
        f"SELECT product_id, keyword FROM product_keywords WHERE keyword IN ({placeholders})",
        query_kws_db,
    ).fetchall()

    # 3) aggregate keyword scores per product
    prod_scores = defaultdict(list)  # pid -> [scores...]
    for pid, kw in kw_rows:
        ck = to_canonical_kw(kw)
        if ck in kw_scores:
            prod_scores[int(pid)].append(kw_scores[ck])

    if not prod_scores:
        return [], [(k, kw_scores[k]) for k in kw_scores.keys()]

    # 4) product score = sum(top N keyword scores)
    scored_pids = []
    for pid, scores in prod_scores.items():
        scores.sort(reverse=True)
        scored_pids.append((pid, sum(scores[:top_per_product])))
    scored_pids.sort(key=lambda x: x[1], reverse=True)
    scored_pids = scored_pids[:candidate_limit]

    # 5) fetch product rows for candidates + apply hard filters in SQL
    candidate_ids = [pid for pid, _ in scored_pids]
    id_placeholders = ",".join(["?"] * len(candidate_ids))

    where = [f"id IN ({id_placeholders})"]
    params: List = candidate_ids[:]

    if price_min is not None:
        where.append("price >= ?")
        params.append(price_min)
    if price_max is not None:
        where.append("price <= ?")
        params.append(price_max)

    # gender in DB might be "Unknown"; keep filter loose
    if gender is not None:
        where.append("LOWER(gender) LIKE ?")
        params.append(f"%{gender}%")

    rows = conn.execute(
        f"""
        SELECT id, name, price, currency, url, gender
        FROM products
        WHERE {" AND ".join(where)}
        """,
        params,
    ).fetchall()

    # 6) merge scores + final sort
    score_map = {pid: score for pid, score in scored_pids}
    products: List[ScoredProduct] = []
    for r in rows:
        pid = int(r[0])
        products.append(
            ScoredProduct(
                id=pid,
                score=float(score_map.get(pid, 0.0)),
                name=r[1],
                price=float(r[2]) if r[2] is not None else 0.0,
                currency=r[3] or "",
                url=r[4] or "",
                gender=r[5] or "",
            )
        )

    products.sort(key=lambda p: p.score, reverse=True)
    products = products[:return_k]

    # return matched keywords (canonical) with scores for debug
    matched_debug = sorted([(k, v) for k, v in kw_scores.items()], key=lambda x: x[1], reverse=True)

    return products, matched_debug

# -------------------------
# Chat runner
# -------------------------
def format_results(items: List[ScoredProduct]) -> str:
    if not items:
        return "No results (after filters). Try removing constraints or changing wording."
    lines = []
    for i, p in enumerate(items, 1):
        lines.append(
            f"{i}) {p.name} — ${p.price:g} {p.currency} — score {p.score:.3f}\n"
            f"   {p.url}"
        )
    return "\n".join(lines)

def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # not required, but ok

    emb = KeywordEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir="data/embeddings",
        keywords=DOMAIN_KEYWORDS,
    )

    print("Draft Chatbot v1 (Embeddings -> Product scoring -> Filters). Type 'quit' to exit.\n")

    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() in {"quit", "exit"}:
            break

        pmin, pmax, gender = parse_filters(user)

        results, matched = retrieve_and_rank(
            conn=conn,
            embedder=emb,
            user_query=user,
            price_min=pmin,
            price_max=pmax,
            gender=gender,
            top_keywords=12,
            kw_threshold=0.42,
            top_per_product=4,
            candidate_limit=300,
            return_k=5,
        )

        print("Matched keywords (AI):", [(k, round(s, 3)) for k, s in matched[:10]])
        if pmin is not None or pmax is not None or gender is not None:
            print("Filters:", {"price_min": pmin, "price_max": pmax, "gender": gender})
        print(format_results(results))
        print()

    conn.close()

if __name__ == "__main__":
    main()

