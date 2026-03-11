import re
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from embedder import KeywordEmbedder, DOMAIN_KEYWORDS, ProductDescriptionEmbedder

DB_PATH = "data/canada_goose.db"

# -------------------------
# Local LLM (Qwen) config
# -------------------------
LOCAL_MODEL = "Qwen/Qwen2.5-3B-Instruct"
USE_4BIT = False
LLM_MAX_NEW_TOKENS_JSON = 220
LLM_MAX_NEW_TOKENS_Q = 80

_tokenizer = None
_model = None


def load_local_llm() -> None:
    global _tokenizer, _model
    if _model is not None and _tokenizer is not None:
        return

    _tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL, trust_remote_code=True)

    kwargs = dict(
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if USE_4BIT:
        kwargs["load_in_4bit"] = True

    _model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL, **kwargs)
    _model.eval()


def llm_generate(system: str, user_payload: str, *, max_new_tokens: int, temperature: float) -> str:
    load_local_llm()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_payload},
    ]
    prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            pad_token_id=_tokenizer.eos_token_id,
        )

    text = _tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split(user_payload, 1)[-1].strip()


def clean_llm_text(text: str) -> str:
    t = (text or "").strip()
    if t.lower().startswith("assistant"):
        t = re.sub(r"^assistant\s*[:：-]?\s*", "", t, flags=re.IGNORECASE)
    return t.strip()


def extract_json_obj(s: str) -> Dict[str, Any]:
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM did not return JSON. Got: {s[:240]}")
    return json.loads(s[start:end + 1])


# -------------------------
# Simple filter parsing (fallback)
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
    return (s or "").strip().lower().replace(" ", "_")


def to_db_variants(canonical_kw: str) -> List[str]:
    k = canonical_kw.strip().lower()
    return list(dict.fromkeys([k, k.replace("_", " ")]))


def normalize_gender_value(s: str) -> str:
    t = (s or "").strip().lower()
    if not t:
        return ""
    if "unisex" in t:
        return "unisex"
    if any(x in t for x in ["women", "woman", "female", "women's", "womens"]):
        return "women"
    if any(x in t for x in ["men", "man", "male", "men's", "mens"]):
        return "men"
    return ""


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


@dataclass
class ConversationState:
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    gender: Optional[str] = None
    tei: Optional[int] = None
    use_case: Optional[str] = None  # school/travel/extreme_cold/rain/everyday/work
    waterproof: Optional[bool] = None
    windproof: Optional[bool] = None
    keywords: List[str] = field(default_factory=list)

    asked_questions: List[str] = field(default_factory=list)
    attempts: Dict[str, int] = field(default_factory=dict)

    def bump_attempt(self, slot: str) -> int:
        self.attempts[slot] = int(self.attempts.get(slot, 0)) + 1
        return self.attempts[slot]

    def missing_slots(self) -> List[str]:
        missing: List[str] = []
        if self.price_min is None and self.price_max is None:
            missing.append("budget")
        if self.gender is None:
            missing.append("gender")
        if self.use_case is None:
            missing.append("use_case")
        return missing


def retrieve_and_rank_hybrid(
    conn: sqlite3.Connection,
    embedder: KeywordEmbedder,
    desc_index: ProductDescriptionEmbedder,
    user_query: str,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    gender: Optional[str] = None,
    top_keywords: int = 12,
    kw_threshold: float = 0.42,
    top_per_product: int = 4,
    candidate_limit: int = 300,
    return_k: int = 5,
    alpha: float = 0.35,
    beta: float = 0.65,
) -> Tuple[List[ScoredProduct], List[Tuple[str, float]]]:
    q = (user_query or "").strip().lower()
    if not q:
        return [], []

    # -------------------------
    # keyword score
    # -------------------------
    matches = embedder.match(q, top_k=top_keywords, threshold=kw_threshold)
    kw_scores: Dict[str, float] = {to_canonical_kw(m.token): float(m.score) for m in matches}

    prod_kw_score: Dict[int, float] = defaultdict(float)

    if kw_scores:
        query_kws_db: List[str] = []
        for ck in kw_scores.keys():
            query_kws_db.extend(to_db_variants(ck))
        query_kws_db = list(dict.fromkeys(query_kws_db))

        if query_kws_db:
            placeholders = ",".join(["?"] * len(query_kws_db))
            kw_rows = conn.execute(
                f"SELECT product_id, keyword FROM product_keywords WHERE keyword IN ({placeholders})",
                query_kws_db,
            ).fetchall()

            prod_scores = defaultdict(list)
            for pid, kw in kw_rows:
                ck = to_canonical_kw(kw)
                if ck in kw_scores:
                    prod_scores[int(pid)].append(kw_scores[ck])

            for pid, scores in prod_scores.items():
                scores.sort(reverse=True)
                prod_kw_score[pid] = sum(scores[:top_per_product])

    # -------------------------
    # description semantic score
    # -------------------------
    desc_hits = desc_index.search(q, top_k=candidate_limit)
    prod_desc_score = {hit.product_id: hit.score for hit in desc_hits}

    candidate_ids = list(set(prod_kw_score.keys()) | set(prod_desc_score.keys()))
    if not candidate_ids:
        return [], sorted(kw_scores.items(), key=lambda x: x[1], reverse=True)

    id_placeholders = ",".join(["?"] * len(candidate_ids))
    where = [f"id IN ({id_placeholders})"]
    params: List[Any] = candidate_ids[:]

    if price_min is not None:
        where.append("price >= ?")
        params.append(price_min)
    if price_max is not None:
        where.append("price <= ?")
        params.append(price_max)

    rows = conn.execute(
        f"""
        SELECT id, name, price, currency, url, gender
        FROM products
        WHERE {" AND ".join(where)}
        """,
        params,
    ).fetchall()

    # Stronger gender filtering in Python to avoid bad SQL matching on dirty values
    if gender is not None:
        filtered_rows = []
        for r in rows:
            row_gender = normalize_gender_value(r["gender"] or "")
            if row_gender == gender or row_gender == "unisex":
                filtered_rows.append(r)
        rows = filtered_rows

    max_kw = max(prod_kw_score.values()) if prod_kw_score else 1.0
    if max_kw == 0:
        max_kw = 1.0

    products: List[ScoredProduct] = []
    for r in rows:
        pid = int(r["id"])
        kw_part = prod_kw_score.get(pid, 0.0) / max_kw
        desc_part = prod_desc_score.get(pid, 0.0)
        final_score = alpha * kw_part + beta * desc_part

        products.append(
            ScoredProduct(
                id=pid,
                score=float(final_score),
                name=r["name"],
                price=float(r["price"]) if r["price"] is not None else 0.0,
                currency=r["currency"] or "",
                url=r["url"] or "",
                gender=r["gender"] or "",
            )
        )

    products.sort(key=lambda p: p.score, reverse=True)
    products = products[:return_k]

    matched_debug = sorted(kw_scores.items(), key=lambda x: x[1], reverse=True)
    return products, matched_debug


# -------------------------
# LLM state + follow-up Qs
# -------------------------
def local_slot_fill(state: ConversationState, history: List[Dict[str, str]], user_msg: str) -> Dict[str, Any]:
    system = (
        "You are a slot-filling assistant for a jacket recommendation chatbot.\n"
        "Extract preference info ONLY if it is explicitly stated in the latest_user_message.\n"
        "Do NOT infer, guess, assume, or carry forward unstated values from earlier dialogue.\n"
        "Do NOT infer, guess, assume the gender to be men or women.\n"
        "Return ONLY valid JSON.\n"
        "If a field is not mentioned, use JSON null, not the string 'null'.\n"
        "Booleans must be true/false, not strings.\n"
        "Numbers must be numbers, not strings.\n\n"
        "Do NOT repeat info already in current_state.\n"

        "Allowed values:\n"
        "- gender: men, women, unisex, null\n"
        "- use_case: school, travel, extreme_cold, rain, everyday, work, null\n"
        "- tei: 1,2,3,4,5 or null\n"
        "- waterproof/windproof: true, false, null\n\n"

        "Keywords must come from the domain vocabulary below.\n"
        "Do NOT invent new keywords.\n\n"

        "Domain vocabulary:\n"
        f"{', '.join(DOMAIN_KEYWORDS)}\n\n"

        "Output format example:\n"
        "{\n"
        '  "price_min": null,\n'
        '  "price_max": 700,\n'
        '  "gender": "men",\n'
        '  "tei": 4,\n'
        '  "use_case": "extreme_cold",\n'
        '  "waterproof": true,\n'
        '  "windproof": true,\n'
        '  "keywords": ["snow", "down", "parka"]\n'
        "}\n"
    )

    payload_obj = {
        "current_state": {
            "price_min": state.price_min,
            "price_max": state.price_max,
            "gender": state.gender,
            "tei": state.tei,
            "use_case": state.use_case,
            "waterproof": state.waterproof,
            "windproof": state.windproof,
            "keywords": state.keywords,
        },
        "recent_dialogue": history[-6:],
        "latest_user_message": user_msg,
        "output_schema": {
            "price_min": "number|null",
            "price_max": "number|null",
            "gender": "men|women|unisex|null",
            "tei": "1|2|3|4|5|null",
            "use_case": "school|travel|extreme_cold|rain|everyday|work|null",
            "waterproof": "true|false|null",
            "windproof": "true|false|null",
            "keywords": "array of strings",
        },
    }

    raw = llm_generate(
        system,
        json.dumps(payload_obj),
        max_new_tokens=LLM_MAX_NEW_TOKENS_JSON,
        temperature=0.1,
    )
    return extract_json_obj(raw)


def normalize_slot_value(key: str, value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, str):
        v = value.strip().lower()

        if v in {"null", "none", "", "unknown", "n/a"}:
            return None

        if key == "gender":
            if v in {"men", "mens", "male"}:
                return "men"
            if v in {"women", "womens", "female"}:
                return "women"
            if v == "unisex":
                return "unisex"
            return None

        if key == "use_case":
            allowed = {"school", "travel", "extreme_cold", "rain", "everyday", "work"}
            return v if v in allowed else None

        if key in {"waterproof", "windproof"}:
            if v in {"true", "yes"}:
                return True
            if v in {"false", "no"}:
                return False
            return None

        if key == "tei":
            if v in {"1", "2", "3", "4", "5"}:
                return int(v)
            return None

        if key in {"price_min", "price_max"}:
            try:
                return float(v)
            except ValueError:
                return None

    if key == "tei" and isinstance(value, (int, float)):
        iv = int(value)
        return iv if iv in {1, 2, 3, 4, 5} else None

    if key in {"price_min", "price_max"} and isinstance(value, (int, float)):
        return float(value)

    if key in {"waterproof", "windproof"} and isinstance(value, bool):
        return value

    return value


def merge_state(state: ConversationState, upd: Dict[str, Any]) -> None:
    for k in ["price_min", "price_max", "gender", "tei", "use_case", "waterproof", "windproof"]:
        print(f"Processing slot '{k}': current value={getattr(state, k)}, new value={upd.get(k)}")
        if k in upd:
            val = normalize_slot_value(k, upd[k])
            if val is not None:
                setattr(state, k, val)

    if "keywords" in upd and isinstance(upd["keywords"], list):
        merged = list(
            dict.fromkeys(
                state.keywords
                + [str(x).strip().lower() for x in upd["keywords"] if str(x).strip()]
            )
        )
        state.keywords = merged


def local_generate_unique_question(state: ConversationState, missing_slot: str, history: List[Dict[str, str]]) -> str:
    system = (
        "You are a conversational assistant helping a user choose a jacket.\n"
        "Ask ONE helpful follow-up question to gather missing information.\n"
        "Do NOT repeat or paraphrase any question in previous_questions.\n"
        "Do NOT ask about information that already has a value in current_state.\n"
        "If the user seems unsure, present 4 short multiple-choice options.\n"
        "Keep the question under 25 words.\n"
        "Ask exactly ONE question.\n"
        "Return ONLY the question text."
    )

    prompt_obj = {
        "missing_slot": missing_slot,
        "current_state": {
            "price_min": state.price_min,
            "price_max": state.price_max,
            "gender": state.gender,
            "tei": state.tei,
            "use_case": state.use_case,
            "waterproof": state.waterproof,
            "windproof": state.windproof,
            "keywords": state.keywords[:10],
        },
        "previous_questions": state.asked_questions[-10:],
        "recent_dialogue": history[-6:],
    }

    raw = llm_generate(
        system,
        json.dumps(prompt_obj),
        max_new_tokens=LLM_MAX_NEW_TOKENS_Q,
        temperature=0.7,
    ).strip()

    q = clean_llm_text(raw.strip().strip('"').strip("'").strip())
    if not q:
        q = "Can you share a bit more about what you need?"
    if q in state.asked_questions:
        q = q + " (Pick the closest option.)"
    return q


def build_final_query(state: ConversationState, user_msg: str) -> str:
    parts: List[str] = []

    if state.gender:
        parts.append(state.gender)

    if state.use_case:
        parts.append(state.use_case.replace("_", " "))

    if state.tei is not None:
        parts.append(f"TEI {state.tei}")

    if state.waterproof is True:
        parts.append("waterproof")
    if state.windproof is True:
        parts.append("windproof")

    if state.keywords:
        parts.extend(state.keywords[:8])

    if user_msg:
        parts.append(user_msg.lower())

    return " ".join(parts)


# -------------------------
# Chat runner
# -------------------------
def format_results(items: List[ScoredProduct]) -> str:
    if not items:
        return "No results (after filters). Try removing constraints or changing wording."
    lines = []
    for i, p in enumerate(items, 1):
        lines.append(f"{i}) {p.name} — ${p.price:g} {p.currency} — score {p.score:.3f}\n   {p.url}")
    return "\n".join(lines)


def map_llm_keywords_to_domain(
    embedder: KeywordEmbedder,
    llm_keywords: List[str],
    *,
    sim_threshold: float = 0.6,
) -> List[Tuple[str, float, str]]:
    mapped = []
    seen = set()

    for kw in llm_keywords:
        kw = (kw or "").strip()
        if not kw:
            continue

        hits = embedder.match(kw, top_k=1, threshold=sim_threshold)
        if not hits:
            continue

        best = hits[0]
        dk = to_canonical_kw(best.token)
        if dk in seen:
            continue

        mapped.append((dk, float(best.score), kw))
        seen.add(dk)

    mapped.sort(key=lambda x: x[1], reverse=True)
    return mapped


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    emb = KeywordEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir="data/embeddings",
        keywords=DOMAIN_KEYWORDS,
    )

    desc_index = ProductDescriptionEmbedder(data_dir="data")
    desc_index.ensure_loaded(conn)

    print("Draft Chatbot vLocal (Qwen slot-fill + hybrid ranking with descriptions). Type 'quit' to exit.\n")

    state = ConversationState()
    history: List[Dict[str, str]] = []

    while True:
        user = input("Tell me what you're looking for: ").strip()
        if not user:
            continue
        if user.lower() in {"quit", "exit"}:
            break

        history.append({"role": "user", "content": user})

        try:
            upd = local_slot_fill(state, history, user)
            merge_state(state, upd)
            mapped = map_llm_keywords_to_domain(emb, state.keywords, sim_threshold=0.6)
            state.keywords = [dk for (dk, sim, orig) in mapped]

            print("LLM→Domain mapping:", [(orig, dk, round(sim, 3)) for (dk, sim, orig) in mapped])
        except Exception:
            pmin, pmax, parsed_gender = parse_filters(user)
            if pmin is not None:
                state.price_min = pmin
            if pmax is not None:
                state.price_max = pmax
            if parsed_gender is not None:
                state.gender = parsed_gender

        missing = state.missing_slots()

        if missing:
            slot = missing[0]
            tries = state.bump_attempt(slot)

            if slot == "use_case" and tries >= 3 and state.use_case is None:
                state.use_case = "everyday"
                missing = state.missing_slots()

            if missing:
                qtext = local_generate_unique_question(state, slot, history)
                state.asked_questions.append(qtext)
                print(f"Bot: {qtext}\n")
                history.append({"role": "assistant", "content": qtext})
                continue

        print("Bot: Searching for jackets...\n")

        final_query = build_final_query(state, user)

        results, matched = retrieve_and_rank_hybrid(
            conn=conn,
            embedder=emb,
            desc_index=desc_index,
            user_query=final_query,
            price_min=state.price_min,
            price_max=state.price_max,
            gender=state.gender,
            top_keywords=12,
            kw_threshold=0.42,
            top_per_product=4,
            candidate_limit=300,
            return_k=5,
            alpha=0.35,
            beta=0.65,
        )

        print("Matched keywords:", [(k, round(s, 6)) for k, s in matched[:10]])
        print(
            "State:",
            {
                "price_min": state.price_min,
                "price_max": state.price_max,
                "gender": state.gender,
                "use_case": state.use_case,
                "tei": state.tei,
                "waterproof": state.waterproof,
                "windproof": state.windproof,
                "keywords": state.keywords[:10],
            },
        )
        print(format_results(results))
        print()

    conn.close()


if __name__ == "__main__":
    main()
