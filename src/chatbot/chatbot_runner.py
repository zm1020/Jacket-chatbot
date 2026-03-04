import re
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from embedder import KeywordEmbedder, DOMAIN_KEYWORDS  # same folder import

DB_PATH = "data/canada_goose.db"

# -------------------------
# Local LLM (Qwen) config
# -------------------------
LOCAL_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # auto-downloads via HF cache
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
    # Best-effort: return tail (assistant answer). Works well enough for simple prompts.
    return text.split(user_payload, 1)[-1].strip()


def extract_json_obj(s: str) -> Dict[str, Any]:
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM did not return JSON. Got: {s[:240]}")
    return json.loads(s[start : end + 1])


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
    top_keywords: int = 12,
    kw_threshold: float = 0.42,
    top_per_product: int = 4,
    candidate_limit: int = 300,
    return_k: int = 5,
) -> Tuple[List[ScoredProduct], List[Tuple[str, float]]]:
    q = (user_query or "").strip().lower()
    if not q:
        return [], []

    matches = embedder.match(q, top_k=top_keywords, threshold=kw_threshold)
    if not matches:
        return [], []

    kw_scores: Dict[str, float] = {to_canonical_kw(m.token): float(m.score) for m in matches}

    query_kws_db: List[str] = []
    for ck in kw_scores.keys():
        query_kws_db.extend(to_db_variants(ck))
    query_kws_db = list(dict.fromkeys(query_kws_db))

    placeholders = ",".join(["?"] * len(query_kws_db))
    kw_rows = conn.execute(
        f"SELECT product_id, keyword FROM product_keywords WHERE keyword IN ({placeholders})",
        query_kws_db,
    ).fetchall()

    prod_scores = defaultdict(list)  # pid -> [scores...]
    for pid, kw in kw_rows:
        ck = to_canonical_kw(kw)
        if ck in kw_scores:
            prod_scores[int(pid)].append(kw_scores[ck])

    if not prod_scores:
        return [], [(k, kw_scores[k]) for k in kw_scores.keys()]

    scored_pids = []
    for pid, scores in prod_scores.items():
        scores.sort(reverse=True)
        scored_pids.append((pid, sum(scores[:top_per_product])))
    scored_pids.sort(key=lambda x: x[1], reverse=True)
    scored_pids = scored_pids[:candidate_limit]

    candidate_ids = [pid for pid, _ in scored_pids]
    if not candidate_ids:
        return [], []

    id_placeholders = ",".join(["?"] * len(candidate_ids))
    where = [f"id IN ({id_placeholders})"]
    params: List[Any] = candidate_ids[:]

    if price_min is not None:
        where.append("price >= ?")
        params.append(price_min)
    if price_max is not None:
        where.append("price <= ?")
        params.append(price_max)
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

    matched_debug = sorted([(k, v) for k, v in kw_scores.items()], key=lambda x: x[1], reverse=True)
    return products, matched_debug


# -------------------------
# LLM state + follow-up Qs
# -------------------------
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
        if self.use_case is None:
            missing.append("use_case")
        return missing


def local_slot_fill(state: ConversationState, history: List[Dict[str, str]], user_msg: str) -> Dict[str, Any]:
    system = (
        "You are a slot-filling assistant for a jacket recommendation chatbot.\n"
        "Your task is to extract structured preferences from the user's latest message.\n"
        "Return ONLY valid JSON (no explanations or extra text).\n"
        "If information is not mentioned, return null.\n\n"

        "Allowed values:\n"
        "- gender: men, women, unisex, null\n"
        "- use_case: school, travel, extreme_cold, rain, everyday, work, null\n"
        "- tei: 1,2,3,4,5 or null\n"
        "- waterproof/windproof: true, false, or null\n\n"

        "Keyword extraction rules:\n"
        "- Extract up to 6 keywords describing the user's needs.\n"
        "- Keywords should represent weather protection, insulation, material, activity, or product type.\n"
        "- Prefer keywords from the domain vocabulary below.\n"
        "- If the user uses similar wording (e.g., snowy, freezing, stormy), map it to the closest keyword.\n"
        "- Do NOT invent new keywords outside the vocabulary.\n\n"

        "Domain vocabulary:\n"
        f"{', '.join(DOMAIN_KEYWORDS)}\n"
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


def merge_state(state: ConversationState, upd: Dict[str, Any]) -> None:
    for k in ["price_min", "price_max", "gender", "tei", "use_case", "waterproof", "windproof"]:
        if k in upd and upd[k] is not None:
            setattr(state, k, upd[k])

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
        "Your goal is to ask ONE helpful follow-up question to gather missing information.\n"
        "\n"
        "STRICT RULES:\n"
        "- Do NOT repeat or paraphrase any question in previous_questions.\n"
        "- Do NOT ask about information that already has a value in current_state.\n"
        "- If the user seems unsure (e.g., 'idk', 'not sure'), present 4 short multiple-choice options.\n"
        "- If the same information has been requested more than twice, do NOT ask again.\n"
        "- Keep the question under 25 words.\n"
        "- Ask exactly ONE question.\n"
        "- Do not mention internal system logic.\n"
        "- Return ONLY the question text."
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

    q = raw.strip().strip('"').strip("'").strip()
    if not q:
        q = "Can you share a bit more about what you need?"
    if q in state.asked_questions:
        q = q + " (Pick the closest option.)"
    return q


def build_final_query(state: ConversationState, user_msg: str) -> str:
    parts: List[str] = []
    if state.keywords:
        parts.extend(state.keywords[:8])
    if state.use_case:
        parts.append(state.use_case)
    if state.waterproof is True:
        parts.append("waterproof")
    if state.windproof is True:
        parts.append("windproof")
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
    """
    Returns list of (domain_kw, similarity, original_kw) for keywords that pass threshold.
    Makes sure the keywords are mapped to the canonical domain keywords, so they can be used for retrieval. 
    The original LLM keywords are also returned for debugging.
    """
    mapped = []
    seen = set()

    for kw in llm_keywords:
        kw = (kw or "").strip()
        if not kw:
            continue

        # match() expects text; here we feed the keyword itself
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

    print("Draft Chatbot vLocal (Qwen slot-fill + unique follow-ups). Type 'quit' to exit.\n")

    state = ConversationState()
    history: List[Dict[str, str]] = []

    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() in {"quit", "exit"}:
            break

        history.append({"role": "user", "content": user})

        # LLM slot-fill; fallback to regex if LLM fails
        try:
            upd = local_slot_fill(state, history, user)
            merge_state(state, upd)
            mapped = map_llm_keywords_to_domain(emb, state.keywords, sim_threshold=0.6)
            state.keywords = [dk for (dk, sim, orig) in mapped]

            print("LLM→Domain mapping:", [(orig, dk, round(sim, 3)) for (dk, sim, orig) in mapped]) # debug print for LLM keyword mapping
        except Exception:
            pmin, pmax, gender = parse_filters(user)
            if pmin is not None:
                state.price_min = pmin
            if pmax is not None:
                state.price_max = pmax
            if gender is not None:
                state.gender = gender

        missing = state.missing_slots()

        if missing:
            slot = missing[0]
            tries = state.bump_attempt(slot)

            # prevent infinite loops: after N tries, set safe default for use_case
            if slot == "use_case" and tries >= 3 and state.use_case is None:
                state.use_case = "everyday"
                missing = state.missing_slots()

            if missing:
                qtext = local_generate_unique_question(state, slot, history)
                state.asked_questions.append(qtext)
                print(f"Bot: {qtext}\n")
                history.append({"role": "assistant", "content": qtext})
                continue
            

        final_query = build_final_query(state, user)

        results, matched = retrieve_and_rank(
            conn=conn,
            embedder=emb,
            user_query=final_query,
            price_min=state.price_min,
            price_max=state.price_max,
            gender=state.gender,
            top_keywords=12,
            kw_threshold=0.42,
            top_per_product=4,
            candidate_limit=300,
            return_k=5,
        )

        print("Matched keywords:", [(k, round(s, 6)) for k, s in matched[:10]])
        print(
            "State:",
            {
                "price_min": state.price_min,
                "price_max": state.price_max,
                "gender": state.gender,
                "use_case": state.use_case,
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