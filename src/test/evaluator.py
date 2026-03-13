import sys
from pathlib import Path
from typing import List, Dict, Any
import sqlite3

# ----------------------------
# Path setup
# ----------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chatbot.chatbot_runner import (
    DB_PATH,
    ConversationState,
    KeywordEmbedder,
    DOMAIN_KEYWORDS,
    ProductDescriptionEmbedder,
    local_slot_fill,
    merge_state,
    map_llm_keywords_to_domain,
    parse_filters,
    retrieve_and_rank_hybrid,
    build_final_query,
    format_results,
)

PROMPTS_FILE = CURRENT_DIR / "prompts.txt"
OUTPUT_FILE = CURRENT_DIR / "evaluation_output.txt"


# ----------------------------
# Load prompts
# ----------------------------
def load_prompts(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)

    return prompts


# ----------------------------
# State printer
# ----------------------------
def state_to_dict(state: ConversationState) -> Dict[str, Any]:
    return {
        "price_min": state.price_min,
        "price_max": state.price_max,
        "gender": state.gender,
        "tei": state.tei,
        "use_case": state.use_case,
        "waterproof": state.waterproof,
        "windproof": state.windproof,
        "keywords": state.keywords[:10],
    }


# ----------------------------
# Run one evaluation
# ----------------------------
def evaluate_one_prompt(prompt: str, conn, emb, desc_index) -> str:
    state = ConversationState()
    history = [{"role": "user", "content": prompt}]

    mapping_debug = []
    fallback_used = False

    try:
        upd = local_slot_fill(state, history, prompt)
        merge_state(state, upd)

        mapped = map_llm_keywords_to_domain(emb, state.keywords, sim_threshold=0.6)
        state.keywords = [dk for (dk, sim, orig) in mapped]

        mapping_debug = [(orig, dk, round(sim, 3)) for (dk, sim, orig) in mapped]

    except Exception as e:
        fallback_used = True

        pmin, pmax, parsed_gender = parse_filters(prompt)
        if pmin is not None:
            state.price_min = pmin
        if pmax is not None:
            state.price_max = pmax
        if parsed_gender is not None:
            state.gender = parsed_gender

        mapping_debug = [("fallback", "fallback", str(e))]

    missing = state.missing_slots()

    final_query = build_final_query(state, prompt)

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

    lines = []
    lines.append("=" * 80)
    lines.append(f"PROMPT: {prompt}")
    lines.append(f"FALLBACK_USED: {fallback_used}")
    lines.append(f"MISSING_SLOTS: {missing}")
    lines.append(f"STATE: {state_to_dict(state)}")
    lines.append(f"MAPPING: {mapping_debug}")
    lines.append("")
    lines.append("RESULTS:")
    lines.append(format_results(results))
    lines.append("")

    return "\n".join(lines)


# ----------------------------
# Main evaluator
# ----------------------------
def main():

    prompts = load_prompts(PROMPTS_FILE)

    print(f"\nLoaded {len(prompts)} prompts from {PROMPTS_FILE}\n")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    emb = KeywordEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir="data/embeddings",
        keywords=DOMAIN_KEYWORDS,
    )

    desc_index = ProductDescriptionEmbedder(data_dir="data")
    desc_index.ensure_loaded(conn)

    outputs = ["BATCH EVALUATION OUTPUT\n"]

    for i, prompt in enumerate(prompts, 1):

        print(f"\n==============================")
        print(f"Running test case {i}/{len(prompts)}")
        print("==============================\n")

        result_text = evaluate_one_prompt(prompt, conn, emb, desc_index)

        # print to terminal
        print(result_text)

        # save to file
        outputs.append(f"TEST CASE {i}")
        outputs.append(result_text)

    OUTPUT_FILE.write_text("\n".join(outputs), encoding="utf-8")

    conn.close()

    print("\nSaved evaluation output to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()