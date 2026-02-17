# This file embeds both product keywords and user queries
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------
# Your config (paste your latest)
# -------------------------
NORMALIZE: Dict[str, str] = {
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

DOMAIN_KEYWORDS = [
    # WEATHER / PROTECTION
    "waterproof", "water_resistant", "water_repellent",
    "windproof", "wind_resistant",
    "snow", "rain", "storm", "downpour",
    "cold", "winter", "extreme_cold",
    "breathable", "venting", "ventilation",
    "seam_sealed", "fully_seam_sealed",
    "storm_flap", "wind_guard",
    "aquaguard", "reflective",

    # INSULATION / WARMTH
    "down", "down_filled", "insulated",
    "thermal", "thermal_mapping",
    "warmth", "arctic", "expedition",
    "tei_1", "tei_2", "tei_3", "tei_4", "tei_5",

    # FABRIC / MATERIAL TECH
    "arctic_tech", "tri_durance", "cordura",
    "merino", "merino_wool",
    "ripstop", "nylon", "cotton",
    "polartec", "power_stretch",
    "tricot", "sueded_tricot",
    "fur", "fur_ruff", "coyote_fur",

    # STRUCTURAL DESIGN
    "snorkel_hood", "helmet_compatible",
    "adjustable_hood", "rib_knit_cuffs",
    "two_way_zipper", "double_zipper",
    "zipper", "drawcord",
    "backpack_straps", "packable",
    "recessed_cuffs",

    # USE CASE / ACTIVITY
    "hiking", "camping", "outdoor",
    "city", "urban", "travel",
    "commute", "performance",
    "active", "everyday",

    # PRODUCT TYPES
    "parka", "jacket", "vest",
    "hoody", "hoodie", "shell",
    "bomber", "coat",
    "cap", "beanie", "accessory"
]

# -------------------------
# Embedder
# -------------------------

@dataclass(frozen=True)
class KeywordMatch:
    token: str
    score: float  # cosine similarity


class KeywordEmbedder:
    """
    Pretrained embedder + cached keyword embeddings.
    - build_cache(): computes embeddings for DOMAIN_KEYWORDS and saves to disk
    - match(query): returns top-k semantically similar keywords (cosine similarity)
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str = "data/embeddings",
        normalize_map: Optional[Dict[str, str]] = None,
        keywords: Optional[List[str]] = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.normalize_map = normalize_map or {}
        self.keywords = list(dict.fromkeys(keywords or []))

        self._model: Optional[SentenceTransformer] = None
        self._kw_tokens: List[str] = []
        self._kw_texts: List[str] = []
        self._kw_emb: Optional[np.ndarray] = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def _token_to_text(token: str) -> str:
        # better matching: "water_repellent" -> "water repellent"
        return token.replace("_", " ")

    def _cache_paths(self) -> Tuple[str, str]:
        os.makedirs(self.cache_dir, exist_ok=True)
        safe_name = self.model_name.replace("/", "__")
        meta_path = os.path.join(self.cache_dir, f"kw_meta__{safe_name}.json")
        emb_path = os.path.join(self.cache_dir, f"kw_emb__{safe_name}.npy")
        return meta_path, emb_path

    def build_cache(self) -> None:
        meta_path, emb_path = self._cache_paths()
        model = self._load_model()

        self._kw_tokens = self.keywords
        self._kw_texts = [self._token_to_text(k) for k in self._kw_tokens]

        emb = model.encode(self._kw_texts, convert_to_numpy=True, normalize_embeddings=True)
        emb = emb.astype(np.float32)

        meta = {
            "model_name": self.model_name,
            "count": len(self._kw_tokens),
            "tokens": self._kw_tokens,
            "texts": self._kw_texts,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        np.save(emb_path, emb)

        self._kw_emb = emb

    def load_cache(self) -> None:
        meta_path, emb_path = self._cache_paths()
        if not (os.path.exists(meta_path) and os.path.exists(emb_path)):
            raise FileNotFoundError("Embedding cache not found. Run build_cache() first.")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self._kw_tokens = list(meta["tokens"])
        self._kw_texts = list(meta["texts"])
        self._kw_emb = np.load(emb_path).astype(np.float32)

    def ensure_loaded(self) -> None:
        if self._kw_emb is None:
            # try load; if not found, build
            try:
                self.load_cache()
            except FileNotFoundError:
                self.build_cache()

    def match(self, query: str, top_k: int = 8, threshold: float = 0.45) -> List[KeywordMatch]:
        self.ensure_loaded()
        assert self._kw_emb is not None

        q = (query or "").strip().lower()
        if not q:
            return []

        model = self._load_model()
        q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]

        scores = self._kw_emb @ q_emb  # cosine similarity (normalized)
        idx = np.argsort(-scores)[: max(1, top_k)]

        out: List[KeywordMatch] = []
        for i in idx:
            s = float(scores[int(i)])
            if s >= threshold:
                out.append(KeywordMatch(self._kw_tokens[int(i)], s))
        return out


# -------------------------
# Run as a script (build + quick test)
# -------------------------
if __name__ == "__main__":
    emb = KeywordEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir="data/embeddings",
        normalize_map=NORMALIZE,
        keywords=DOMAIN_KEYWORDS,
    )

    emb.build_cache()
    print("Built keyword embedding cache.")

    tests = [
        "I want a heavy jacket for cold winter",
        "Need something waterproof for rain",
        "packable travel jacket",
        "fur hood parka",
        "TEI 4 warmest for extreme cold",
    ]
    for t in tests:
        matches = emb.match(t, top_k=8, threshold=0.42)
        print("\nQuery:", t)
        print("Matches:", [(m.token, round(m.score, 3)) for m in matches])
