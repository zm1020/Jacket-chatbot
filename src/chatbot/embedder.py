# This file embeds both product keywords and user queries
# updated 3/10: now it embeds keywords + product descriptions, and matches user queries to both (for better recall)
from __future__ import annotations

import os
import json
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------
# Your config
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
# Shared helpers
# -------------------------

def normalize_l2(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


def build_product_text(
    name: Optional[str],
    gender: Optional[str],
    description: Optional[str],
) -> str:
    parts = [
        str(name or "").strip().lower(),
        str(gender or "").strip().lower(),
        str(description or "").strip().lower(),
    ]
    return " ".join([p for p in parts if p]).strip()


# -------------------------
# Keyword embedder
# -------------------------

@dataclass(frozen=True)
class KeywordMatch:
    token: str
    score: float


class KeywordEmbedder:
    """
    Pretrained embedder + cached keyword embeddings.
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

        emb = model.encode(
            self._kw_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

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
            raise FileNotFoundError("Keyword embedding cache not found. Run build_cache() first.")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self._kw_tokens = list(meta["tokens"])
        self._kw_texts = list(meta["texts"])
        self._kw_emb = np.load(emb_path).astype(np.float32)

    def ensure_loaded(self) -> None:
        if self._kw_emb is None:
            try:
                self.load_cache()
            except FileNotFoundError:
                self.build_cache()

    def encode_query(self, text: str) -> np.ndarray:
        model = self._load_model()
        vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        return vec[0]

    def match(self, query: str, top_k: int = 8, threshold: float = 0.45) -> List[KeywordMatch]:
        self.ensure_loaded()
        assert self._kw_emb is not None

        q = (query or "").strip().lower()
        if not q:
            return []

        q_emb = self.encode_query(q)
        scores = self._kw_emb @ q_emb
        idx = np.argsort(-scores)[:max(1, top_k)]

        out: List[KeywordMatch] = []
        for i in idx:
            s = float(scores[int(i)])
            if s >= threshold:
                out.append(KeywordMatch(self._kw_tokens[int(i)], s))
        return out


# -------------------------
# Product description embedder
# -------------------------

@dataclass(frozen=True)
class ProductSemanticHit:
    product_id: int
    score: float


class ProductDescriptionEmbedder:
    """
    Builds, stores, loads, and searches product-description embeddings.
    Embeddings and metadata are saved into data/.
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        data_dir: str = "data",
    ) -> None:
        self.model_name = model_name
        self.data_dir = data_dir

        self._model: Optional[SentenceTransformer] = None
        self.product_ids: List[int] = []
        self.product_texts: List[str] = []
        self.product_embs: Optional[np.ndarray] = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _cache_paths(self) -> Tuple[str, str]:
        os.makedirs(self.data_dir, exist_ok=True)
        safe_name = self.model_name.replace("/", "__")
        meta_path = os.path.join(self.data_dir, f"product_desc_meta__{safe_name}.json")
        emb_path = os.path.join(self.data_dir, f"product_desc_emb__{safe_name}.npy")
        return meta_path, emb_path

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        model = self._load_model()
        emb = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        return emb

    def build_from_db(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT id, name, gender, description
            FROM products
            ORDER BY id
            """
        ).fetchall()

        ids: List[int] = []
        texts: List[str] = []

        for row in rows:
            text = build_product_text(
                row["name"],
                row["gender"],
                row["description"],
            )
            if not text:
                continue
            ids.append(int(row["id"]))
            texts.append(text)

        self.product_ids = ids
        self.product_texts = texts
        self.product_embs = self.encode_texts(texts) if texts else None

    def save_cache(self) -> None:
        if self.product_embs is None:
            raise ValueError("No product embeddings to save. Run build_from_db() first.")

        meta_path, emb_path = self._cache_paths()

        meta = {
            "model_name": self.model_name,
            "count": len(self.product_ids),
            "product_ids": self.product_ids,
            "product_texts": self.product_texts,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        np.save(emb_path, self.product_embs.astype(np.float32))

    def load_cache(self) -> None:
        meta_path, emb_path = self._cache_paths()
        if not (os.path.exists(meta_path) and os.path.exists(emb_path)):
            raise FileNotFoundError("Product description cache not found. Run build_from_db() + save_cache() first.")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.product_ids = [int(x) for x in meta["product_ids"]]
        self.product_texts = list(meta["product_texts"])
        self.product_embs = np.load(emb_path).astype(np.float32)

    def ensure_loaded(self, conn: Optional[sqlite3.Connection] = None) -> None:
        if self.product_embs is not None:
            return

        try:
            self.load_cache()
        except FileNotFoundError:
            if conn is None:
                raise FileNotFoundError(
                    "Product description cache not found and no DB connection was provided to rebuild it."
                )
            self.build_from_db(conn)
            self.save_cache()

    def rebuild_cache(self, conn: sqlite3.Connection) -> None:
        self.build_from_db(conn)
        self.save_cache()

    def search(self, query: str, top_k: int = 50) -> List[ProductSemanticHit]:
        self.ensure_loaded()
        assert self.product_embs is not None

        q = (query or "").strip().lower()
        if not q:
            return []

        q_emb = self.encode_texts([q])[0]
        scores = self.product_embs @ q_emb
        idx = np.argsort(-scores)[:max(1, top_k)]

        out: List[ProductSemanticHit] = []
        for i in idx:
            out.append(ProductSemanticHit(
                product_id=int(self.product_ids[int(i)]),
                score=float(scores[int(i)])
            ))
        return out


# -------------------------
# Run as a script
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