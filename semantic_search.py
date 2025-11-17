import os
import json
import pickle
import re
from typing import List, Dict, Tuple
import numpy as np

try:
    import faiss  # type: ignore
except ImportError as e:
    raise ImportError("faiss is required. Install with `pip install faiss-cpu`") from e

from sentence_transformers import SentenceTransformer


def _simple_sentence_split(text: str) -> List[str]:
    """
    Lightweight sentence splitter that avoids external downloads.
    Keeps punctuation attached to the sentence.
    - Splits on ., ?, ! followed by space or EoS
    - Handles common abbreviations to reduce over-splitting
    """
    if not text:
        return []

    # Protect some common abbreviations
    protected = {
        "e.g.": "e<abbr>g<abbr>",
        "i.e.": "i<abbr>e<abbr>",
        "Dr.": "Dr<abbr>",
        "Mr.": "Mr<abbr>",
        "Mrs.": "Mrs<abbr>",
        "vs.": "vs<abbr>",
        "etc.": "etc<abbr>",
        "Prof.": "Prof<abbr>",
        "Sr.": "Sr<abbr>",
        "Jr.": "Jr<abbr>",
        "No.": "No<abbr>",
    }
    safe = text
    for k, v in protected.items():
        safe = safe.replace(k, v)

    # Split on sentence enders.
    # Use a regex that captures punctuation as part of the sentence.
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"(\[])', safe.strip())

    # Restore abbreviations
    def restore(s: str) -> str:
        out = s
        for k, v in protected.items():
            out = out.replace(v, k)
        return out.strip()

    sentences = [restore(p) for p in parts if p.strip()]
    return sentences


def _normalize(v: np.ndarray) -> np.ndarray:
    """L2 normalize rows of a 2D array for Inner Product similarity."""
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


class SemanticSearchEngine:
    """
    Sentence-level RAG index with neighbor expansion.

    Public Methods:
      - load_chunks(json_path): load documents (expects list of {content, source})
      - create_embeddings(): compute embeddings for all sentences
      - build_index(embeddings): build FAISS index (Inner Product over normalized vectors)
      - save_index(): persist FAISS + metadata
      - load_index(): load FAISS + metadata
      - search(query, top_k=5, window=2): retrieve sentence hits and expand with neighbors

    Stored metadata (self.meta):
      - docs: List[Dict] original docs with {source, content}
      - sentences: List[str] all sentences flattened
      - sent_map: List[Tuple[doc_id, sent_idx]] mapping each sentence to (doc, index in doc)
      - doc_sent_spans: List[List[int]] mapping doc_id -> list of sentence indices in `sentences`
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None  # FAISS index
        self.meta: Dict = {
            "docs": [],
            "sentences": [],
            "sent_map": [],         # [(doc_id, sent_idx_in_doc)]
            "doc_sent_spans": [],   # [[global_sent_ids...], ...]
        }
        # Filenames (kept same as your previous code so test2.py doesn't need edits)
        self.faiss_path = "faiss_index.bin"
        self.meta_path = "chunks_data.pkl"

    # ---------- Building ----------
    def load_chunks(self, json_path: str):
        """
        Load documents from a JSON file expected to be:
          [{"content": "...", "source": "..."}, ...]
        If your current file already contains "chunks", it's fine—each will be re-split into sentences.
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Cannot find {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Expected a list of {content, source} objects in chunks.json")

        self.meta["docs"] = []
        self.meta["sentences"] = []
        self.meta["sent_map"] = []
        self.meta["doc_sent_spans"] = []

        for doc_id, item in enumerate(data):
            content = (item.get("content") or "").strip()
            source = item.get("source") or f"doc_{doc_id}"
            self.meta["docs"].append({"source": source, "content": content})

            sents = _simple_sentence_split(content)
            global_ids = []
            for sent_idx, sent in enumerate(sents):
                global_id = len(self.meta["sentences"])
                self.meta["sentences"].append(sent)
                self.meta["sent_map"].append((doc_id, sent_idx))
                global_ids.append(global_id)
            self.meta["doc_sent_spans"].append(global_ids)

        total_sents = len(self.meta["sentences"])
        print(f"Loaded {len(self.meta['docs'])} docs → {total_sents} sentences.")

    def create_embeddings(self) -> np.ndarray:
        """Create embeddings for all sentences."""
        sentences = self.meta["sentences"]
        if not sentences:
            raise RuntimeError("No sentences loaded. Call load_chunks() first.")
        embs = self.model.encode(sentences, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=True)
        embs = _normalize(embs.astype("float32"))
        return embs

    def build_index(self, embeddings: np.ndarray):
        """Build a FAISS inner-product index on normalized vectors."""
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        self.index = index
        print(f"Built FAISS index with {index.ntotal} vectors (dim={d}).")

    def save_index(self):
        """Persist FAISS index + metadata."""
        if self.index is None:
            raise RuntimeError("Index not built.")
        faiss.write_index(self.index, self.faiss_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.meta, f)
        print(f"Saved index → {self.faiss_path}, metadata → {self.meta_path}")

    # ---------- Loading ----------
    def load_index(self):
        """Load FAISS index + metadata from disk."""
        if not os.path.exists(self.faiss_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError("Index or metadata not found. Build and save the index first.")

        self.index = faiss.read_index(self.faiss_path)
        with open(self.meta_path, "rb") as f:
            self.meta = pickle.load(f)
        print(f"Loaded index with {self.index.ntotal} vectors")
        print(f"Loaded {len(self.meta['sentences'])} sentences from metadata")

    # ---------- Retrieval ----------
    def _expand_hit(self, global_sent_id: int, window: int) -> Tuple[int, int]:
        """
        Given a global sentence id, expand to [start, end] (inclusive) over the same document,
        clamped by bounds, using a +/- window of neighbor sentences.
        """
        doc_id, sent_idx_in_doc = self.meta["sent_map"][global_sent_id]
        doc_global_ids = self.meta["doc_sent_spans"][doc_id]
        # find position of global_sent_id in doc_global_ids
        pos = doc_global_ids.index(global_sent_id)  # list is contiguous; index() is fine
        start_pos = max(0, pos - window)
        end_pos = min(len(doc_global_ids) - 1, pos + window)
        return doc_global_ids[start_pos], doc_global_ids[end_pos]

    def _merge_ranges(self, ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping or adjacent [start, end] ranges on the flattened sentence index space."""
        if not ranges:
            return []
        ranges = sorted(ranges, key=lambda x: x[0])
        merged = [ranges[0]]
        for cur in ranges[1:]:
            prev = merged[-1]
            if cur[0] <= prev[1] + 1:  # overlap or adjacency
                merged[-1] = (prev[0], max(prev[1], cur[1]))
            else:
                merged.append(cur)
        return merged

    def search(self, query: str, top_k: int = 5, window: int = 2) -> List[Dict]:
        """
        Retrieve top sentences for the query, then expand each hit with +/- `window` neighbor sentences
        from the same document. Finally, merge overlapping expansions and return coherent passages.

        Returns: List of dicts with:
          - content: merged passage text
          - source: originating document source
          - similarity_score: max similarity among the sentences inside the window (for ranking)
        """
        if self.index is None:
            raise RuntimeError("Index not loaded/built.")
        if not query.strip():
            return []

        # Encode query and search
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
        q = _normalize(q.astype("float32"))
        D, I = self.index.search(q, top_k)  # D: similarities, I: indices

        hits = I[0].tolist()
        sims = D[0].tolist()

        # Group expansions by document to create passages
        doc_to_ranges: Dict[int, List[Tuple[int, int]]] = {}
        doc_to_scores: Dict[int, List[float]] = {}

        for global_sent_id, score in zip(hits, sims):
            if global_sent_id < 0:
                continue
            doc_id, _ = self.meta["sent_map"][global_sent_id]
            start_g, end_g = self._expand_hit(global_sent_id, window)
            doc_to_ranges.setdefault(doc_id, []).append((start_g, end_g))
            doc_to_scores.setdefault(doc_id, []).append(score)

        results: List[Dict] = []
        # Build passages per document
        for doc_id, ranges in doc_to_ranges.items():
            merged = self._merge_ranges(ranges)

            for (start_g, end_g) in merged:
                # collect sentence strings
                sents = self.meta["sentences"][start_g : end_g + 1]
                passage = " ".join(sents).strip()
                source = self.meta["docs"][doc_id]["source"]

                # Score: take the max of sentence-level sims that fall inside the window range
                # (approximate ranking; good enough and stable)
                local_max = -1.0
                for global_sent_id, score in zip(hits, sims):
                    if start_g <= global_sent_id <= end_g:
                        local_max = max(local_max, score)
                if local_max < 0:
                    local_max = max(doc_to_scores.get(doc_id, [0.0]) or [0.0])

                results.append({
                    "content": passage,
                    "source": source,
                    "similarity_score": float(local_max),
                })

        # Sort by similarity desc, then length desc (prefer richer passages on ties)
        results.sort(key=lambda r: (r["similarity_score"], len(r["content"])), reverse=True)
        # Keep top_k passages overall
        return results[:top_k]
