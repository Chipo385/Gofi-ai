"""
rag_pipeline.py
================
Gofi AI — RAG Pipeline (Zero ML-Framework Edition)
Uses a pure-Python TF-IDF embedding to index and search documents.
- No PyTorch required
- No TensorFlow required
- No sentence-transformers required
- Fully offline, no model downloads
- ChromaDB for persistence

For better accuracy (optional upgrade later):
  pip install sentence-transformers
  Then switch the embedding function in get_collection()

Usage:
    python rag_pipeline.py --index --docs_dir ./docs
    python rag_pipeline.py --query "Dot Com Zambia IPO results"
"""

import argparse
import math
import os
import re
import hashlib
from collections import Counter
from pathlib import Path
from typing import List, Optional

import chromadb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PERSIST_DIR     = "./chroma_db"
COLLECTION_NAME = "luse_knowledge_base"
CHUNK_SIZE      = 700
CHUNK_OVERLAP   = 80
TOP_K           = 5


# ---------------------------------------------------------------------------
# 1. Pure Python TF-IDF Embedding Function (works without ANY ML framework)
# ---------------------------------------------------------------------------

try:
    from chromadb import EmbeddingFunction
except ImportError:
    class EmbeddingFunction:
        pass

class TFIDFEmbeddingFunction(EmbeddingFunction):
    """
    Simple bag-of-words TF-IDF embedding.
    Produces 512-dim sparse-to-dense vectors — good enough for keyword-level
    semantic retrieval of financial news and reports.
    No downloads, no GPU, no external deps beyond stdlib.
    """
    DIM = 512

    def __init__(self):
        super().__init__()

    def name(self) -> str:
        return "TFIDFEmbeddingFunction"

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if len(t) > 2]

    def _embed(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        total = max(len(tokens), 1)
        vector = [0.0] * self.DIM
        for token, count in tf.items():
            # Map token to bucket via hash
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.DIM
            tf_score = count / total
            # Simple IDF approximation: log(1 + 1/rank)
            idf_score = math.log(1 + 10 / max(count, 1))
            vector[bucket] += tf_score * idf_score
        # L2 normalise
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]


# Singleton instances
_tfidf_ef = TFIDFEmbeddingFunction()


# ---------------------------------------------------------------------------
# 2. Text Chunker
# ---------------------------------------------------------------------------

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks on paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            while len(para) > size:
                chunks.append(para[:size])
                para = para[size - overlap:]
            current = para
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) > 50]


# ---------------------------------------------------------------------------
# 3. Document Loaders
# ---------------------------------------------------------------------------

def load_txt(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
        r = PdfReader(str(path))
        return "\n\n".join(p.extract_text() or "" for p in r.pages)
    except Exception as e:
        print(f"[RAG] PDF read error ({path.name}): {e}")
        return ""


def load_documents(docs_dir: str = "./docs") -> List[dict]:
    p = Path(docs_dir)
    if not p.exists():
        raise FileNotFoundError(f"'{docs_dir}' not found. Run: python news_scraper.py")
    docs = []
    for f in p.rglob("*.txt"):
        c = load_txt(f)
        if c.strip():
            docs.append({"source": str(f), "content": c})
    for f in p.rglob("*.pdf"):
        c = load_pdf(f)
        if c.strip():
            docs.append({"source": str(f), "content": c})
    print(f"[RAG] Loaded {len(docs)} document(s)")
    return docs


# ---------------------------------------------------------------------------
# 4. ChromaDB Client
# ---------------------------------------------------------------------------

def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=PERSIST_DIR)


def _get_collection(client, create: bool = False):
    if create:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        return client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=_tfidf_ef,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=_tfidf_ef,
    )


# ---------------------------------------------------------------------------
# 5. Build Index
# ---------------------------------------------------------------------------

def build_vector_store(docs_dir: str = "./docs") -> None:
    documents = load_documents(docs_dir)
    if not documents:
        print("[RAG] No documents to index.")
        return

    client     = _get_client()
    collection = _get_collection(client, create=True)

    all_texts, all_ids, all_metas = [], [], []
    for idx, doc in enumerate(documents):
        for chunk in chunk_text(doc["content"]):
            all_texts.append(chunk)
            all_ids.append(f"doc_{idx}_{len(all_ids)}")
            all_metas.append({"source": doc["source"]})

    # Batch insert (ChromaDB safe batch = 500)
    batch = 500
    for i in range(0, len(all_texts), batch):
        collection.upsert(
            ids=all_ids[i: i + batch],
            documents=all_texts[i: i + batch],
            metadatas=all_metas[i: i + batch],
        )
        print(f"[RAG]   Indexed batch {i // batch + 1} ({min(i + batch, len(all_texts))}/{len(all_texts)} chunks)")

    print(f"[RAG] ✓ Done. {len(all_texts)} chunks indexed → {PERSIST_DIR}")


# ---------------------------------------------------------------------------
# 6. Retrieve Context (used by chatbot.py)
# ---------------------------------------------------------------------------

def retrieve_context(query: str, top_k: int = TOP_K) -> str:
    """Returns top-k relevant chunks as a single formatted string."""
    client     = _get_client()
    collection = _get_collection(client, create=False)
    count      = collection.count()
    if count == 0:
        return "Knowledge base is empty. Run: python rag_pipeline.py --index"

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, count),
    )
    docs   = results.get("documents", [[]])[0]
    metas  = results.get("metadatas", [[]])[0]

    if not docs:
        return "No relevant context found."

    chunks = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        src = Path(meta.get("source", "Unknown")).name
        # Strip timestamp suffix from scraper filenames
        src_clean = re.sub(r"_\d{8}_\d{6}\.txt$", ".txt", src)
        chunks.append(f"[Context {i} | {src_clean}]\n{doc.strip()}")
    return "\n\n---\n\n".join(chunks)


# ---------------------------------------------------------------------------
# 7. API compat shims (for chatbot.py)
# ---------------------------------------------------------------------------

def get_embeddings():
    return _tfidf_ef


def load_vector_store(embeddings=None):
    """Returns the ChromaDB collection (embeddings arg kept for compat)."""
    if not Path(PERSIST_DIR).exists():
        raise RuntimeError(f"No store at '{PERSIST_DIR}'. Run: python rag_pipeline.py --index")
    return _get_collection(_get_client(), create=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gofi AI RAG Pipeline")
    parser.add_argument("--index",    action="store_true")
    parser.add_argument("--query",    type=str, default=None)
    parser.add_argument("--docs_dir", type=str, default="./docs")
    args = parser.parse_args()

    if args.index:
        build_vector_store(args.docs_dir)

    if args.query:
        print(f"\n{'='*60}\nQuery: {args.query}\n{'='*60}")
        print(retrieve_context(args.query))
