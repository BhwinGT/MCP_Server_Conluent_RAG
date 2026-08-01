"""
embed_store.py
--------------
Reads the pre-chunked records from fetched_docs.json (produced by
scrape_support.py) and stores them in Qdrant with BOTH a dense vector
(semantic search) and a sparse BM25 vector (lexical/keyword search) per
point, enabling true hybrid search at query time.

Run AFTER scrape_support.py, and with Qdrant running:
    docker compose run --rm ingestion python ingest/embed_store.py

Usage:
    python ingest/embed_store.py           # embed new + refresh updated articles
    python ingest/embed_store.py --reset   # wipe collection and re-embed everything

Change detection keys on (article_id, updated_at). Zendesk article IDs are
immutable across edits, so a changed article keeps its ID and only its
updated_at moves. Updated articles are deleted and re-inserted rather than
upserted, so a shortened article leaves no orphan chunks behind.

Why two vectors per point:
  - "dense"  -- qwen3-embedding:0.6b via Ollama, captures semantic meaning
  - "bm25"   -- fastembed's Qdrant/bm25 sparse model, captures exact keyword
                 matches (error codes, config keys, exact terms embeddings
                 tend to blur together)
  Both live on the same point so a single hybrid query can fuse both
  rankings via Reciprocal Rank Fusion (see qdrant_server.py).
"""

import os
import sys
import json
import uuid
from collections import defaultdict
import hashlib
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseVector, PointStruct,
    Filter, FieldCondition, MatchValue, FilterSelector,
)
from fastembed import SparseTextEmbedding
from ollama import Client as OllamaClient
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL      = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
SPARSE_MODEL     = "Qdrant/bm25"
QDRANT_HOST      = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT      = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME  = os.getenv("QDRANT_COLLECTION", "confluent_support")
INPUT_FILE       = Path("./ingest/fetched_docs.json")
BATCH_SIZE       = 64

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

FASTEMBED_CACHE_DIR = os.getenv("FASTEMBED_CACHE_DIR", "./model_cache")

_ollama_client = OllamaClient(host=OLLAMA_HOST)
_sparse_model  = SparseTextEmbedding(model_name=SPARSE_MODEL, cache_dir=FASTEMBED_CACHE_DIR)


def point_id(article_id, chunk_index: int) -> str:
    """Stable UUID derived from article_id + chunk_index (Qdrant requires UUID or int IDs)."""
    md5_hex = hashlib.md5(f"{article_id}::{chunk_index}".encode()).hexdigest()
    return str(uuid.UUID(hex=md5_hex))


def embed_dense_batch(texts: list[str]) -> list[list[float]]:
    return _ollama_client.embed(model=EMBED_MODEL, input=texts)["embeddings"]


def embed_sparse_batch(texts: list[str]) -> list[SparseVector]:
    vectors = []
    for sparse_emb in _sparse_model.embed(texts):
        vectors.append(SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist(),
        ))
    return vectors


def get_dense_dim() -> int:
    test_vec = _ollama_client.embed(model=EMBED_MODEL, input=["test"])["embeddings"][0]
    return len(test_vec)


def load_existing_articles(client: QdrantClient) -> dict[str, str]:
    """
    Returns {article_id: updated_at} for stored official articles.

    Only source_type="official_article" is read, so saved conversations are
    never considered for refresh or deletion by ingestion.
    """
    stored: dict[str, str] = {}
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[FieldCondition(
                key="source_type", match=MatchValue(value="official_article"))]),
            limit=5000,
            offset=next_offset,
            with_payload=["article_id", "updated_at"],
            with_vectors=False,
        )
        for p in points:
            aid = p.payload.get("article_id")
            if aid is not None:
                stored[str(aid)] = p.payload.get("updated_at")
        if next_offset is None:
            break
    return stored


def needs_refresh(stored_updated_at, incoming_updated_at) -> bool:
    """A missing or unreadable timestamp on either side counts as changed."""
    if not stored_updated_at or not incoming_updated_at:
        return True
    return str(incoming_updated_at) > str(stored_updated_at)


def delete_article(client: QdrantClient, article_id) -> None:
    """
    Drops every point for one article before its fresh chunks go in.

    Deleting first is what clears orphans: if an edit shortens an article from
    5 chunks to 3, upserting alone would leave chunks 3 and 4 behind.
    """
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(filter=Filter(must=[
            FieldCondition(key="article_id", match=MatchValue(value=article_id))])),
    )


def main():
    reset = "--reset" in sys.argv

    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Run scrape_support.py first.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} pre-chunked record(s)")

    print(f"Checking Ollama embed model '{EMBED_MODEL}' ...")
    try:
        dense_dim = get_dense_dim()
        print(f"  '{EMBED_MODEL}' is ready (vector size: {dense_dim})")
    except Exception as e:
        print(f"  Cannot reach Ollama or model not pulled: {e}")
        print(f"  Run:  ollama pull {EMBED_MODEL}")
        return

    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT} ...")
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.get_collections()
    except Exception as e:
        print(f"  Cannot reach Qdrant: {e}")
        print("  Make sure the qdrant service is running: docker compose up -d qdrant")
        return
    print("  Connected")

    collection_exists = client.collection_exists(COLLECTION_NAME)

    if reset and collection_exists:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")
        collection_exists = False

    if not collection_exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=dense_dim, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "bm25": SparseVectorParams(),
            },
        )
        print(f"Created collection '{COLLECTION_NAME}' (dense={dense_dim}, + bm25 sparse)")

    existing_count = client.count(COLLECTION_NAME).count
    print(f"Qdrant collection '{COLLECTION_NAME}' ready")
    print(f"  Existing points : {existing_count:,}")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  Mode            : {'full re-embed (--reset)' if reset else 'resume'}")

    stored = {} if reset else load_existing_articles(client)
    if stored:
        print(f"  Loaded {len(stored):,} stored article(s) for change detection")

    by_article: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_article[str(rec["article_id"])].append(rec)

    new_records = []
    n_new = n_updated = n_unchanged = 0
    to_delete = []
    for aid, recs in by_article.items():
        if aid not in stored:
            n_new += 1
        elif needs_refresh(stored[aid], recs[0].get("updated_at")):
            n_updated += 1
            # The raw value, not the string key: article_id is stored as an int
            # and a string MatchValue silently matches nothing.
            to_delete.append(recs[0]["article_id"])
        else:
            n_unchanged += 1
            continue
        for rec in recs:
            new_records.append((point_id(rec["article_id"], rec["chunk_index"]), rec))

    total_new = len(new_records)
    print(f"  Articles        : {n_new:,} new | {n_updated:,} updated | {n_unchanged:,} unchanged")
    print(f"  Chunks to embed : {total_new:,} of {len(records):,}")

    if total_new == 0:
        print("Nothing to embed -- everything is up to date.")
        return

    # Clear updated articles first so a shortened article leaves no orphans.
    for aid in to_delete:
        delete_article(client, aid)
    if to_delete:
        print(f"  Cleared {len(to_delete):,} updated article(s) before re-embedding")

    added = 0
    with tqdm(total=total_new, desc="Embedding chunks (dense + sparse)", unit="chunk") as pbar:
        for b_start in range(0, total_new, BATCH_SIZE):
            batch = new_records[b_start:b_start + BATCH_SIZE]
            texts = [rec["text"] for _, rec in batch]

            dense_vecs  = embed_dense_batch(texts)
            sparse_vecs = embed_sparse_batch(texts)

            points = [
                PointStruct(
                    id=pid,
                    vector={"dense": dense_vecs[i], "bm25": sparse_vecs[i]},
                    payload={
                        "text": rec["text"],
                        "url": rec["url"],
                        "title": rec["title"],
                        "article_id": rec["article_id"],
                        "section_id": rec.get("section_id"),
                        "updated_at": rec.get("updated_at"),
                        "chunk_index": rec["chunk_index"],
                        "source_type": "official_article",
                        "source_ai": "",
                        "applies_to": "",
                    },
                )
                for i, (pid, rec) in enumerate(batch)
            ]
            client.upsert(collection_name=COLLECTION_NAME, points=points)

            added += len(points)
            pbar.update(len(points))

    print(f"\nDone. Embedded {added:,} chunk(s) across "
          f"{n_new:,} new + {n_updated:,} updated article(s); "
          f"{n_unchanged:,} article(s) already current.")
    print(f"Total points in collection: {client.count(COLLECTION_NAME).count:,}")


if __name__ == "__main__":
    main()