import os
import csv
import hashlib
import logging
from pathlib import Path
from typing import List

import psycopg2
from llama_cpp import Llama

# ==========================================================
# CONFIG
# ==========================================================

CSV_FOLDER = "./"  # Folder containing ALL CSV files
EMBED_MODEL_PATH = "/mnt/storage/webui/models/Qwen3-Embedding-0.6B-f16_2.gguf"

PG_HOST = "127.0.0.1"
PG_PORT = 15432
PG_USER = "diamond"
PG_PASSWORD = 'c47ae8b8d4d1f6fc01e8ba3da2c29c0973b4f0f9'
PG_DB = "diamond_rag"

VECTOR_DIM = 1024
BATCH_SIZE = 32  # Increase to 64 if VRAM allows (3090 can handle it)

logging.basicConfig(level=logging.INFO)

# ==========================================================
# UTILITIES
# ==========================================================

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def clean_text(text: str) -> str:
    text = text.strip()
    text = " ".join(text.split())
    return text

def meaningful(text: str) -> bool:
    if not text:
        return False
    if len(text.split()) < 5:
        return False
    return True

# ==========================================================
# DATABASE SETUP
# ==========================================================

def setup_db():
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DB
    )

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS rag_chunks (
            id SERIAL PRIMARY KEY,
            source TEXT,
            chunk TEXT,
            content_hash TEXT UNIQUE,
            embedding VECTOR({VECTOR_DIM})
        );
        """)
    conn.commit()
    return conn

# ==========================================================
# INSERT BATCH
# ==========================================================

def insert_batch(conn, source, texts: List[str], embeddings: List[List[float]]):
    with conn.cursor() as cur:
        for text, emb in zip(texts, embeddings):
            content_hash = hash_text(text)
            cur.execute("""
                INSERT INTO rag_chunks (source, chunk, content_hash, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (content_hash) DO NOTHING;
            """, (source, text, content_hash, emb))
    conn.commit()

# ==========================================================
# MAIN EMBEDDING LOGIC
# ==========================================================

def embed_all_csv():

    # Load embedding model
    logging.info("🚀 Loading embedding model (GPU)...")
    embedder = Llama(
        model_path=EMBED_MODEL_PATH,
        embedding=True,
        n_gpu_layers=-1,
        n_ctx=512,
        verbose=False
    )

    conn = setup_db()

    total_inserted = 0

    for csv_file in Path(CSV_FOLDER).glob("*.csv"):

        logging.info(f"📄 Processing {csv_file.name}")
        source_name = csv_file.stem

        batch_texts = []

        with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)

            for row in reader:
                if not row:
                    continue

                text = clean_text(" ".join(row))

                if not meaningful(text):
                    continue

                batch_texts.append(text)

                if len(batch_texts) >= BATCH_SIZE:

                    embeddings = embedder.embed(batch_texts)
                    insert_batch(conn, source_name, batch_texts, embeddings)

                    total_inserted += len(batch_texts)
                    batch_texts = []

        # Remaining rows
        if batch_texts:
            embeddings = embedder.embed(batch_texts)
            insert_batch(conn, source_name, batch_texts, embeddings)
            total_inserted += len(batch_texts)

    logging.info(f"✅ Done. Total embedded rows: {total_inserted}")

# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    embed_all_csv()
