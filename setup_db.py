# setup_db.py
import os
from dotenv import load_dotenv
import psycopg2
import redis

load_dotenv()

print("🔧 Diamond Database Setup")

# PostgreSQL + pgvector
try:
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", 15432)),
        user=os.getenv("PG_USER", "diamond"),
        password=os.getenv("PG_PASSWORD"),
        dbname=os.getenv("PG_DB", "diamond_rag")
    )
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rag_chunks (
            id SERIAL PRIMARY KEY,
            file TEXT,
            chunk TEXT,
            embedding VECTOR(1024)
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ PostgreSQL table 'rag_chunks' created successfully")
except Exception as e:
    print(f"❌ Postgres error: {e}")

# Redis (just test + optional flush)
try:
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 63791)),
        password=os.getenv("REDIS_PASSWORD"),
        decode_responses=True
    )
    r.ping()
    print("✅ Redis connected")
    if input("Do you want to flush Redis cache? (y/n): ").lower() == "y":
        r.flushdb()
        print("✅ Redis cache flushed")
except Exception as e:
    print(f"❌ Redis error: {e}")

print("✅ Setup complete! You can now run diamond.py")
