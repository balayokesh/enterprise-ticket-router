import math
import os
import re
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pgvector.psycopg import register_vector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ticket_rag")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = 768
KB_DIR = Path("knowledge_base")


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


def parse_markdown_sections(md_path: Path) -> list[dict]:
    """Splits a Markdown runbook into section-level chunks with citation metadata."""
    text = md_path.read_text(encoding="utf-8").strip()
    lines = text.splitlines()

    doc_title = md_path.stem.replace("_", " ").title()
    for line in lines:
        if line.startswith("# "):
            doc_title = line[2:].strip()
            break

    # Split on '## ' headings
    raw_sections = re.split(r"(?m)^##\s+", text)
    chunks = []

    for section in raw_sections[1:]:  # Skip header preamble before first '##'
        sec_lines = section.strip().splitlines()
        if not sec_lines:
            continue
        section_title = sec_lines[0].strip()
        body = "\n".join(sec_lines[1:]).strip()
        if not body:
            continue

        citation = f"{doc_title} — {section_title}"
        embed_text = f"Document: {doc_title}\nSection: {section_title}\nContent: {body}"

        chunks.append(
            {
                "chunk_id": f"{md_path.name}::{section_title}",
                "source_file": md_path.name,
                "doc_title": doc_title,
                "section_title": section_title,
                "citation": citation,
                "content": body,
                "embed_text": embed_text,
            }
        )
    return chunks


def get_embeddings(client: genai.Client, texts: list[str], task_type: str) -> list[list[float]]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return [l2_normalize(list(e.values)) for e in response.embeddings]


def init_db(conn: psycopg.Connection) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    register_vector(conn)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS runbook_chunks (
            chunk_id TEXT PRIMARY KEY,
            source_file TEXT NOT NULL,
            doc_title TEXT NOT NULL,
            section_title TEXT NOT NULL,
            citation TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding vector({EMBEDDING_DIM}) NOT NULL
        );
        """
    )


def ingest() -> None:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing in .env")

    md_files = sorted(KB_DIR.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {KB_DIR.resolve()}")

    all_chunks = []
    for md_file in md_files:
        all_chunks.extend(parse_markdown_sections(md_file))

    print(f"Parsed {len(all_chunks)} section chunks across {len(md_files)} runbooks.")

    client = genai.Client(api_key=GEMINI_API_KEY)
    embeddings = get_embeddings(
        client,
        [c["embed_text"] for c in all_chunks],
        task_type="RETRIEVAL_DOCUMENT",
    )

    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        init_db(conn)
        with conn.cursor() as cur:
            for chunk, emb in zip(all_chunks, embeddings):
                cur.execute(
                    """
                    INSERT INTO runbook_chunks (
                        chunk_id, source_file, doc_title, section_title, citation, content, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        source_file = EXCLUDED.source_file,
                        doc_title = EXCLUDED.doc_title,
                        section_title = EXCLUDED.section_title,
                        citation = EXCLUDED.citation,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding;
                    """,
                    (
                        chunk["chunk_id"],
                        chunk["source_file"],
                        chunk["doc_title"],
                        chunk["section_title"],
                        chunk["citation"],
                        chunk["content"],
                        emb,
                    ),
                )
        # Create HNSW cosine index for fast retrieval
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_runbook_chunks_embedding
            ON runbook_chunks USING hnsw (embedding vector_cosine_ops);
            """
        )

    print(f"Successfully upserted {len(all_chunks)} runbook sections into pgvector!")


if __name__ == "__main__":
    ingest()
