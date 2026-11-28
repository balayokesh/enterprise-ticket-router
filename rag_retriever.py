import math
import os
import time

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

_genai_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing in .env")
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


def embed_query(query_text: str) -> list[float]:
    client = _get_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query_text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return _l2_normalize(list(response.embeddings[0].values))


def retrieve_relevant_runbooks(
    product: str,
    subject: str,
    description: str,
    top_k: int = 3,
    min_similarity: float = 0.35,
) -> tuple[list[dict], float]:
    """
    Retrieves top-k relevant runbook sections from pgvector.
    Returns (chunks_list, retrieval_latency_ms).
    """
    start = time.perf_counter()
    query_text = f"Product: {product}\nSubject: {subject}\nIssue: {description}"

    try:
        query_vec = embed_query(query_text)
        with psycopg.connect(DATABASE_URL, connect_timeout=5) as conn:
            register_vector(conn)
            rows = conn.execute(
                """
                SELECT
                    citation,
                    source_file,
                    content,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM runbook_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_vec, query_vec, top_k),
            ).fetchall()

        results = []
        for citation, source_file, content, similarity in rows:
            sim_score = round(float(similarity), 4)
            if sim_score >= min_similarity:
                results.append(
                    {
                        "citation": citation,
                        "source_file": source_file,
                        "content": content,
                        "similarity": sim_score,
                    }
                )
    except Exception as exc:
        print(f"[RAG Warning] Retrieval failed, falling back to zero-context triage: {exc}")
        results = []

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return results, latency_ms


if __name__ == "__main__":
    # Quick smoke test against dataset examples
    chunks, ms = retrieve_relevant_runbooks(
        product="Dell XPS",
        subject="Network problem",
        description="The Dell XPS is not turning on. I'm using the original charger, but it's not charging properly.",
    )
    print(f"Retrieved {len(chunks)} chunks in {ms} ms:\n")
    for c in chunks:
        print(f"- [{c['similarity']}] {c['citation']}")
        print(f"  {c['content'][:120]}...\n")
