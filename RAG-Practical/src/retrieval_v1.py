
from pathlib import Path
import json
import faiss
from sentence_transformers import SentenceTransformer
from src.security import prepare_for_llm

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = BASE_DIR / "output" / "chunks.json"
INDEX_PATH = BASE_DIR / "output" / "faiss.index"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 2
MIN_SIMILARITY_SCORE = 0.20


def retrieve(query: str, model, index, chunks: list[dict]):
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    scores, indices = index.search(
        query_vector,
        min(TOP_K, len(chunks)),
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        results.append({
            "score": float(score),
            "chunk_id": chunks[idx]["chunk_id"],
            "source": chunks[idx]["source"],
            "text": chunks[idx]["text"],
        })

    return results


def main():
    if not CHUNKS_PATH.exists() or not INDEX_PATH.exists():
        raise FileNotFoundError(
            "Missing chunks or FAISS index. "
            "Run chunking, embeddings, and vector_store first."
        )

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    index = faiss.read_index(str(INDEX_PATH))

    if not chunks or index.ntotal != len(chunks):
        raise ValueError("Chunk count and FAISS index size do not match.")

    model = SentenceTransformer(MODEL_NAME)

    query = input("Ask a question about the bank policy: ")

    # Layer 1: Sanitize the user question
    safe_query = prepare_for_llm(query)

    # Retrieve using the sanitized question
    results = retrieve(safe_query, model, index, chunks)

    relevant_results = [
        result
        for result in results
        if result["score"] >= MIN_SIMILARITY_SCORE
    ]
    # Layer 2: Sanitize retrieved document chunks
    for result in relevant_results:
        result["text"] = prepare_for_llm(result["text"])

    
    print("\nRetrieved similarity scores:")

    for result in results:
        status = (
            "ACCEPTED"
            if result["score"] >= MIN_SIMILARITY_SCORE
            else "REJECTED"
        )
        print(
            f"{result['chunk_id']}: "
            f"{result['score']:.4f} - {status}"
        )

    if not relevant_results:
        print(
            "\nI couldn't find relevant information "
            "in the provided documents."
        )
        return

    print("\nRelevant results:")
    for result in relevant_results:
        print(f"\nChunk ID: {result['chunk_id']}")
        print(f"Source: {result['source']}")
        print(f"Similarity score: {result['score']:.4f}")
        print(f"Text:\n{result['text']}")


if __name__ == "__main__":
    main()
