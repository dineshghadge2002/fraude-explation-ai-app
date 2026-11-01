
import json
import faiss

from sentence_transformers import SentenceTransformer

from src.security import prepare_for_llm
from src.retrieval_v1 import (
    BASE_DIR,
    CHUNKS_PATH,
    INDEX_PATH,
    MODEL_NAME,
    TOP_K,
    MIN_SIMILARITY_SCORE,
    retrieve,
)
from src.prompt_builder import build_prompt


def load_resources():
    """Load FAISS index, chunks, and embedding model."""

    if not CHUNKS_PATH.exists() or not INDEX_PATH.exists():
        raise FileNotFoundError(
            "Missing chunks or FAISS index."
        )

    chunks = json.loads(
        CHUNKS_PATH.read_text(encoding="utf-8")
    )

    index = faiss.read_index(str(INDEX_PATH))

    if not chunks or index.ntotal != len(chunks):
        raise ValueError(
            "Chunk count and FAISS index size do not match."
        )

    model = SentenceTransformer(MODEL_NAME)

    return model, index, chunks


def build_rag_prompt(question, model, index, chunks):
    """Retrieve real document chunks and build the prompt."""

    safe_question = prepare_for_llm(question)

    # Search FAISS vector database
    results = retrieve(
        safe_question,
        model,
        index,
        chunks,
    )

    # Apply similarity threshold
    relevant_results = [
        result
        for result in results
        if result["score"] >= MIN_SIMILARITY_SCORE
    ]

    print("\nRetrieval Results:")

    for result in results:
        status = (
            "ACCEPTED"
            if result["score"] >= MIN_SIMILARITY_SCORE
            else "REJECTED"
        )

        print(
            f"{result['chunk_id']} | "
            f"Score: {result['score']:.4f} | "
            f"{status}"
        )

    if not relevant_results:
        return None, []

    # Extract actual text returned by FAISS retrieval
    context_chunks = [
        result["text"]
        for result in relevant_results
    ]

    # Build prompt using actual retrieved context
    prompt = build_prompt(
        question=safe_question,
        context_chunks=context_chunks,
    )

    return prompt, relevant_results


def main():
    model, index, chunks = load_resources()

    question = input(
        "\nAsk a question about the bank policy: "
    )

    prompt, results = build_rag_prompt(
        question,
        model,
        index,
        chunks,
    )

    if prompt is None:
        print(
            "\nNo relevant information found "
            "in the available documents."
        )
        return

    print("\n========== ACTUAL RETRIEVED CONTEXT ==========")

    for result in results:
        print(f"\nChunk ID: {result['chunk_id']}")
        print(f"Source: {result['source']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text']}")

    print("\n========== FINAL GENERATED PROMPT ==========\n")
    print(prompt)


if __name__ == "__main__":
    main()
