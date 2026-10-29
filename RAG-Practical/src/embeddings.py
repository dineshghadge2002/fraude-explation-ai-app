from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "output" / "chunks.json"
EMBEDDINGS_PATH = BASE_DIR / "output" / "embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {CHUNKS_PATH}. "
            "Run src/chunking.py first."
        )

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    if not chunks:
        raise ValueError("No chunks found to embed.")

    texts = [chunk["text"] for chunk in chunks]

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    np.save(EMBEDDINGS_PATH, embeddings)

    print(f"Number of chunks: {len(texts)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Vector dimensions: {embeddings.shape[1]}")
    print(f"Saved embeddings to: {EMBEDDINGS_PATH}")
    print(f"First vector preview: {embeddings[0][:5]}")


if __name__ == "__main__":
    main()