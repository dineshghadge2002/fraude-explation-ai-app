from pathlib import Path
import json
import numpy as np
import faiss

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "output" / "chunks.json"
EMBEDDINGS_PATH = BASE_DIR / "output" / "embeddings.npy"
INDEX_PATH = BASE_DIR / "output" / "faiss.index"


def main():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    embeddings = np.load(EMBEDDINGS_PATH)

    if len(chunks) != len(embeddings):
        raise ValueError("Chunk count and embedding count do not match.")

    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("Embeddings must be a non-empty 2D array.")

    dimension = embeddings.shape[1]

    # Exact inner-product search; normalized vectors give cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))

    faiss.write_index(index, str(INDEX_PATH))

    print(f"Vectors indexed: {index.ntotal}")
    print(f"Vector dimensions: {dimension}")
    print(f"FAISS index saved to: {INDEX_PATH}")


if __name__ == "__main__":
    main()