from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "output"
CHUNK_OUTPUT = BASE_DIR / "output" / "chunks.json"

CHUNK_SIZE = 50
CHUNK_OVERLAP = 10


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


def main():
    txt_files = sorted(INPUT_DIR.glob("*.txt"))

    if not txt_files:
        print("No text files found in output/")
        return

    all_chunks = []

    for file_path in txt_files:
        text = file_path.read_text(encoding="utf-8")
        chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for index, chunk in enumerate(chunks, start=1):
            all_chunks.append({
                "chunk_id": f"{file_path.stem}_{index}",
                "source": file_path.name,
                "chunk_index": index,
                "text": chunk,
            })

    CHUNK_OUTPUT.write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved chunks to: {CHUNK_OUTPUT}")

    for chunk in all_chunks:
        print(f"\n--- {chunk['chunk_id']} ---")
        print(chunk["text"])


if __name__ == "__main__":
    main()