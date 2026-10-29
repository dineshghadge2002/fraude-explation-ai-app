
from pathlib import Path
import pymupdf

BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
OUTPUT_DIR = BASE_DIR / "output"


def extract_pdf_text(pdf_path: Path) -> list[dict]:
    pages = []

    with pymupdf.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            pages.append({
                "source": pdf_path.name,
                "page": page_number,
                "text": page.get_text("text").strip(),
            })

    return pages


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    for pdf_path in pdf_files:
        try:
            pages = extract_pdf_text(pdf_path)

            extracted_text = "\n\n".join(
                f"--- Page {page['page']} ---\n{page['text']}"
                for page in pages
            )

            output_path = OUTPUT_DIR / f"{pdf_path.stem}.txt"
            output_path.write_text(
                extracted_text,
                encoding="utf-8",
            )

            print(f"\nFile: {pdf_path.name}")
            print(f"Pages: {len(pages)}")
            print(f"Characters: {len(extracted_text)}")
            print(f"Saved to: {output_path}")
            print("\nPreview:")
            print(extracted_text[:500])

            if not extracted_text.strip():
                print("WARNING: No text extracted; OCR may be needed.")

        except Exception as exc:
            print(f"Failed to process {pdf_path.name}: {exc}")


if __name__ == "__main__":
    main()