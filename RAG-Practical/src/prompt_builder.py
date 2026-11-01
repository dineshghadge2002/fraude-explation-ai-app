
from pathlib import Path
from src.security import prepare_for_llm

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_FILE = BASE_DIR / "prompts" / "bank_assistant_v1.txt"


def build_prompt(question: str, context_chunks: list[str]) -> str:

    # Sanitize customer question
    safe_question = prepare_for_llm(question)

    # Sanitize retrieved documents
    safe_chunks = [
        prepare_for_llm(chunk)
        for chunk in context_chunks
    ]

    # Combine retrieved chunks
    context = "\n\n".join(safe_chunks)

    # Load external prompt template
    template = PROMPT_FILE.read_text(encoding="utf-8")

    # Insert dynamic values
    prompt = template.format(
        context=context,
        question=safe_question
    )

    # Final security validation
    return prepare_for_llm(prompt)
