
from pathlib import Path
from html import escape

from src.security import prepare_for_llm

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_FILE = BASE_DIR / "prompts" / "bank_assistant_v1.txt"


def build_prompt(question: str, context_chunks: list[str]) -> str:
    """Build a prompt with sanitized, escaped untrusted input."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")

    if not context_chunks:
        raise ValueError("context_chunks cannot be empty.")

    # Sanitize customer question
    safe_question = prepare_for_llm(question)

    # Sanitize retrieved documents
    safe_chunks = [
        prepare_for_llm(chunk)
        for chunk in context_chunks
    ]

    # Escape special HTML/XML characters in untrusted content.
    # This prevents retrieved text from closing our context tags.
    safe_question = escape(safe_question, quote=False)

    safe_chunks = [
        escape(chunk, quote=False)
        for chunk in safe_chunks
    ]

    # Combine retrieved chunks
    context = "\n\n".join(safe_chunks)

    # Load prompt template
    template = PROMPT_FILE.read_text(encoding="utf-8")

    # Insert dynamic values
    prompt = template.format(
        context=context,
        question=safe_question
    )

    return prompt
