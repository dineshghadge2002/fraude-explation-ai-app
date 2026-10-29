
import re

PII_PATTERNS = {
    "EMAIL": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),
    "PHONE": re.compile(
        r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"
    ),
    "AADHAAR": re.compile(
        r"(?<![\d])(?:\d{4}[\s-]?){2}\d{4}(?!\d)"
        r"|(?<![\d])\d{12}(?!\d)"
    ),
    "PAN": re.compile(
        r"\b[A-Z]{5}\d{4}[A-Z]\b",
        re.IGNORECASE,
    ),

    "ACCOUNT_NUMBER": re.compile(
        r"\b(?:account\s*(?:number|no\.?|#)|a/c\s*(?:no\.?|number|#)?)"
        r"\s*(?:(?:is|:|=|#|-)\s*)?"
        r"(?:\d[\s-]?){8,18}\b",
        re.IGNORECASE,
    ),

}


def mask_pii(text: str) -> str:
    """Mask common PII patterns before sending text to an LLM."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    sanitized = text

    # Mask labelled account numbers before Aadhaar.
    sanitized = PII_PATTERNS["ACCOUNT_NUMBER"].sub(
        "[ACCOUNT_NUMBER_REDACTED]",
        sanitized,
    )

    # Mask remaining PII.
    for pii_type in ("AADHAAR", "PAN", "EMAIL", "PHONE"):
        sanitized = PII_PATTERNS[pii_type].sub(
            f"[{pii_type}_REDACTED]",
            sanitized,
        )

    return sanitized



def contains_pii(text: str) -> bool:
    """Return True if a supported PII pattern remains detectable."""
    return any(
        pattern.search(text)
        for pattern in PII_PATTERNS.values()
    )


def prepare_for_llm(text: str) -> str:
    """Fail closed if detectable PII remains after masking."""
    sanitized = mask_pii(text)

    if contains_pii(sanitized):
        raise ValueError(
            "Potential PII remains. Do not send this text to the LLM."
        )

    return sanitized
