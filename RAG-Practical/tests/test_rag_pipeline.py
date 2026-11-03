
from unittest.mock import patch

import pytest

from src.rag_pipeline import build_rag_prompt


@pytest.fixture
def fake_model():
    return object()


@pytest.fixture
def fake_index():
    return object()


@pytest.fixture
def fake_chunks():
    return [
        {
            "chunk_id": "bank_policy_1",
            "source": "bank_policy.txt",
            "text": "Never share your PIN, password, or OTP.",
        }
    ]


def show_result(test_name, question, actual, expected):
    print("\n" + "=" * 60)
    print(f"TEST: {test_name}")
    print(f"INPUT: {question}")
    print(f"ACTUAL OUTPUT: {actual}")
    print(f"EXPECTED OUTPUT: {expected}")
    print("=" * 60)


def test_relevant_question_builds_prompt(
    fake_model, fake_index, fake_chunks
):
    question = "How can I protect my account?"

    results = [
        {
            "score": 0.75,
            "chunk_id": "bank_policy_1",
            "source": "bank_policy.txt",
            "text": "Never share your PIN, password, or OTP.",
        }
    ]

    with patch(
        "src.rag_pipeline.retrieve",
        return_value=results,
    ), patch(
        "src.rag_pipeline.build_prompt",
        return_value="Generated test prompt",
    ) as mock_builder:

        prompt, retrieved = build_rag_prompt(
            question, fake_model, fake_index, fake_chunks
        )

    actual = {
        "prompt": prompt,
        "retrieved_chunks": len(retrieved),
        "context": retrieved[0]["text"],
    }

    expected = {
        "prompt": "Generated test prompt",
        "retrieved_chunks": 1,
        "context": "Never share your PIN, password, or OTP.",
    }

    show_result("Relevant question", question, actual, expected)

    assert prompt == "Generated test prompt"
    assert len(retrieved) == 1
    mock_builder.assert_called_once()


def test_irrelevant_question_returns_no_prompt(
    fake_model, fake_index, fake_chunks
):
    question = "What is Git?"

    results = [
        {
            "score": 0.05,
            "chunk_id": "bank_policy_1",
            "source": "bank_policy.txt",
            "text": "Never share your PIN.",
        }
    ]

    with patch(
        "src.rag_pipeline.retrieve",
        return_value=results,
    ), patch(
        "src.rag_pipeline.build_prompt"
    ) as mock_builder:

        prompt, retrieved = build_rag_prompt(
            question, fake_model, fake_index, fake_chunks
        )

    actual = {
        "prompt": prompt,
        "retrieved_chunks": len(retrieved),
    }

    expected = {
        "prompt": None,
        "retrieved_chunks": 0,
    }

    show_result("Irrelevant question", question, actual, expected)

    assert prompt is None
    assert retrieved == []
    mock_builder.assert_not_called()


def test_account_number_is_masked_before_retrieval(
    fake_model, fake_index, fake_chunks
):
    question = "My account number is 123456789012. Help me."

    results = [
        {
            "score": 0.75,
            "chunk_id": "bank_policy_1",
            "source": "bank_policy.txt",
            "text": "Never share your PIN.",
        }
    ]

    with patch(
        "src.rag_pipeline.retrieve",
        return_value=results,
    ) as mock_retrieve, patch(
        "src.rag_pipeline.build_prompt",
        return_value="Generated test prompt",
    ):

        prompt, retrieved = build_rag_prompt(
            question, fake_model, fake_index, fake_chunks
        )

    sanitized_query = mock_retrieve.call_args.args[0]

    actual = {
        "sanitized_query": sanitized_query,
        "original_account_number_present": (
            "123456789012" in sanitized_query
        ),
    }

    expected = {
        "sanitized_query": (
            "My account number is "
            "[ACCOUNT_NUMBER_REDACTED]. Help me."
        ),
        "original_account_number_present": False,
    }

    show_result("Account number masking", question, actual, expected)

    assert "123456789012" not in sanitized_query
    assert "[ACCOUNT_NUMBER_REDACTED]" in sanitized_query


def test_empty_question_is_rejected(
    fake_model, fake_index, fake_chunks
):
    question = ""

    try:
        build_rag_prompt(
            question, fake_model, fake_index, fake_chunks
        )
        actual = "No exception raised"
    except ValueError as error:
        actual = f"{type(error).__name__}: {error}"

    expected = "ValueError: text must be a string"  # Updated below
    show_result("Empty question", question, actual,
                "ValueError is raised")

    assert actual.startswith("ValueError:")
