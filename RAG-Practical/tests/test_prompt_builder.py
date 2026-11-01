
import pytest
from src.prompt_builder import build_prompt


def test_build_prompt():

    prompt = build_prompt(
        "Should I share my OTP with bank support?",
        ["Never share your OTP with anyone."]
    )

    assert "Customer Question:" in prompt
    assert "Bank Policy Context:" in prompt
    assert "Should I share my OTP with bank support?" in prompt


def test_account_number_masking():

    prompt = build_prompt(
        "My account number is 123456789012.",
        ["Contact your bank for assistance."]
    )

    assert "123456789012" not in prompt
    assert "[ACCOUNT_NUMBER_REDACTED]" in prompt
  

def test_multiple_context_chunks():

    prompt = build_prompt(
        "How do I protect my account?",
        [
            "Never share your OTP.",
            "Use a strong banking password."
        ]
    )

    assert "Never share your OTP." in prompt
    assert "Use a strong banking password." in prompt
  

def test_invalid_context():

    with pytest.raises((TypeError, ValueError)):
        build_prompt(
            "How do I contact my bank?",
            [None]
        )
