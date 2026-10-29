
from src.security import prepare_for_llm, contains_pii


def test_account_number_is_masked():
    text = "My account number is 123456789012."
    safe_text = prepare_for_llm(text)

    assert "123456789012" not in safe_text
    assert "[ACCOUNT_NUMBER_REDACTED]" in safe_text


def test_email_is_masked():
    text = "Contact me at customer@example.com."
    safe_text = prepare_for_llm(text)

    assert "customer@example.com" not in safe_text
    assert "[EMAIL_REDACTED]" in safe_text


def test_phone_is_masked():
    text = "My phone number is 9876543210."
    safe_text = prepare_for_llm(text)

    assert "9876543210" not in safe_text
    assert "[PHONE_REDACTED]" in safe_text


def test_pan_is_masked():
    text = "My PAN is ABCDE1234F."
    safe_text = prepare_for_llm(text)

    assert "ABCDE1234F" not in safe_text
    assert "[PAN_REDACTED]" in safe_text


def test_aadhaar_is_masked():
    text = "My Aadhaar number is 2345 6789 0123."
    safe_text = prepare_for_llm(text)

    assert "2345 6789 0123" not in safe_text
    assert "[AADHAAR_REDACTED]" in safe_text


def test_no_pii_text_remains_unchanged():
    text = "What is the bank's OTP policy?"
    assert prepare_for_llm(text) == text


def test_pii_detector():
    assert contains_pii("Email: customer@example.com")
