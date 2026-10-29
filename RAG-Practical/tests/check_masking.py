
from src.security import prepare_for_llm

samples = [
    "My account number is 123456789012.",
    "My email is customer@example.com.",
    "My phone number is 9876543210.",
    "My PAN is ABCDE1234F.",
    "My Aadhaar number is 2345 6789 0123.",
]

for text in samples:
    safe_text = prepare_for_llm(text)

    print("\nOriginal text:")
    print(text)

    print("Masked text:")
    print(safe_text)

    print("-" * 50)
