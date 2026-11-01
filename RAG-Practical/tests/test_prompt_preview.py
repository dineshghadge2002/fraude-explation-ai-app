
from src.prompt_builder import build_prompt

question = "My account number is 123456789012. How can I protect my account?"

context_chunks = [
    "Never share your OTP, PIN, or password with anyone.",
    "Contact your bank immediately if you notice an unauthorized transaction."
]

prompt = build_prompt(question, context_chunks)

print("\n========== GENERATED PROMPT ==========\n")
print(prompt)
print("\n======================================\n")