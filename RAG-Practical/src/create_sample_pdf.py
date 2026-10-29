from pathlib import Path
import pymupdf

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "documents" / "bank_policy.pdf"

content = """
BANK POLICY DOCUMENT

1. Transaction Alerts
Customers may receive an alert when a transaction requires
additional security verification.

2. Transaction Review
A transaction alert does not automatically mean that fraud
has occurred. The transaction may require further review.

3. Customer Support
Customers should contact the bank through its official
support channel if they do not recognize a transaction.

4. Account Security
Customers must never share their PIN, password, or OTP
with anyone, including people claiming to represent the bank.
"""

doc = pymupdf.open()
page = doc.new_page()

page.insert_textbox(
    pymupdf.Rect(50, 50, 550, 750),
    content,
    fontsize=12,
    fontname="helv",
)

doc.save(PDF_PATH)
doc.close()

print(f"Sample PDF created: {PDF_PATH}")