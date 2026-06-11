"""Extract text from a PDF file."""
import sys

try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf", "--quiet"])
        from pypdf import PdfReader

path = r"C:\Users\Sasha\Desktop\Nakonechny-Shevchuk.pdf"
out_path = r"temp_scripts\extracted.txt"

reader = PdfReader(path)
with open(out_path, "w", encoding="utf-8") as fout:
    fout.write(f"PAGES_TOTAL: {len(reader.pages)}\n")
    for i, page in enumerate(reader.pages):
        fout.write(f"\n===== PAGE {i+1} =====\n")
        try:
            fout.write(page.extract_text() or "")
        except Exception as exc:
            fout.write(f"[error extracting page: {exc}]")
print(f"Wrote: {out_path}")
