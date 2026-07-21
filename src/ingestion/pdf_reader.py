"""
pdf_reader.py
-------------
Ingestion step 1: reads a PDF file and extracts the text content of
each page. Pure text extraction only — no chunking, no embedding.
"""

from pypdf import PdfReader


def read_pdf(path: str) -> list[str]:
    """
    Read a PDF file and return a list where each element is the
    extracted text of one page.

    Args:
        path: Path to the PDF file.

    Returns:
        A list of strings, one per page, in page order.
    """
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        pages.append(text or "")
    return pages


if __name__ == "__main__":
    import sys
    from src.paths import DEFAULT_PDF_PATH

    test_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF_PATH
    result_pages = read_pdf(test_path)
    print(f"Extracted {len(result_pages)} pages from the PDF")
    if result_pages:
        print("--- Page 1 content ---")
        print(result_pages[0][:500])
