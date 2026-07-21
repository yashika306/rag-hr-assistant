"""
chunker.py
----------
Ingestion step 2: splits the full text extracted from a PDF into
overlapping chunks, so each chunk is small enough to embed and later
fits cleanly into an LLM's context window without losing meaning at
the boundaries.
"""


def chunk_pages(pages: list[str], chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """
    Join all page text into one string, then split it into
    overlapping chunks.

    Args:
        pages: List of page text strings (as returned by read_pdf).
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters each chunk repeats from the end
                 of the previous chunk, so context isn't lost at a
                 chunk boundary (e.g. a sentence cut in half).

    Returns:
        A list of text chunks.
    """
    full_text = "".join(pages)
    text_length = len(full_text)

    chunks = []
    start = 0
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = full_text[start:end]
        chunks.append(chunk)

        if end == text_length:
            break

        # Move the start back by `overlap` characters so the next
        # chunk repeats a bit of context from the end of this one.
        start = end - overlap

    return chunks


if __name__ == "__main__":
    from src.paths import DEFAULT_PDF_PATH
    from src.ingestion.pdf_reader import read_pdf

    pdf_pages = read_pdf(DEFAULT_PDF_PATH)
    result_chunks = chunk_pages(pdf_pages, chunk_size=900, overlap=150)
    print(f"Total chunks created: {len(result_chunks)}")
    if result_chunks:
        print("--- Chunk 1 ---")
        print(result_chunks[0])
