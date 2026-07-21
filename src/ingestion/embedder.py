"""
embedder.py
-----------
Ingestion step 3: converts text chunks into vector embeddings using a
FREE, LOCAL embedding model (sentence-transformers, all-MiniLM-L6-v2).

No API key, no signup, no cost — the model downloads once (~90MB) the
first time you run this, then everything runs on your own machine,
fully offline.
"""

from sentence_transformers import SentenceTransformer

# Loaded once at import time and reused for every call to embed_chunks().
_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """
    Embed a list of text chunks into vector representations.

    Args:
        chunks: List of text chunks (as returned by chunk_pages).

    Returns:
        A list of embedding vectors (each a list of 384 floats),
        one per chunk, in the same order.
    """
    vectors = _model.encode(chunks, show_progress_bar=False)
    return vectors.tolist()


if __name__ == "__main__":
    from src.paths import DEFAULT_PDF_PATH
    from src.ingestion.pdf_reader import read_pdf
    from src.ingestion.chunker import chunk_pages

    pdf_pages = read_pdf(DEFAULT_PDF_PATH)
    text_chunks = chunk_pages(pdf_pages, chunk_size=900, overlap=150)
    embedded_chunks = embed_chunks(text_chunks)

    print(f"Total embedded chunks: {len(embedded_chunks)}")
    print(f"Embedding dimensions: {len(embedded_chunks[0])}")
