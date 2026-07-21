
"""
embedder.py
-----------
Converts text chunks into vector embeddings using fastembed
(ONNX Runtime, CPU-only, no PyTorch) - much lighter than
sentence-transformers, same all-MiniLM-L6-v2 model and same
384-dim vectors, ideal for memory-constrained hosting.
"""


from fastembed import TextEmbedding

_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """
    Embed a list of text chunks into vector representations.

    Args:
        chunks: List of text chunks (as returned by chunk_pages).

    Returns:
        A list of embedding vectors (each a list of 384 floats),
        one per chunk, in the same order.
    """
    embeddings = list(_model.embed(chunks))
    return [vec.tolist() for vec in embeddings]


if __name__ == "__main__":
    from src.paths import DEFAULT_PDF_PATH
    from src.ingestion.pdf_reader import read_pdf
    from src.ingestion.chunker import chunk_pages

    pdf_pages = read_pdf(DEFAULT_PDF_PATH)
    text_chunks = chunk_pages(pdf_pages, chunk_size=900, overlap=150)
    embedded_chunks = embed_chunks(text_chunks)

    print(f"Total embedded chunks: {len(embedded_chunks)}")
    print(f"Embedding dimensions: {len(embedded_chunks[0])}")
