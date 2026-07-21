"""
vector_store.py
----------------
Ingestion step 4: stores text chunks and their embeddings in a local
ChromaDB database (a folder on your own disk, at ./chroma_db).

No account, no API key, no cost — this is a completely free, fully
local replacement for a hosted vector database like Pinecone.
"""

import chromadb

from src.paths import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME

_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)


def store_in_chroma(chunks: list[str], embeddings: list[list[float]]) -> None:
    """
    Add each (chunk, embedding) pair into a local Chroma collection.
    The chunk's own text is stored alongside its vector, so a later
    similarity search can return the original text.

    Args:
        chunks: List of text chunks.
        embeddings: List of embedding vectors, same order/length as chunks.
    """
    collection = _client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    ids = [f"chunk-{i}" for i in range(len(chunks))]
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
    )


if __name__ == "__main__":
    from src.paths import DEFAULT_PDF_PATH
    from src.ingestion.pdf_reader import read_pdf
    from src.ingestion.chunker import chunk_pages
    from src.ingestion.embedder import embed_chunks

    pdf_pages = read_pdf(DEFAULT_PDF_PATH)
    text_chunks = chunk_pages(pdf_pages, chunk_size=900, overlap=150)
    embedded_chunks = embed_chunks(text_chunks)
    store_in_chroma(text_chunks, embedded_chunks)

    print(f"Stored {len(text_chunks)} chunks in local ChromaDB collection '{CHROMA_COLLECTION_NAME}'")
