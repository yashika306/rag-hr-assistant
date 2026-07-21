"""
pipeline.py
-----------
Orchestrates the full ingestion pipeline (Phase 1 of the RAG project):

    1. Read the PDF and extract text page by page      (pdf_reader.py)
    2. Split the text into overlapping chunks           (chunker.py)
    3. Embed each chunk locally (sentence-transformers)  (embedder.py)
    4. Store the chunks + embeddings in local ChromaDB  (vector_store.py)

100% free — no OpenAI, no Pinecone, no API key needed for this step.

Run it directly with:
    python -m src.pipeline
"""

from src.paths import DEFAULT_PDF_PATH
from src.ingestion.pdf_reader import read_pdf
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import embed_chunks
from src.ingestion.vector_store import store_in_chroma

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


def process_document(pdf_path: str = DEFAULT_PDF_PATH) -> None:
    # Step 1: read the PDF
    print(f"Step 1: Reading PDF from {pdf_path} ...")
    pages = read_pdf(pdf_path)
    print(f"  Extracted {len(pages)} pages from the PDF")

    # Step 2: split into overlapping chunks
    print(f"Step 2: Chunking text (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    print(f"  Created {len(chunks)} chunks")

    # Step 3: embed each chunk locally (free, no API call)
    print("Step 3: Generating embeddings locally (sentence-transformers) ...")
    embeddings = embed_chunks(chunks)
    print(f"  Generated {len(embeddings)} embeddings ({len(embeddings[0])} dimensions each)")

    # Step 4: store in local ChromaDB
    print("Step 4: Storing chunks + embeddings in local ChromaDB ...")
    store_in_chroma(chunks, embeddings)
    print("  Done. Your local vector database (./chroma_db) is now populated.")


if __name__ == "__main__":
    process_document()
