"""
retriever.py
------------
Given a user's question, embeds it with the same local model used
during ingestion, then runs a similarity search against the local
ChromaDB collection to pull back the most relevant chunks of the
original document. This is the module the ADK agent's tool (and the
optional terminal chatbot) both call.
"""

import chromadb

from src.paths import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME
from src.ingestion.embedder import embed_chunks

_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)


def retrieve_relevant_chunks(question: str, top_k: int = 3) -> list[str]:
    """
    Embed the user's question and return the top_k most similar
    chunks stored in ChromaDB.

    Args:
        question: The user's natural-language question.
        top_k: How many chunks to retrieve.

    Returns:
        A list of chunk text strings, most relevant first.
    """
    collection = _client.get_collection(name=CHROMA_COLLECTION_NAME)

    # Embed the question the exact same way the document chunks were
    # embedded, so they live in the same vector space and are comparable.
    question_embedding = embed_chunks([question])[0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
    )

    return results["documents"][0]


if __name__ == "__main__":
    test_question = "How many days of annual leave do employees get?"
    top_chunks = retrieve_relevant_chunks(test_question, top_k=2)
    print(f"Question: {test_question}\n")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"--- Retrieved chunk {i} ---")
        print(chunk)
        print()
