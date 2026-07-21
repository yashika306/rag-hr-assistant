"""
terminal_chat.py
-----------------
Optional, minimal way to chat with the HR policy document from a plain
terminal `input()` loop — no ADK, no browser, no server. Useful for
quickly testing retrieval + generation together before running the
full ADK agent.

For each question the user asks:
    1. Retrieve the most relevant chunks from ChromaDB (src/retrieval/retriever.py)
    2. Build a prompt containing those chunks + the question
    3. Send it to the free Gemini API (gemini-2.5-flash) to generate a grounded answer
    4. Print the answer

100% Google stack, 100% free tier - same GOOGLE_API_KEY used by the
ADK agent in hr_policy_agent/.env.

Run it directly with:
    python -m scripts.terminal_chat
"""

import os

from dotenv import load_dotenv
from google import genai

from src.retrieval.retriever import retrieve_relevant_chunks

load_dotenv()

_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
LLM_MODEL = "gemini-3.1-flash-lite"
TOP_K = 3

SYSTEM_PROMPT = (
    "You are an HR policy assistant. Answer the user's question using ONLY "
    "the context provided below, which comes from the company's HR policy "
    "document. If the answer isn't in the context, say you don't have that "
    "information in the policy document - do not make anything up."
)


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    return (
        f"Context from the HR policy document:\n"
        f"---\n{context}\n---\n\n"
        f"Question: {question}"
    )


def answer_question(question: str) -> str:
    """
    Run the full retrieve -> prompt -> generate flow for one question.
    """
    context_chunks = retrieve_relevant_chunks(question, top_k=TOP_K)
    user_prompt = build_prompt(question, context_chunks)

    response = _client.models.generate_content(
        model=LLM_MODEL,
        contents=user_prompt,
        config={"system_instruction": SYSTEM_PROMPT},
    )
    return response.text if response.text else "No answer generated."


def chat_loop():
    print("HR Policy Assistant (type 'exit' to quit)\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not question:
            continue

        answer = answer_question(question)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    chat_loop()
