"""
hr_policy_agent/agent.py
-------------------------
Defines the HR Policy Assistant using Google's Agent Development Kit
(ADK). This is the primary, recommended way to run this project — an
agent served via `adk web` (browser UI) or `adk api_server` (REST API),
powered end-to-end by Google's free Gemini API.

The retrieval logic (embed the question, search ChromaDB) is exposed
to the agent as a single tool, `search_hr_policy`. The agent's
instruction forces it to always call that tool before answering, and
to answer only from the retrieved context — this is what keeps the
agent "grounded" in the actual HR policy document instead of making
things up.
"""

import os
import sys

# src/ lives at the project root, one directory up from this agent
# package — make sure that directory is importable regardless of
# where `adk web` / `adk api_server` is launched from.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.retrieval.retriever import retrieve_relevant_chunks  # noqa: E402

from google.adk.agents import Agent

LLM_MODEL = "gemini-3.1-flash-lite"
TOP_K = 3


def search_hr_policy(question: str) -> dict:
    """Searches the company's HR policy document for information relevant
    to the user's question and returns the most relevant excerpts.

    Always call this tool before answering any question about HR policy -
    do not answer from general knowledge, since the answer must come only
    from the actual policy document.

    Args:
        question: The user's question about HR policy (leave, payroll,
                  attendance, reimbursement, code of conduct, etc.).

    Returns:
        A dict with:
          - status: "success" or "error"
          - context: a list of the most relevant policy excerpts (on success)
          - error_message: why the search failed (on error)
    """
    try:
        chunks = retrieve_relevant_chunks(question, top_k=TOP_K)
        return {"status": "success", "context": chunks}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


root_agent = Agent(
    name="hr_policy_agent",
    model=LLM_MODEL,
    description="Answers employee questions using the company's HR policy document.",
    instruction=(
        "You are an HR policy assistant. For every question about HR policy "
        "(leave, payroll, attendance, reimbursement, code of conduct, etc.), "
        "you MUST call the search_hr_policy tool first to retrieve the "
        "relevant excerpts from the actual policy document, then answer "
        "using ONLY that retrieved context. "
        "If the tool's context doesn't contain the answer, tell the user "
        "you don't have that information in the policy document - never "
        "make up an answer from general knowledge."
    ),
    tools=[search_hr_policy],
)
