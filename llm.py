"""
LLM interaction layer for DocMind, using Groq's free-tier API.

Feature 4: Conversation memory - follow-up questions ("what about the
second point?") get rewritten into standalone queries using recent chat
history before retrieval runs, so retrieval doesn't just search for
"the second point" with no context.

Feature 2: Citations - the answer prompt requires the model to reference
which source chunk each claim comes from, using [1], [2] style markers
that the UI maps back to actual filenames/pages.
"""

from groq import Groq

import config

client = Groq(api_key=config.GROQ_API_KEY) if config.GROQ_API_KEY else None


def _check_client():
    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
            "free key from https://console.groq.com/keys"
        )


FOLLOWUP_INDICATORS = {
    "it", "its", "that", "this", "these", "those", "them", "they",
    "each", "former", "latter", "same", "above", "previous", "again",
    "too", "also", "one", "ones",
}

# A question starting with one of these is almost always a complete,
# standalone question on its own - even if short - so word-count alone
# shouldn't classify it as a follow-up.
QUESTION_STARTERS = {
    "what", "why", "how", "where", "who", "which", "when",
    "can", "does", "do", "is", "are", "will", "should",
}


def _looks_like_followup(question: str) -> bool:
    """
    Feature 4 (rule-based): a question is treated as a follow-up if it
    contains a word that only makes sense with prior context (pronouns
    like "it"/"each"/"that"). Failing that, a short question (<=4 words)
    that does NOT start with a normal question word ("what", "how", ...)
    is also treated as a follow-up, since that pattern - e.g. "sorting
    also", "and the second one" - is typical of a quick, fragment-style
    follow-up rather than a genuine standalone question.
    """
    words = question.lower().replace("?", "").replace(",", "").split()
    if not words:
        return False

    if any(w in FOLLOWUP_INDICATORS for w in words):
        return True

    if words[0] in QUESTION_STARTERS:
        return False

    return len(words) <= 4


def reformulate_query(current_question: str, history: list) -> str:
    """
    Feature 4: Conversation memory via a deterministic rule rather than an
    LLM call. An earlier version asked the LLM to rewrite follow-up
    questions, but small/fast models were unreliable at this - they'd
    sometimes invent a connection to the wrong prior topic, or fail to
    recognize a genuine follow-up. Instead: if the question looks like a
    follow-up (see _looks_like_followup), we simply prepend the last user
    question so the retrieval step has both pieces of context to search
    with. This is less "clever" but far more predictable.
    """
    if not history:
        return current_question

    if not _looks_like_followup(current_question):
        return current_question

    last_user_turn = next(
        (t["content"] for t in reversed(history) if t["role"] == "user"), None
    )
    if not last_user_turn:
        return current_question

    return f"{last_user_turn} {current_question}"


def generate_answer(question: str, chunks: list, history: list = None):
    """
    Generates an answer grounded in the retrieved chunks, with inline
    citation markers like [1], [2] that map to chunks (and therefore to
    source file + page via metadata).
    """
    _check_client()

    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        src = chunk["metadata"].get("source", "unknown")
        page = chunk["metadata"].get("page", "?")
        context_blocks.append(f"[{i}] (Source: {src}, Page: {page})\n{chunk['text']}")

    context_text = "\n\n".join(context_blocks)

    history_text = ""
    if history:
        recent = history[-config.MAX_HISTORY_TURNS:]
        history_text = "\n".join(f"{t['role']}: {t['content']}" for t in recent)

    system_prompt = """You are a precise document Q&A assistant. Answer ONLY using
the provided context chunks, and answer ONLY the current question being asked -
do not answer a different or previous question, even if earlier conversation
turns discussed another topic. For every factual claim, cite the source using
its bracket number, e.g. [1] or [2]. If the context doesn't fully answer the
CURRENT question, say so explicitly rather than guessing or falling back to
an earlier topic. Do not use outside knowledge."""

    user_prompt = f"""Conversation so far:
{history_text}

Context:
{context_text}

Question: {question}

Answer the question using only the context above, with [n] citations."""


    print("GROQ MODEL:", config.GROQ_MODEL)
    print("GROQ KEY EXISTS:", bool(config.GROQ_API_KEY))

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()
