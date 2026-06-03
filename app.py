"""
PaperLens — Streamlit UI.

Hybrid-retrieval RAG over your PDFs, with cross-encoder reranking,
grounded citations, auto-summaries, and suggested follow-ups.
"""
from __future__ import annotations

import io
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from pipeline import Answer, PaperLens

load_dotenv()

st.set_page_config(
    page_title="PaperLens — Chat with your PDFs",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------

if "engine" not in st.session_state:
    st.session_state.engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[{role, content, answer?}]
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def _api_key() -> str | None:
    """Read API key from Streamlit secrets if available, else fall back to .env."""
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except (FileNotFoundError, KeyError):
        return os.getenv("GOOGLE_API_KEY")


def _history_text(limit: int = 6) -> str:
    msgs = st.session_state.messages[-limit:]
    lines = []
    for m in msgs:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def _markdown_export() -> str:
    lines = [
        "# PaperLens conversation",
        f"_Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        f"**Documents:** {', '.join(st.session_state.ingested_files) or '(none)'}",
        "",
        "---",
        "",
    ]
    for m in st.session_state.messages:
        if m["role"] == "user":
            lines.append(f"### 🧑 Question\n{m['content']}\n")
        else:
            lines.append(f"### 🤖 Answer\n{m['content']}\n")
            ans: Answer | None = m.get("answer")
            if ans and ans.citations:
                lines.append("**Citations**")
                for i, c in enumerate(ans.citations, 1):
                    lines.append(f"- [{i}] {c.source} · p.{c.page} — \"{c.excerpt}\"")
                lines.append("")
    return "\n".join(lines)


def _render_answer(answer: Answer, key_prefix: str) -> None:
    """Render the answer body, citations, and follow-up buttons."""
    st.markdown(answer.text)

    if answer.citations:
        with st.expander(f"🔎 {len(answer.citations)} sources"):
            for i, c in enumerate(answer.citations, 1):
                st.markdown(
                    f"**[{i}] {c.source} — page {c.page}** "
                    f"<span style='color:#94a3b8;font-size:0.85em'>"
                    f"score {c.score:+.2f}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"> {c.excerpt}")

    if answer.followups:
        cols = st.columns(len(answer.followups))
        for idx, (col, q) in enumerate(zip(cols, answer.followups)):
            if col.button(q, key=f"{key_prefix}_{idx}"):
                st.session_state.pending_question = q
                st.rerun()


# ---------------------------------------------------------------------------
# Sidebar — upload, summaries, stats, export
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📄 PaperLens")
    st.caption("Hybrid retrieval + cross-encoder reranking + grounded citations.")
    st.divider()

    api_key = _api_key()
    if not api_key:
        st.error(
            "GOOGLE_API_KEY not set. Add it to `.env` (local) or "
            "`.streamlit/secrets.toml` (deployed)."
        )

    pdf_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button(
        "Process",
        type="primary",
        use_container_width=True,
        disabled=not pdf_files or not api_key,
    ):
        with st.spinner("Reading, chunking, embedding, summarizing…"):
            try:
                engine = PaperLens(api_key=api_key)
                engine.ingest(
                    (f.name, io.BytesIO(f.getvalue())) for f in pdf_files
                )
                st.session_state.engine = engine
                st.session_state.ingested_files = [f.name for f in pdf_files]
                st.session_state.messages = []
                st.success(f"Indexed {len(pdf_files)} document(s).")
            except Exception as e:
                st.error(f"Failed: {e}")

    if st.session_state.engine:
        st.divider()
        st.markdown("**📚 Document summaries**")
        for name, summary in st.session_state.engine.summaries.items():
            with st.expander(name, expanded=False):
                st.write(summary)

        st.divider()
        eng = st.session_state.engine
        st.markdown("**📊 Session stats**")
        cols = st.columns(2)
        cols[0].metric("Tokens in", f"{eng.total_tokens_in:,}")
        cols[1].metric("Tokens out", f"{eng.total_tokens_out:,}")

        st.divider()
        st.download_button(
            "⬇️ Export chat (Markdown)",
            data=_markdown_export(),
            file_name="paperlens_conversation.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=not st.session_state.messages,
        )
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.markdown("## 📄 PaperLens")
st.caption(
    "Ask grounded, cited questions about your PDFs. "
    "Hybrid retrieval (semantic + BM25) → cross-encoder reranking → Gemini."
)

if st.session_state.engine is None:
    st.info(
        "👈 Upload one or more PDFs in the sidebar and click **Process** to begin."
    )
    st.stop()

# Render past messages
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            ans: Answer | None = msg.get("answer")
            if ans:
                _render_answer(ans, key_prefix=f"past_{i}")
            else:
                st.markdown(msg["content"])

# Handle queued follow-up click OR new chat input
prompt = st.session_state.pending_question
st.session_state.pending_question = None
if not prompt:
    prompt = st.chat_input("Ask a question about your documents…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving, reranking, generating…"):
            try:
                answer = st.session_state.engine.ask(
                    prompt, history=_history_text(limit=6)
                )
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        _render_answer(answer, key_prefix=f"new_{len(st.session_state.messages)}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer.text, "answer": answer}
    )