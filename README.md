# 📄 PaperLens — Chat with your PDFs

> Hybrid-retrieval RAG over your PDFs with cross-encoder reranking and grounded citations.

**Live Demo:** https://vishwas-paperlens-chat-with-pdf.streamlit.app

Most "chat with PDF" tutorials use plain vector search and call it a day. PaperLens uses the same retrieval techniques production search systems do — and shows its work.

## 🚀 What makes it different

- **Hybrid retrieval** — Combines semantic (FAISS) and keyword (BM25) search, fused via Reciprocal Rank Fusion. Catches both meaning and exact technical terms.
- **Cross-encoder reranking** — Top-20 retrieved chunks are rescored by a cross-encoder before being sent to the LLM. Production search systems work this way; tutorials don't.
- **Grounded citations** — Every claim is cited with `[1] [2]` markers pointing to specific chunks. Sources panel shows the exact excerpt, source PDF, page number, and relevance score.
- **Auto-generated summaries** — Each uploaded PDF is summarized on ingest, so you know what you have before you ask.
- **Suggested follow-ups** — After each answer, three contextual follow-up questions appear as clickable buttons.
- **Token tracking** — Live count of input and output tokens for the session.
- **Markdown export** — Download the entire conversation with citations.

## 🛠️ Tech Stack

- **Streamlit** — Web UI
- **LangChain 0.3** — Pipeline orchestration
- **Gemini 2.5 Flash** — LLM
- **gemini-embedding-001** — Embeddings
- **FAISS** — Vector store
- **BM25** — Keyword retrieval
- **cross-encoder/ms-marco-MiniLM-L-6-v2** — Reranker
- **pypdf** — PDF parsing

## 📐 Architecture

```
![PaperLens Architecture](https://github.com/VishwasPrabhakara/Chat_with_PDF/raw/main/architecture.svg)

```

## 🏃 Run locally

```bash
git clone https://github.com/VishwasPrabhakara/Chat_with_PDF.git
cd Chat_with_PDF
python -m venv venv
.\venv\Scripts\Activate.ps1     # Windows
# source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt

echo "GOOGLE_API_KEY=your_key" > .env

streamlit run app.py
```

Get a free Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

## 📁 Project Structure

```
.
├── app.py             # Streamlit UI
├── pipeline.py        # RAG pipeline (load, chunk, retrieve, rerank, generate)
├── prompts.py         # Prompt templates
├── test_pipeline.py   # Smoke test for the pipeline
├── requirements.txt
└── .env.example
```

## 📝 Built By

**Vishwas Prabhakara** — ML Engineer @ IISc
[GitHub](https://github.com/VishwasPrabhakara) · [LinkedIn](https://www.linkedin.com/in/vishwas-prabhakara-2050821b6/)

## 📄 License

MIT
