"""Full pipeline test: load -> chunk -> retrieve -> rerank -> generate."""
from dotenv import load_dotenv
import os

from pipeline import (
    load_pdf,
    chunk_documents,
    HybridRetriever,
    Reranker,
    Generator,
)

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")

print("Loading PDF...")
docs = load_pdf("test.pdf", "test.pdf")
chunks = chunk_documents(docs)
print(f"  {len(docs)} pages -> {len(chunks)} chunks")

print("Building components...")
retriever = HybridRetriever(chunks, key)
reranker = Reranker()
generator = Generator(key)
print("Ready.\n")

query = "What is this person's background and what kind of ML work do they do?"
print(f"Q: {query}\n")

retrieved = retriever.retrieve(query)
reranked = reranker.rerank(query, retrieved)
answer = generator.answer(query, reranked)

print("---ANSWER---")
print(answer.text)
print()

print("---CITATIONS---")
for i, c in enumerate(answer.citations, 1):
    print(f"  [{i}] {c.source} p.{c.page} (score {c.score:+.2f})")
    print(f"      \"{c.excerpt}\"")
print()

print("---FOLLOW-UPS---")
followups = generator.suggest_followups(query, answer.text)
for q in followups:
    print(f"  • {q}")

print()
print(f"Tokens: in={answer.tokens_in}, out={answer.tokens_out}")