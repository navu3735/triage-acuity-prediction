"""LangChain-based RAG pipelines for patient summaries and workflow recommendations.

Keeps imports lean (no langchain_community document loaders / text_splitters package
root) so TensorFlow/spaCy side-effects do not break local startup. Retrieval uses
FAISS when available, otherwise an in-memory cosine index.
"""

from __future__ import annotations

import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"


class HashEmbedding(Embeddings):
    """Lightweight local embeddings so RAG works without downloading models."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vec
        for tok in tokens:
            h = hash(tok)
            idx = h % self.dim
            sign = 1.0 if (h & 1) == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class SimpleVectorStore:
    """Minimal cosine similarity store used when FAISS is unavailable."""

    def __init__(self, docs: Sequence[Document], embeddings: Embeddings):
        self.docs = list(docs)
        self.embeddings = embeddings
        self.matrix = embeddings.embed_documents([d.page_content for d in self.docs])

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        q = self.embeddings.embed_query(query)
        scored = []
        for doc, vec in zip(self.docs, self.matrix):
            score = sum(a * b for a, b in zip(q, vec))
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:k]]


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]


def _load_documents() -> List[Document]:
    docs: List[Document] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        for chunk in _chunk_text(content):
            docs.append(Document(page_content=chunk, metadata={"source": str(path)}))
    return docs


@lru_cache(maxsize=1)
def get_vectorstore():
    splits = _load_documents()
    embeddings = HashEmbedding()
    try:
        from langchain_community.vectorstores import FAISS

        return FAISS.from_documents(splits, embeddings)
    except Exception:
        return SimpleVectorStore(splits, embeddings)


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content.strip() for d in docs)


def _openai_available() -> bool:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return False
    # Ignore common placeholder values so local demos stay offline-safe.
    lowered = key.lower()
    if "your" in lowered or "replace" in lowered or key.endswith("here"):
        return False
    return key.startswith("sk-")


def _retrieve(query: str, k: int = 4) -> List[Document]:
    store = get_vectorstore()
    return store.similarity_search(query, k=k)


def _synthesize_local(context: str, question: str, mode: str) -> str:
    bullets = [ln.strip("- ").strip() for ln in context.splitlines() if ln.strip()]
    bullets = [b for b in bullets if len(b) > 20][:6]
    evidence = "\n".join(f"- {b}" for b in bullets) or "- Follow standard ESI triage protocols."

    if mode == "summary":
        return (
            "AI-assisted patient summary\n\n"
            f"Clinical context: {question.strip()}\n\n"
            f"Retrieved protocol guidance:\n{evidence}\n\n"
            "Narrative: Based on the intake vitals/complaint and matching ED protocols, "
            "prioritize acuity-aligned monitoring, complete indicated workup resources, "
            "and reassess frequently until disposition is clear."
        )
    return (
        "Workflow recommendations\n\n"
        f"Query: {question.strip()}\n\n"
        f"Recommended next steps:\n{evidence}\n\n"
        "Operational note: Route lower-acuity cases to fast-track when census is high; "
        "escalate ESI 1–2 immediately to monitored care."
    )


def _llm_generate(prompt_text: str) -> str:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2)
    return llm.invoke(prompt_text).content


def run_patient_summary_chain(
    patient_blob: str,
    acuity: int | None = None,
    include_recommendations: bool = True,
) -> Tuple[str, str, List[str], str]:
    query_bits = [patient_blob]
    if acuity:
        query_bits.append(f"ESI level {acuity} workflow")
    query = " | ".join(query_bits)

    docs = _retrieve(query, k=4)
    context = _format_docs(docs)
    sources = [d.metadata.get("source", "knowledge") for d in docs]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a clinical operations assistant. Use the retrieved ED protocols "
                "to draft a concise patient summary and actionable workflow recommendations. "
                "Do not invent lab results. Be clear this is decision support, not a diagnosis.",
            ),
            (
                "human",
                "Patient intake:\n{patient}\n\nRetrieved protocols:\n{context}\n\n"
                "Write: (1) a short patient summary, (2) workflow recommendations.",
            ),
        ]
    )

    if _openai_available():
        messages = prompt.format_messages(patient=patient_blob, context=context)
        text = _llm_generate("\n".join(m.content for m in messages))
        model_name = "langchain-rag+openai"
    else:
        chain = (
            {
                "patient": RunnablePassthrough(),
                "context": RunnableLambda(lambda _: context),
            }
            | RunnableLambda(
                lambda x: _synthesize_local(x["context"], x["patient"], "summary")
            )
        )
        text = chain.invoke(patient_blob)
        model_name = "langchain-rag+local"

    if "Recommended" in text or "recommendations" in text.lower():
        parts = re.split(r"(?i)workflow recommendations|recommended next steps", text, maxsplit=1)
        summary = parts[0].strip()
        recs = parts[1].strip() if len(parts) > 1 else ""
    else:
        summary = text.strip()
        recs = ""
        if include_recommendations:
            recs = _synthesize_local(context, patient_blob, "workflow")

    return summary, recs, sources, model_name


def run_workflow_recommendation_chain(
    chief_complaint: str,
    acuity: int | None = None,
    context: str | None = None,
) -> Tuple[str, List[str], str]:
    query = chief_complaint
    if acuity:
        query = f"ESI {acuity}: {chief_complaint}"
    if context:
        query = f"{query}. {context}"

    docs = _retrieve(query, k=4)
    retrieved = _format_docs(docs)
    sources = sorted({Path(d.metadata.get("source", "knowledge")).name for d in docs})

    prompt = ChatPromptTemplate.from_template(
        "Chief complaint: {complaint}\nAcuity: {acuity}\nExtra context: {extra}\n\n"
        "Protocols:\n{protocols}\n\nProvide numbered workflow recommendations for ED staff."
    )

    if _openai_available():
        msg = prompt.format(
            complaint=chief_complaint,
            acuity=acuity or "unknown",
            extra=context or "n/a",
            protocols=retrieved,
        )
        text = _llm_generate(msg)
        model_name = "langchain-rag+openai"
    else:
        text = _synthesize_local(retrieved, query, "workflow")
        model_name = "langchain-rag+local"

    return text, sources, model_name
