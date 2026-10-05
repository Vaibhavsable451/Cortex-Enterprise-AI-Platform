"""Agentic + Corrective + Self-RAG as a LangGraph state machine.

START -> router -> (direct | retrieve) -> grade -> (rewrite -> retrieve)* -> generate
      -> verify -> (regenerate)? -> END
"""
from typing import TypedDict

from langgraph.graph import END, StateGraph

from core.llm import chat, chat_json
from rag.retrieval import dedupe, retrieve

MAX_RETRIEVE_RETRIES = 2
MAX_REGENS = 1


class State(TypedDict, total=False):
    question: str
    query: str
    route: str
    docs: list
    answer: str
    grounded: bool
    feedback: str
    retries: int
    regens: int
    trace: list
    filter: dict
    top_k: int


def _t(s, msg):
    return list(s.get("trace", [])) + [msg]


# ---- Supervisor / router (adaptive retrieval) ----
def router(s: State):
    r = chat_json([
        {"role": "system", "content": (
            "Classify the user question for a document-QA system. Reply JSON: "
            '{"route": "direct" | "simple" | "complex"}. '
            "direct = greeting/general chit-chat needing no documents; "
            "simple = answerable from one passage; complex = multi-part/comparative/multi-hop.")},
        {"role": "user", "content": s["question"]}])
    route = r.get("route", "simple")
    route = route if route in ("direct", "simple", "complex") else "simple"
    return {"route": route, "query": s["question"], "retries": 0, "regens": 0,
            "trace": _t(s, f"router -> {route}")}


# ---- Research agent / retriever ----
def retrieve_node(s: State):
    k = s.get("top_k", 5)
    queries = [s["query"]]
    if s["route"] == "complex" and s.get("retries", 0) == 0:
        r = chat_json([
            {"role": "system", "content": 'Split the question into at most 3 standalone search queries. Reply JSON: {"queries": ["..."]}'},
            {"role": "user", "content": s["question"]}])
        queries = [q for q in r.get("queries", []) if isinstance(q, str)][:3] or queries
    docs = []
    for q in queries:
        docs += retrieve(q, top_k=k, filter=s.get("filter"))
    docs = dedupe(docs)[: k * 2]
    return {"docs": docs, "trace": _t(s, f"retrieve {len(queries)} query(ies) -> {len(docs)} chunks")}


# ---- Corrective RAG: grade retrieved docs ----
def grade(s: State):
    docs = s.get("docs", [])
    if not docs:
        return {"trace": _t(s, "grade: nothing retrieved")}
    listing = "\n".join(f"[{i}] {d['text'][:500]}" for i, d in enumerate(docs))
    r = chat_json([
        {"role": "system", "content": 'Return JSON {"relevant": [indices]} listing passages that help answer the question.'},
        {"role": "user", "content": f"Question: {s['query']}\n\nPassages:\n{listing}"}])
    keep = [i for i in r.get("relevant", []) if isinstance(i, int) and 0 <= i < len(docs)]
    kept = [docs[i] for i in keep]
    return {"docs": kept, "trace": _t(s, f"grade: {len(kept)}/{len(docs)} relevant")}


def after_grade(s: State):
    if s.get("docs"):
        return "generate"
    return "rewrite" if s.get("retries", 0) < MAX_RETRIEVE_RETRIES else "generate"


# ---- Query rewrite agent (retrieval feedback loop) ----
def rewrite(s: State):
    q = chat([
        {"role": "system", "content": "Rewrite the question as a better search query (keywords, synonyms, expand acronyms). Return only the query."},
        {"role": "user", "content": s["query"]}], temperature=0.3, max_tokens=80).strip()
    return {"query": q, "retries": s.get("retries", 0) + 1, "trace": _t(s, f"rewrite -> '{q}'")}


# ---- RAG agent ----
def generate(s: State):
    docs = s.get("docs", [])
    if not docs:
        return {"answer": "I couldn't find relevant information in the uploaded documents.",
                "grounded": True, "trace": _t(s, "generate: no context")}
    ctx = "\n\n".join(f"[{i+1}] (source: {d['metadata'].get('source')}, p.{d['metadata'].get('page')})\n{d['text']}"
                      for i, d in enumerate(docs))
    fb = f"\nA previous draft was rejected: {s['feedback']}. Fix this." if s.get("feedback") else ""
    ans = chat([
        {"role": "system", "content": "Answer ONLY from the context. Cite sources like [1]. If the context is insufficient, say so." + fb},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {s['question']}"}], max_tokens=800)
    return {"answer": ans, "trace": _t(s, "generate")}


def generate_direct(s: State):
    ans = chat([{"role": "system", "content": "You are a helpful enterprise assistant. Be brief."},
                {"role": "user", "content": s["question"]}], max_tokens=300)
    return {"answer": ans, "grounded": True, "docs": [], "trace": _t(s, "direct answer")}


# ---- Verification agent (Self-RAG groundedness) ----
def verify(s: State):
    if not s.get("docs"):
        return {"grounded": True, "trace": _t(s, "verify: skipped")}
    ctx = "\n".join(d["text"][:600] for d in s["docs"])
    r = chat_json([
        {"role": "system", "content": 'Check whether every claim in the answer is supported by the context. Reply JSON {"grounded": true|false, "issues": "..."}'},
        {"role": "user", "content": f"Context:\n{ctx}\n\nAnswer:\n{s['answer']}"}])
    g = bool(r.get("grounded", True))
    return {"grounded": g, "feedback": r.get("issues", ""), "trace": _t(s, f"verify: grounded={g}")}


def after_verify(s: State):
    if not s.get("grounded", True) and s.get("regens", 0) < MAX_REGENS:
        return "regen"
    return END


def regen(s: State):
    return {"regens": s.get("regens", 0) + 1, "trace": _t(s, "regenerate")}


def _build():
    g = StateGraph(State)
    for n, f in [("router", router), ("retrieve", retrieve_node), ("grade", grade), ("rewrite", rewrite),
                 ("generate", generate), ("generate_direct", generate_direct), ("verify", verify), ("regen", regen)]:
        g.add_node(n, f)
    g.set_entry_point("router")
    g.add_conditional_edges("router", lambda s: "generate_direct" if s["route"] == "direct" else "retrieve",
                            {"generate_direct": "generate_direct", "retrieve": "retrieve"})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", after_verify, {"regen": "regen", END: END})
    g.add_edge("regen", "generate")
    g.add_edge("generate_direct", END)
    return g.compile()


_graph = None


def run_agent(question: str, filter: dict | None = None, top_k: int = 5) -> dict:
    global _graph
    if _graph is None:
        _graph = _build()
    out = _graph.invoke({"question": question, "filter": filter, "top_k": top_k, "trace": []})
    return {"answer": out.get("answer", ""), "route": out.get("route"), "grounded": out.get("grounded", True),
            "sources": [{"source": d["metadata"].get("source"), "page": d["metadata"].get("page"),
                         "score": round(d["score"], 3), "snippet": d["text"][:200]} for d in out.get("docs", [])],
            "trace": out.get("trace", [])}
