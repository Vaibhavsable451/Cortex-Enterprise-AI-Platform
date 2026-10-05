"""Retrieval + generation evaluation.

Dataset JSON: [{"question": "...", "relevant_sources": ["handbook.pdf"], "reference": "optional"}]
"""
import json
import statistics
import time

from core.llm import chat_json, usage_var
from evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from rag.retrieval import retrieve


def judge(question: str, answer: str, context: str) -> dict:
    return chat_json([
        {"role": "system", "content": (
            "You are a strict RAG evaluator. Reply JSON with scores in [0,1]: "
            '{"faithfulness": x, "answer_relevance": x, "context_relevance": x}')},
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{context[:3000]}\n\nAnswer:\n{answer}"}])


def run_eval(dataset: list[dict], k: int = 5, with_generation: bool = True) -> dict:
    from agents.graph import run_agent

    rows = []
    for item in dataset:
        t0 = time.time()
        rel = set(item["relevant_sources"])
        docs = retrieve(item["question"], top_k=k)
        ids = [d["metadata"].get("source") for d in docs]
        row = {"question": item["question"], "recall@k": recall_at_k(ids, rel, k),
               "precision@k": precision_at_k(ids, rel, k), "mrr": mrr(ids, rel), "ndcg@k": ndcg_at_k(ids, rel, k)}
        if with_generation:
            usage = {"tokens": 0, "calls": 0}
            usage_var.set(usage)
            res = run_agent(item["question"], top_k=k)
            ctx = "\n".join(s["snippet"] for s in res["sources"])
            row.update(judge(item["question"], res["answer"], ctx))
            row["hallucinated"] = 0 if res["grounded"] else 1
            row["tokens"] = usage["tokens"]
        row["latency_s"] = round(time.time() - t0, 2)
        rows.append(row)
    keys = [k_ for k_ in rows[0] if k_ != "question"] if rows else []
    summary = {k_: round(statistics.mean(float(r[k_]) for r in rows if k_ in r), 3) for k_ in keys}
    return {"summary": summary, "rows": rows}


if __name__ == "__main__":
    import sys

    print(json.dumps(run_eval(json.load(open(sys.argv[1]))), indent=2))
