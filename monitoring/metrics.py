from prometheus_client import Counter, Histogram

REQS = Counter("rag_requests_total", "Queries", ["status"])
LAT = Histogram("rag_latency_seconds", "Query latency")
TOKENS = Counter("rag_tokens_total", "LLM tokens used")
