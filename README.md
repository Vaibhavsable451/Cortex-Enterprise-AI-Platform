# Cortex: Enterprise AI Platform

**Cortex** is an enterprise agentic RAG platform with governance, evaluation and MLOps.

Upload documents → clean → chunk → embed → **Pinecone** → agentic retrieval → **Groq** LLM → verify → evaluate → govern → deploy.

| Layer | Implementation |
|---|---|
| LLM | Groq (`llama-3.3-70b-versatile`) via `core/llm.py` |
| Vector DB | Pinecone serverless (`rag/vectorstore.py`); FAISS fallback with `VECTOR_BACKEND=faiss` |
| Embeddings | `all-MiniLM-L6-v2` (384-d, local) — Groq has no embeddings API |
| Agents | LangGraph: router → retrieve → grade (Corrective RAG) → rewrite loop → generate → verify (Self-RAG) → regenerate |
| API | FastAPI + JWT + RBAC + rate limiting + audit log + human approval |
| UI | Streamlit (chat, ingestion, dashboard, approvals, evaluation) |
| Eval | Recall@K, Precision@K, MRR, NDCG, LLM-judge faithfulness / relevance, latency, tokens |
| Governance | prompt-injection, PII masking, toxicity, risk score, output guardrail |
| MLOps | MLflow (`ml/pipeline.py`), Docker Compose, K8s manifests, GitHub Actions → ECR → EC2 |
| DL from scratch | `deep_learning/` backprop, BatchNorm/LayerNorm, attention, transformers, LoRA/QLoRA |

## Quick start (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-frontend.txt
cp .env.example .env        # add GROQ_API_KEY and PINECONE_API_KEY
uvicorn backend.main:app --reload            # terminal 1  (http://localhost:8000/docs)
streamlit run frontend/app.py                # terminal 2  (http://localhost:8501)
```
Login: `admin / admin123` (can ingest, see dashboard, approve) or `analyst / analyst123` (chat only). **Change these in `.env`.**
No Pinecone key yet? Set `VECTOR_BACKEND=faiss` (in-memory, resets on restart).

## Docker / Kubernetes / EC2
```bash
docker compose up -d --build          # UI at http://localhost, API at /api, MLflow :5000
kubectl apply -f kubernetes/platform.yaml   # replace ECR_REGISTRY + secrets first
bash ec2_setup.sh                     # on the EC2 host, then docker compose up -d --build
```
GitHub secrets for CI/CD: `AWS_ROLE_ARN`, `EC2_HOST`, `EC2_SSH_KEY`. Create ECR repos `cortex-backend`, `cortex-frontend`.

## Learning modules
```bash
pip install -r requirements-dl.txt
python -m deep_learning.backprop        # manual backprop, verified vs autograd
python -m deep_learning.normalization
python -m deep_learning.transformer     # tiny GPT
python -m deep_learning.lora_finetune --model Qwen/Qwen2.5-0.5B --samples 500
python -m ml.pipeline                   # sklearn + MLflow
```
Not included (build in later phases): Graph RAG, multi-vector retrieval, SQL agent, FSDP/DeepSpeed configs, DVC, Prometheus/Grafana dashboards, drift detection.

## Tests
`pytest -q tests`
