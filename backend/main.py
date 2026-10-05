import logging
import os
import time

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from core.config import settings
from core.llm import usage_var
from governance import audit
from governance.auth import authenticate, create_token, decode_token
from governance.guardrails import RateLimiter, check_input, detect_pii, mask_pii, risk_score
from monitoring.metrics import LAT, REQS, TOKENS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")
app = FastAPI(title="Cortex: Enterprise AI Platform")
bearer = HTTPBearer()
limiter = RateLimiter(settings.RATE_LIMIT_PER_MIN)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)


@app.on_event("startup")
def _startup():
    audit.init_db()


def user(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    try:
        return decode_token(creds.credentials)
    except Exception:
        raise HTTPException(401, "Invalid or expired token")


def admin(u: dict = Depends(user)) -> dict:
    if u["role"] != "admin":
        raise HTTPException(403, "Admin role required")
    return u


class QueryIn(BaseModel):
    question: str
    top_k: int = 5
    department: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/auth/token")
def token(form: OAuth2PasswordRequestForm = Depends()):
    role = authenticate(form.username, form.password)
    if not role:
        raise HTTPException(401, "Bad credentials")
    return {"access_token": create_token(form.username, role), "token_type": "bearer", "role": role}


@app.post("/ingest")
def ingest(file: UploadFile = File(...), department: str = Form("general"), u: dict = Depends(admin)):
    from rag.embeddings import embed
    from rag.ingestion import ingest_file
    from rag.vectorstore import get_store

    path = os.path.join(settings.UPLOAD_DIR, os.path.basename(file.filename))
    with open(path, "wb") as f:
        f.write(file.file.read())
    try:
        chunks = ingest_file(path, department)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not chunks:
        raise HTTPException(400, "No text could be extracted")
    n = get_store().upsert(chunks, embed([c["text"] for c in chunks]))
    audit.log(user=u["sub"], question=f"[ingest] {file.filename}", status="ok")
    return {"file": file.filename, "chunks": n}


@app.post("/query")
def query(body: QueryIn, u: dict = Depends(user)):
    from agents.graph import run_agent

    if not limiter.allow(u["sub"]):
        REQS.labels("rate_limited").inc()
        raise HTTPException(429, "Rate limit exceeded")
    t0 = time.time()
    usage = {"tokens": 0, "calls": 0}
    usage_var.set(usage)

    g = check_input(body.question)
    if g["injection"]:
        REQS.labels("blocked").inc()
        audit.log(user=u["sub"], question=body.question, status="blocked", risk=1.0)
        raise HTTPException(400, "Request blocked: possible prompt injection / jailbreak")

    flt = {"department": {"$eq": body.department}} if body.department else None
    try:
        res = run_agent(g["sanitized"], filter=flt, top_k=body.top_k)
    except Exception as e:
        log.exception("agent failure")
        REQS.labels("error").inc()
        audit.log(user=u["sub"], question=body.question, status="error", answer=str(e)[:500])
        raise HTTPException(500, "Agent error")

    pii_out = detect_pii(res["answer"])
    answer = mask_pii(res["answer"])
    risk = risk_score(injection=False, toxic=g["toxic"], pii_in=g["pii"], pii_out=pii_out,
                      grounded=res["grounded"], n_sources=len(res["sources"]), route=res["route"])
    pending = risk >= settings.RISK_APPROVAL_THRESHOLD
    lat = int((time.time() - t0) * 1000)
    rid = audit.log(user=u["sub"], question=body.question, answer=answer, route=res["route"] or "",
                    risk=risk, status="pending_approval" if pending else "ok", grounded=res["grounded"],
                    n_sources=len(res["sources"]), latency_ms=lat, tokens=usage["tokens"])
    REQS.labels("pending" if pending else "ok").inc()
    LAT.observe(lat / 1000)
    TOKENS.inc(usage["tokens"])
    return {"id": rid, "answer": "Held for human approval (high risk)." if pending else answer,
            "status": "pending_approval" if pending else "ok", "risk": risk, "route": res["route"],
            "grounded": res["grounded"], "sources": res["sources"], "trace": res["trace"],
            "latency_ms": lat, "tokens": usage["tokens"]}


@app.get("/audit")
def get_audit(limit: int = 200, _: dict = Depends(admin)):
    return audit.rows(limit)


@app.get("/approvals")
def approvals(_: dict = Depends(admin)):
    return audit.rows(100, status="pending_approval")


@app.post("/approvals/{row_id}/{decision}")
def decide(row_id: int, decision: str, _: dict = Depends(admin)):
    if decision not in ("approve", "reject"):
        raise HTTPException(400, "decision must be approve|reject")
    r = audit.set_status(row_id, "approved" if decision == "approve" else "rejected")
    if not r:
        raise HTTPException(404, "Not found")
    return r


@app.post("/eval")
def run_evaluation(dataset: list[dict], k: int = 5, _: dict = Depends(admin)):
    from evaluation.runner import run_eval

    return run_eval(dataset, k=k)
