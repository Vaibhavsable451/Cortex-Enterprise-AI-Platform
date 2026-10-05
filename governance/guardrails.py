"""Prompt-injection / jailbreak / toxicity / PII detection, risk scoring, rate limiting."""
import re
import threading
import time
from collections import defaultdict, deque

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the )?(system|previous) (prompt|instructions)",
    r"reveal (your|the) (system )?prompt",
    r"you are now (dan|in developer mode)",
    r"act as .* without (any )?restrictions",
    r"jailbreak", r"do anything now", r"bypass (your )?(safety|filters|guardrails)",
]
TOXIC_WORDS = {"idiot", "stupid", "kill yourself", "hate you", "moron"}
PII_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "phone": r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d[ -]?){13,16}\b",
}


def detect_injection(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in INJECTION_PATTERNS)


def detect_toxicity(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in TOXIC_WORDS)


def detect_pii(text: str) -> list[str]:
    return [k for k, p in PII_PATTERNS.items() if re.search(p, text)]


def mask_pii(text: str) -> str:
    for k in ("ssn", "credit_card", "email", "phone"):
        text = re.sub(PII_PATTERNS[k], f"[{k.upper()}]", text)
    return text


def check_input(text: str) -> dict:
    return {"injection": detect_injection(text), "toxic": detect_toxicity(text),
            "pii": detect_pii(text), "sanitized": mask_pii(text)}


def risk_score(*, injection=False, toxic=False, pii_in=(), pii_out=(), grounded=True, n_sources=1, route="simple") -> float:
    s = 0.0
    s += 0.6 if injection else 0
    s += 0.2 if toxic else 0
    s += 0.1 * min(len(pii_in), 2)
    s += 0.3 * min(len(pii_out), 2)
    s += 0.4 if not grounded else 0
    s += 0.1 if (n_sources == 0 and route != "direct") else 0
    return round(min(s, 1.0), 2)


class RateLimiter:
    def __init__(self, per_min: int):
        self.per_min, self.hits, self.lock = per_min, defaultdict(deque), threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_min:
                return False
            q.append(now)
            return True
