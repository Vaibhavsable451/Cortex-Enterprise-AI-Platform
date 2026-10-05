"""Document loading, cleaning, chunking, metadata extraction."""
import hashlib
import os
import re
from datetime import datetime, timezone


def load_file(path: str) -> list[dict]:
    """Return a list of {text, page} for a PDF / DOCX / TXT / MD file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        return [{"text": p.extract_text() or "", "page": i + 1}
                for i, p in enumerate(PdfReader(path).pages)]
    if ext == ".docx":
        import docx

        return [{"text": "\n".join(p.text for p in docx.Document(path).paragraphs), "page": 1}]
    if ext in (".txt", ".md", ".csv"):
        with open(path, encoding="utf-8", errors="ignore") as f:
            return [{"text": f.read(), "page": 1}]
    raise ValueError(f"Unsupported file type: {ext}")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)          # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Paragraph-aware sliding-window chunker."""
    if size <= overlap:
        raise ValueError("size must be > overlap")
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= size:
            cur = f"{cur}\n{p}".strip()
            continue
        if cur:
            chunks.append(cur)
        while len(p) > size:                        # very long paragraph
            chunks.append(p[:size])
            p = p[size - overlap:]
        cur = (chunks[-1][-overlap:] + "\n" + p).strip() if chunks and len(p) < size else p
    if cur:
        chunks.append(cur)
    return chunks


def ingest_file(path: str, department: str = "general") -> list[dict]:
    source = os.path.basename(path)
    doc_id = hashlib.md5(source.encode()).hexdigest()[:10]
    now = datetime.now(timezone.utc).isoformat()
    out = []
    for page in load_file(path):
        for j, c in enumerate(chunk_text(clean_text(page["text"]))):
            out.append({
                "id": f"{doc_id}-p{page['page']}-c{j}",
                "text": c,
                "metadata": {"source": source, "page": page["page"], "chunk": j,
                             "department": department, "ingested_at": now},
            })
    return out
