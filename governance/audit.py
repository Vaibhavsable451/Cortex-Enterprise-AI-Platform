from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from core.config import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
Session = sessionmaker(engine)


class Base(DeclarativeBase):
    pass


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    user: Mapped[str] = mapped_column(String(64))
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text, default="")
    route: Mapped[str] = mapped_column(String(16), default="")
    risk: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(24), default="ok")  # ok|blocked|pending_approval|approved|rejected|error
    grounded: Mapped[bool] = mapped_column(Boolean, default=True)
    n_sources: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    tokens: Mapped[int] = mapped_column(Integer, default=0)


def init_db():
    Base.metadata.create_all(engine)


def log(**kw) -> int:
    with Session() as s:
        row = AuditLog(**kw)
        s.add(row)
        s.commit()
        return row.id


def rows(limit=200, status=None):
    with Session() as s:
        q = select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)
        if status:
            q = q.where(AuditLog.status == status)
        return [{c.name: (getattr(r, c.name).isoformat() if isinstance(getattr(r, c.name), datetime)
                          else getattr(r, c.name)) for c in AuditLog.__table__.columns}
                for r in s.scalars(q)]


def set_status(row_id: int, status: str) -> dict | None:
    with Session() as s:
        r = s.get(AuditLog, row_id)
        if not r:
            return None
        r.status = status
        s.commit()
        return {"id": r.id, "answer": r.answer, "status": r.status}
