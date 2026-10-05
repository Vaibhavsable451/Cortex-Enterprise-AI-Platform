import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX", "enterprise-rag")
    PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
    VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "pinecone")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBED_DIM = 384
    JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
    ANALYST_PASSWORD = os.getenv("ANALYST_PASSWORD", "analyst123")
    RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))
    RISK_APPROVAL_THRESHOLD = float(os.getenv("RISK_APPROVAL_THRESHOLD", "0.7"))
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/audit.db")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")


settings = Settings()
