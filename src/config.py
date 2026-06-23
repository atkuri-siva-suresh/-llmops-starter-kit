"""
Configuration for LLMOps Starter Kit.
All environment variables, paths, and thresholds in one place.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── LLM (Groq free tier) ──────────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0"))
    GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "1024"))

    # ── Embeddings (HuggingFace, local, no API key) ───────────────────────────
    EMBED_MODEL: str = os.getenv(
        "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBED_DEVICE: str = os.getenv("EMBED_DEVICE", "cpu")

    # ── MLflow ────────────────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "llmops-starter-kit"
    )

    # ── Inference Gateway ─────────────────────────────────────────────────────
    GATEWAY_HOST: str = os.getenv("GATEWAY_HOST", "0.0.0.0")
    GATEWAY_PORT: int = int(os.getenv("GATEWAY_PORT", "8000"))

    # ── Prompt Registry ───────────────────────────────────────────────────────
    PROMPTS_DIR: str = os.getenv("PROMPTS_DIR", "prompts")

    # ── Drift Detection ───────────────────────────────────────────────────────
    DRIFT_WINDOW_SIZE: int = int(os.getenv("DRIFT_WINDOW_SIZE", "5"))
    DRIFT_Z_THRESHOLD: float = float(os.getenv("DRIFT_Z_THRESHOLD", "1.5"))

    # ── Alert Thresholds ──────────────────────────────────────────────────────
    DEFAULT_FAITHFULNESS_THRESHOLD: float = float(
        os.getenv("DEFAULT_FAITHFULNESS_THRESHOLD", "0.70")
    )
    DEFAULT_LATENCY_THRESHOLD_MS: int = int(
        os.getenv("DEFAULT_LATENCY_THRESHOLD_MS", "3000")
    )
    DEFAULT_RELEVANCY_THRESHOLD: float = float(
        os.getenv("DEFAULT_RELEVANCY_THRESHOLD", "0.65")
    )

    # ── Data Paths ────────────────────────────────────────────────────────────
    SAMPLE_DOCS_DIR: str = os.getenv("SAMPLE_DOCS_DIR", "data/samples")
    TEST_QA_PATH: str = os.getenv("TEST_QA_PATH", "data/samples/test_qa.json")

    @classmethod
    def validate(cls) -> None:
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set.\n"
                "1. Sign up free at https://console.groq.com (no credit card)\n"
                "2. Copy .env.example to .env\n"
                "3. Add your key: GROQ_API_KEY=gsk_..."
            )


config = Config()
