import logging
import os
import time
import requests
from dotenv import load_dotenv
from typing import Any, List, Optional
from pydantic import Field
from langchain_core.language_models.llms import LLM
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# ===========================
# Read config from environment
# ===========================
RUNPOD_ENDPOINT_URL = os.getenv( "RUNPOD_ENDPOINT_URL", "https://api.runpod.ai/v2/wg49n0prq013uh/run")
RUNPOD_API_KEY      = os.getenv( "RUNPOD_API_KEY", "")
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL",  "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE    = os.getenv("EMBEDDING_DEVICE", "cpu")

# SQL Agent — separate RunPod endpoint (same API key)
RUNPOD_SQL_ENDPOINT_URL = os.getenv("RUNPOD_SQL_ENDPOINT_URL", "https://api.runpod.ai/v2/8lgr7xxh32rymr/run")

# Derive the GET status base URLs
_ENDPOINT_BASE        = RUNPOD_ENDPOINT_URL.rsplit("/run", 1)[0]
RUNPOD_STATUS_URL     = f"{_ENDPOINT_BASE}/status"

_SQL_ENDPOINT_BASE    = RUNPOD_SQL_ENDPOINT_URL.rsplit("/run", 1)[0]
RUNPOD_SQL_STATUS_URL = f"{_SQL_ENDPOINT_BASE}/status"


# ===========================
# RunPod Serverless LLM
# ===========================
class RunPodLLM(LLM):
    """
    Custom LangChain LLM that calls a RunPod Serverless endpoint.

    Per .invoke() call:
      1. POST {endpoint_url}              → get a unique job_id from RunPod
      2. GET  {status_url}/{job_id}       → poll until COMPLETED / FAILED
      3. Return output text to caller
    """

    # Pydantic fields — required by LangChain's LLM base class
    endpoint_url:   str   = Field(description="RunPod POST URL, e.g. .../run")
    status_url:     str   = Field(description="RunPod GET status base URL, e.g. .../status")
    api_key:        str   = Field(description="RunPod API key")
    temperature:    float = Field(default=0.2)
    max_tokens:     int   = Field(default=512)
    poll_interval:  float = Field(default=1.5, description="Seconds between status polls")
    timeout:        float = Field(default=120.0, description="Max wait seconds before timeout")

    @property
    def _llm_type(self) -> str:
        return "runpod_serverless"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # ── STEP 1 : POST → submit job, receive dynamic job_id ──────────
        post_resp = requests.post(
            self.endpoint_url,
            json={
                "input": {
                    "prompt": prompt,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
            },
            headers=headers,
            timeout=30,
        )
        post_resp.raise_for_status()

        post_data = post_resp.json()
        job_id    = post_data.get("id")

        if not job_id:
            raise ValueError(f"RunPod did not return a job id. Response: {post_data}")

        logger.info(f"[RunPod] Job submitted → id: {job_id}")

        # ── STEP 2 : GET with dynamic job_id, poll until COMPLETED ──────
        poll_url = f"{self.status_url}/{job_id}"
        elapsed  = 0.0

        while elapsed < self.timeout:
            time.sleep(self.poll_interval)
            elapsed += self.poll_interval

            get_resp = requests.get(poll_url, headers=headers, timeout=60)
            get_resp.raise_for_status()
            data   = get_resp.json()
            status = data.get("status", "UNKNOWN")

            logger.info(f"[RunPod] job {job_id} → {status} ({elapsed:.1f}s)")

            if status == "COMPLETED":
                output = data.get("output", "")
                logger.info(f"[RunPod] job {job_id} COMPLETED")
                if isinstance(output, dict):
                    # RunPod workers may return {"text": "..."} or {"output": "..."}
                    return output.get("text", output.get("output", str(output)))
                return str(output)

            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(
                    f"[RunPod] job {job_id} {status}: {data.get('error', 'No error message')}"
                )

        raise TimeoutError(
            f"[RunPod] job {job_id} did not complete within {self.timeout}s"
        )


# ===========================
# Shared AI Resources (Singleton)
# ===========================
class AIResources:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AIResources, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info("Initializing shared AI resources...")
        logger.info(f"RunPod Endpoint     : {RUNPOD_ENDPOINT_URL}")
        logger.info(f"RunPod Status       : {RUNPOD_STATUS_URL}")
        logger.info(f"RunPod SQL Endpoint : {RUNPOD_SQL_ENDPOINT_URL}")
        logger.info(f"Embeddings          : {EMBEDDING_MODEL} (device={EMBEDDING_DEVICE})")

        # Routing LLM — fast, only needs one-word intent output
        self.routing_llm = RunPodLLM(
            endpoint_url=RUNPOD_ENDPOINT_URL,
            status_url=RUNPOD_STATUS_URL,
            api_key=RUNPOD_API_KEY,
            temperature=0.2,
            max_tokens=15,
            poll_interval=0.8,
            timeout=60.0,
        )

        # Response LLM — standard, full answer
        self.response_llm = RunPodLLM(
            endpoint_url=RUNPOD_ENDPOINT_URL,
            status_url=RUNPOD_STATUS_URL,
            api_key=RUNPOD_API_KEY,
            temperature=0.2,
            max_tokens=1024,
            poll_interval=0.8,
            timeout=120.0,
        )

        # SQL Agent LLM — dedicated endpoint for Text-to-SQL (sqlcoder)
        self.sql_llm = RunPodLLM(
            endpoint_url=RUNPOD_SQL_ENDPOINT_URL,
            status_url=RUNPOD_SQL_STATUS_URL,
            api_key=RUNPOD_API_KEY,
            temperature=0.1,
            max_tokens=512,
            poll_interval=0.8,
            timeout=200.0,
        )

        # Shared Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': EMBEDDING_DEVICE}
        )

        self._initialized = True
        logger.info("Shared AI resources initialized ✅")


# ===========================
# SQL Endpoint Direct Caller
# ===========================
def call_sql_endpoint(query: str, schema: str, timeout: float = 200.0) -> str:
    """
    Call the SQL RunPod endpoint with the correct payload format.
    The SQL endpoint expects {"input": {"query": ..., "schema": ...}}
    NOT the standard {"input": {"prompt": ...}} format.
    """
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }

    post_resp = requests.post(
        RUNPOD_SQL_ENDPOINT_URL,
        json={"input": {"query": query, "schema": schema}},
        headers=headers,
        timeout=30,
    )
    post_resp.raise_for_status()

    post_data = post_resp.json()
    job_id = post_data.get("id")
    if not job_id:
        raise ValueError(f"RunPod SQL endpoint did not return a job id. Response: {post_data}")

    logger.info(f"[RunPod SQL] Job submitted → id: {job_id}")

    poll_url = f"{RUNPOD_SQL_STATUS_URL}/{job_id}"
    elapsed = 0.0
    poll_interval = 0.8

    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval

        get_resp = requests.get(poll_url, headers=headers, timeout=60)
        get_resp.raise_for_status()
        data = get_resp.json()
        status = data.get("status", "UNKNOWN")

        logger.info(f"[RunPod SQL] job {job_id} → {status} ({elapsed:.1f}s)")

        if status == "COMPLETED":
            output = data.get("output", "")
            logger.info(f"[RunPod SQL] job {job_id} COMPLETED")
            if isinstance(output, dict):
                return output.get("text", output.get("output", str(output)))
            return str(output)

        if status in ("FAILED", "CANCELLED"):
            raise RuntimeError(
                f"[RunPod SQL] job {job_id} {status}: {data.get('error', 'No error message')}"
            )

    raise TimeoutError(
        f"[RunPod SQL] job {job_id} did not complete within {timeout}s"
    )


# Global singleton instance
ai_resources = AIResources()
