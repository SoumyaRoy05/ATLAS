import os
import socket
import logging
from pathlib import Path
from dotenv import load_dotenv
from pydantic import SecretStr

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv(Path(__file__).resolve().parent / ".env")


class _AfcWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Direct use of automatic function calling" not in record.getMessage()


logging.getLogger("google_genai.models").addFilter(_AfcWarningFilter())

# -----------------------------------------------------------------------------
# 1. Network Connectivity Probe
# -----------------------------------------------------------------------------
def is_online(host: str = "8.8.8.8", port: int = 53, timeout: float = 1.0) -> bool:
    """Checks DNS connectivity to skip cloud timeouts when offline."""
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
        return True
    except OSError:
        return False

# -----------------------------------------------------------------------------
# 2. 5-Tier Waterfall Setup
# -----------------------------------------------------------------------------
gemini_key = os.getenv("GEMINI_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

# Tier 1: Google GenAI
t1a = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=0.7,
    max_tokens=128,
    timeout=10,
    max_retries=0,
    thinking_level="low",
    google_api_key=SecretStr(gemini_key) if gemini_key else None,
)
t1b = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=128,
    timeout=10,
    max_retries=0,
    thinking_level="low",
    google_api_key=SecretStr(gemini_key) if gemini_key else None,
)

# Tier 2: Groq Cloud
t2a = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=128,
    timeout=10,
    max_retries=0,
    api_key=SecretStr(groq_key) if groq_key else None,
)
t2b = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=128,
    timeout=10,
    max_retries=0,
    api_key=SecretStr(groq_key) if groq_key else None,
)

# Tier 3: Local Ollama Air-Gap (RTX 5060)
t3 = ChatOllama(model="qwen2.5:3b", temperature=0.7, num_predict=128)

# Standardized waterfall runnable
llm_waterfall = t1a.with_fallbacks([t1b, t2a, t2b, t3])

def get_llm():
    """Returns the runnable LLM. Drops straight to local Ollama if offline."""
    return llm_waterfall if is_online() else t3