"""LLM client abstraction with Gemini and Ollama implementations for clause extraction/summarization."""

import os
import time
from abc import ABC, abstractmethod

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "http://localhost:11434"
# llama3.2:3b was chosen over larger models (e.g. llama3.1:8b) for CPU-only
# inference speed: on modest hardware an 8B model can take 10-20+ minutes for
# a single contract-length call, making a 50-contract batch impractical. The
# 3B model trades some extraction/summarization quality for a batch that
# actually completes in reasonable time; swap DEFAULT_MODEL (or set
# OLLAMA_MODEL) if running on a machine with more RAM/a GPU.
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_TIMEOUT = 600
DEFAULT_RETRIES = 1

DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"
DEFAULT_GEMINI_TIMEOUT = 60
DEFAULT_GEMINI_RETRIES = 4


class LLMClient(ABC):
    """Abstract interface for a text-completion LLM backend."""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt: The user-facing prompt to send to the model.
            system: Optional system prompt/instructions, kept separate from the
                user prompt on backends that support it natively.

        Returns:
            The generated text, stripped of leading/trailing whitespace.
        """
        raise NotImplementedError


class OllamaClient(LLMClient):
    """LLMClient backed by a local Ollama server (POST /api/generate)."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
    ):
        """Initialize the Ollama client.

        Args:
            base_url: Ollama server base URL. Falls back to the OLLAMA_BASE_URL
                env var, then "http://localhost:11434".
            model: Model name to use. Falls back to the OLLAMA_MODEL env var,
                then "llama3.2:3b".
            timeout: Request timeout in seconds for each attempt.
            retries: Number of retries on timeout/connection error, in addition
                to the initial attempt.
        """
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.retries = retries

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a completion via Ollama's /api/generate endpoint.

        Retries on timeout/connection error with exponential backoff before
        raising a ConnectionError.

        Args:
            prompt: The user-facing prompt to send to the model.
            system: Optional system prompt, passed via Ollama's native "system"
                request field so it stays separate from the user prompt.

        Returns:
            The generated text, stripped of leading/trailing whitespace.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        last_error = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return response.json()["response"].strip()
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(2 ** attempt)

        raise ConnectionError(
            f"Failed to reach Ollama at {url} after {self.retries + 1} attempt(s): {last_error}"
        )


class GeminiClient(LLMClient):
    """LLMClient backed by the Gemini API (generateContent REST endpoint).

    Used in place of a local Ollama model when GEMINI_API_KEY is available:
    on CPU-only/low-RAM hardware, local models are both much slower (minutes
    per call) and less reliable at following the exact JSON schema we require
    (see extractor.py's retry/fallback logic, which exists largely to absorb
    this). A hosted model like Gemini responds in seconds and follows
    structured-output instructions far more consistently.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = DEFAULT_GEMINI_TIMEOUT,
        retries: int = DEFAULT_GEMINI_RETRIES,
    ):
        """Initialize the Gemini client.

        Args:
            api_key: Gemini API key. Falls back to the GEMINI_API_KEY env var.
            model: Model name to use. Falls back to the GEMINI_MODEL env var,
                then "gemini-flash-lite-latest".
            timeout: Request timeout in seconds for each attempt.
            retries: Number of retries on timeout/connection error, in addition
                to the initial attempt.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set (pass api_key= or set it in .env)")
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        self.timeout = timeout
        self.retries = retries

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a completion via the Gemini generateContent API.

        Retries with exponential backoff on timeout/connection errors and on
        5xx responses (transient server-side issues, e.g. temporary overload).
        4xx responses (bad model name, invalid key, malformed request) are
        raised immediately without retrying, since retrying won't fix them.

        Args:
            prompt: The user-facing prompt to send to the model.
            system: Optional system prompt, passed via Gemini's native
                "systemInstruction" field so it stays separate from the
                user prompt.

        Returns:
            The generated text, stripped of leading/trailing whitespace.
        """
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        last_error = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(
                    url, params={"key": self.api_key}, json=payload, timeout=self.timeout
                )
                # 429 (rate limit) and 5xx (transient server issues) are worth
                # retrying; other 4xx (bad model/key/request) will never
                # succeed on retry, so fail fast instead of wasting time.
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = RuntimeError(
                        f"Gemini API returned {response.status_code}: {response.text[:500]}"
                    )
                    if attempt < self.retries:
                        time.sleep(5 * (attempt + 1))
                    continue
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"Gemini API returned {response.status_code}: {response.text[:500]}"
                    )
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(2 ** attempt)

        raise ConnectionError(
            f"Failed to reach Gemini API after {self.retries + 1} attempt(s): {last_error}"
        )


def get_default_llm_client() -> LLMClient:
    """Return the best available LLMClient given the current environment.

    Prefers GeminiClient (faster, more reliable JSON output) if GEMINI_API_KEY
    is set; otherwise falls back to a local OllamaClient.
    """
    if os.getenv("GEMINI_API_KEY"):
        return GeminiClient()
    return OllamaClient()


if __name__ == "__main__":
    client = get_default_llm_client()
    label = client.model if isinstance(client, OllamaClient) else f"Gemini/{client.model}"
    print(f"Sending test prompt using {label}...")
    try:
        reply = client.generate("Say hello in one sentence.")
        print("Response:", reply)
    except (ConnectionError, RuntimeError, ValueError) as exc:
        print("Failed to get a response:", exc)
        if isinstance(client, OllamaClient):
            print("Make sure the Ollama server is running (`ollama serve`) and the model is")
            print(f"pulled (`ollama pull {client.model}`).")
