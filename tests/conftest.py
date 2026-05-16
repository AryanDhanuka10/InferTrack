# tests/integration/conftest.py
"""
Shared fixtures for integration tests.
All tests in this folder require Ollama running at http://localhost:11434

Setup:
    ollama serve                  # start Ollama daemon
    ollama pull qwen2.5:0.5b     # pull smallest/fastest model
    pytest tests/integration/ -v -m integration
"""
from __future__ import annotations

import pytest
from pathlib import Path
from openai import OpenAI
from unittest.mock import MagicMock


@pytest.fixture
def mock_openai_response():
    """Mimics openai.types.chat.ChatCompletion structure."""
    response = MagicMock()
    response.model = "qwen2.5:0.5b"
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    response.choices[0].message.content = "Hello"
    return response


# Ollama client fixture                                                

@pytest.fixture(scope="session")
def ollama_client():
    """Real OpenAI-compatible client pointing at local Ollama."""
    return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


@pytest.fixture(scope="session")
def model():
    """Smallest/fastest Ollama model for testing."""
    return "qwen2.5:0.5b"


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Fresh SQLite DB for each test."""
    from infertrack.storage.db import init_db
    db = tmp_path / "integration_test.db"
    init_db(db)
    return db


# Ollama availability check                                            

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require Ollama running locally"
    )


@pytest.fixture(scope="session", autouse=True)
def check_ollama():
    """Skip entire integration suite if Ollama is not reachable."""
    import httpx
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=3)
        r.raise_for_status()
    except Exception:
        pytest.skip(
            "Ollama not running at http://localhost:11434. "
            "Start it with: ollama serve"
        )