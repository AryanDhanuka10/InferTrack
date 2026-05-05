"""
conftest.py — Shared pytest fixtures for llm-watchdog tests
============================================================

Two categories:
    Unit fixtures:        Pure mocks. No network. No Ollama. Always fast.
    Integration fixtures: Real Ollama calls. Mark with @pytest.mark.integration.

Key fixture: tmp_db
    Every unit test gets a fresh in-memory SQLite DB.
    This means tests NEVER touch ~/.llm-watchdog/logs.db
    and NEVER interfere with each other.
"""

import pytest
from unittest.mock import MagicMock
from llm_watchdog.config import configure, reset_config


# ── Config isolation ─────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolate_config():
    """
    Reset config to defaults after every test.
    Prevents state leakage between tests.
    autouse=True means this runs for EVERY test automatically.
    """
    yield
    reset_config()


@pytest.fixture
def tmp_db(tmp_path):
    """
    Point the DB at a temp file for this test only.
    After the test, it's deleted automatically by pytest.

    Usage:
        def test_something(tmp_db):
            # DB is isolated — does not touch ~/.llm-watchdog/
            ...
    """
    db_path = str(tmp_path / "test_watchdog.db")
    configure(db_path=db_path)
    return db_path


# ── Mock LLM responses ───────────────────────────────────────────────

@pytest.fixture
def mock_openai_response():
    """
    Simulates a real OpenAI/Ollama ChatCompletion response object.
    Matches exact structure of openai.types.chat.ChatCompletion.

    Token counts:
        prompt_tokens=10, completion_tokens=20, total_tokens=30
    """
    response = MagicMock()
    response.model = "qwen2.5:0.5b"
    response.object = "chat.completion"
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    response.usage.total_tokens = 30
    response.choices[0].message.content = "Hello! How can I assist you today?"
    response.choices[0].finish_reason = "stop"
    return response


@pytest.fixture
def mock_openai_response_gpt4():
    """
    Simulates a GPT-4o response (paid model, non-zero cost).
    Use this for cost calculation tests.
    """
    response = MagicMock()
    response.model = "gpt-4o"
    response.object = "chat.completion"
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    response.usage.total_tokens = 150
    response.choices[0].message.content = "This is a GPT-4o response."
    response.choices[0].finish_reason = "stop"
    return response


@pytest.fixture
def mock_anthropic_response():
    """
    Simulates a real Anthropic Message response object.
    Matches exact structure of anthropic.types.Message.
    """
    response = MagicMock()
    response.model = "claude-3-5-haiku-20241022"
    response.type = "message"
    response.usage.input_tokens = 15
    response.usage.output_tokens = 25
    response.content[0].text = "Hello from Claude!"
    response.stop_reason = "end_turn"
    return response


@pytest.fixture
def mock_failed_response():
    """Simulates an API call that raises an exception."""
    def raise_error(*args, **kwargs):
        raise ConnectionError("Ollama not running")
    return raise_error


# ── Integration fixtures (require Ollama running) ─────────────────────

@pytest.fixture
def ollama_client():
    """
    Real OpenAI client pointed at local Ollama.
    Only used in @pytest.mark.integration tests.

    Requires:
        ollama serve  (running in background)
        ollama pull qwen2.5:0.5b

    Usage:
        @pytest.mark.integration
        def test_real_call(ollama_client, tmp_db):
            response = ollama_client.chat.completions.create(
                model="qwen2.5:0.5b",
                messages=[{"role": "user", "content": "hi"}]
            )
            assert response.usage.prompt_tokens > 0
    """
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        return client
    except ImportError:
        pytest.skip("openai package not installed")


@pytest.fixture
def ollama_model():
    """Default model for integration tests — smallest and fastest."""
    return "qwen2.5:0.5b"
