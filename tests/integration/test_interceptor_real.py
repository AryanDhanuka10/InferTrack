# tests/integration/test_interceptor_real.py
"""
Integration tests for intercept() with real Ollama calls.
Requires: ollama serve && ollama pull qwen2.5:0.5b
"""
from __future__ import annotations

import pytest
from infertrack import intercept, stop, is_active
from infertrack.storage.db import query_logs


@pytest.fixture(autouse=True)
def ensure_stopped():
    """Always stop interceptor after each test."""
    yield
    stop()


@pytest.mark.integration
class TestInterceptorRealCalls:

    def test_intercept_logs_real_call(self, ollama_client, model, tmp_db):
        intercept(tag="real-intercept", db_path=tmp_db)

        ollama_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}]
        )
        stop()

        logs = query_logs(db_path=tmp_db)
        assert len(logs) == 1

    def test_provider_detected(self, ollama_client, model, tmp_db):
        intercept(db_path=tmp_db)

        ollama_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}]
        )
        stop()

        assert query_logs(db_path=tmp_db)[0].provider == "openai"

    def test_tokens_logged(self, ollama_client, model, tmp_db):
        intercept(db_path=tmp_db)

        ollama_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is 2+2?"}]
        )
        stop()

        log = query_logs(db_path=tmp_db)[0]
        assert log.input_tokens  > 0
        assert log.output_tokens > 0

    def test_tag_stored(self, ollama_client, model, tmp_db):
        intercept(tag="global-tag", db_path=tmp_db)

        ollama_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}]
        )
        stop()

        assert query_logs(db_path=tmp_db)[0].tag == "global-tag"

    def test_multiple_clients_all_intercepted(self, model, tmp_db):
        """All client instances are patched at the class level."""
        from openai import OpenAI
        client_a = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        client_b = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

        intercept(db_path=tmp_db)

        client_a.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "hi"}]
        )
        client_b.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "hello"}]
        )
        stop()

        assert len(query_logs(db_path=tmp_db)) == 2

    def test_stop_restores_normal_behaviour(self, ollama_client, model, tmp_db):
        """After stop(), calls still work but are NOT logged."""
        intercept(db_path=tmp_db)
        ollama_client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "hi"}]
        )
        stop()

        # This call is after stop() — must NOT be logged
        ollama_client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "hi again"}]
        )

        assert len(query_logs(db_path=tmp_db)) == 1

    def test_is_active_states(self, ollama_client, model, tmp_db):
        assert is_active() is False
        intercept(db_path=tmp_db)
        assert is_active() is True
        stop()
        assert is_active() is False

