# tests/integration/test_decorator_real.py
"""
Integration tests for @watchdog decorator with real Ollama calls.
Requires: ollama serve && ollama pull qwen2.5:0.5b
"""
from __future__ import annotations

import pytest
from pathlib import Path

from infertrack import watchdog
from infertrack.storage.db import query_logs


@pytest.mark.integration
class TestWatchdogRealCalls:

    def test_basic_call_logged(self, ollama_client, model, tmp_db):
        """Real call must produce exactly one log entry."""
        @watchdog(db_path=tmp_db)
        def ask(prompt):
            return ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )

        ask("Say the word 'hello' only.")
        logs = query_logs(db_path=tmp_db)
        assert len(logs) == 1

    def test_provider_detected_as_openai(self, ollama_client, model, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask(prompt):
            return ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )

        ask("hi")
        log = query_logs(db_path=tmp_db)[0]
        assert log.provider == "openai"

    def test_model_name_recorded(self, ollama_client, model, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask(prompt):
            return ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )

        ask("hi")
        log = query_logs(db_path=tmp_db)[0]
        assert log.model == model

    def test_tokens_nonzero(self, ollama_client, model, tmp_db):
        """Ollama returns real token counts in response.usage."""
        @watchdog(db_path=tmp_db)
        def ask(prompt):
            return ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )

        ask("What is 2 + 2?")
        log = query_logs(db_path=tmp_db)[0]
        assert log.input_tokens  > 0
        assert log.output_tokens > 0
        assert log.total_tokens  > 0

    def test_cost_zero_for_ollama(self, ollama_client, model, tmp_db):
        """Ollama models are free — cost must always be 0.0."""
        @watchdog(db_path=tmp_db)
        def ask(prompt):
            return ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )

        ask("hi")
        assert query_logs(db_path=tmp_db)[0].cost_usd == 0.0

    def test_latency_positive(self, ollama_client, model, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask(prompt):
            return ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )

        ask("hi")
        assert query_logs(db_path=tmp_db)[0].latency_ms > 0

    def test_success_true(self, ollama_client, model, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask(prompt):
            return ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )

        ask("hi")
        assert query_logs(db_path=tmp_db)[0].success is True

    def test_tag_stored(self, ollama_client, model, tmp_db):
        @watchdog(tag="integration-test", db_path=tmp_db)
        def ask(prompt):
            return ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )

        ask("hi")
        assert query_logs(db_path=tmp_db)[0].tag == "integration-test"

    def test_user_id_stored(self, ollama_client, model, tmp_db):
        @watchdog(user_id="real-user", db_path=tmp_db)
        def ask(prompt):
            return ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )

        ask("hi")
        assert query_logs(db_path=tmp_db)[0].user_id == "real-user"

    def test_multiple_calls_all_logged(self, ollama_client, model, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask(prompt):
            return ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )

        ask("What is 1+1?")
        ask("What is 2+2?")
        ask("What is 3+3?")
        assert len(query_logs(db_path=tmp_db)) == 3

    def test_response_content_returned(self, ollama_client, model, tmp_db):
        """Decorator must pass response through unchanged."""
        @watchdog(db_path=tmp_db)
        def ask(prompt):
            return ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )

        response = ask("Say only the word YES.")
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_failed_call_logged(self, ollama_client, tmp_db):
        """Calling a non-existent model must log a failure."""
        @watchdog(db_path=tmp_db)
        def ask(prompt):
            return ollama_client.chat.completions.create(
                model="this-model-does-not-exist:latest",
                messages=[{"role": "user", "content": prompt}]
            )

        with pytest.raises(Exception):
            ask("hi")

        logs = query_logs(db_path=tmp_db)
        assert len(logs) == 1
        assert logs[0].success is False
        assert logs[0].error_msg is not None
