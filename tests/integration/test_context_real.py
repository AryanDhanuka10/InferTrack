# tests/integration/test_context_real.py
"""
Integration tests for watch() context manager with real Ollama calls.
Requires: ollama serve && ollama pull qwen2.5:0.5b
"""
from __future__ import annotations

import pytest
from infertrack import watch
from infertrack.storage.db import query_logs


@pytest.mark.integration
class TestWatchContextRealCalls:

    def test_single_response_tracked(self, ollama_client, model, tmp_db):
        with watch(db_path=tmp_db) as w:
            resp = ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}]
            )
            w.add_response(resp, db_path=tmp_db)

        assert w.call_count    == 1
        assert w.input_tokens  > 0
        assert w.output_tokens > 0
        assert w.tokens_used   > 0

    def test_cost_zero_ollama(self, ollama_client, model, tmp_db):
        with watch(db_path=tmp_db) as w:
            resp = ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}]
            )
            w.add_response(resp, db_path=tmp_db)

        assert w.cost_usd == 0.0

    def test_latency_covers_block(self, ollama_client, model, tmp_db):
        """Block latency must be >= time of one real API call."""
        with watch(db_path=tmp_db) as w:
            resp = ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}]
            )
            w.add_response(resp, db_path=tmp_db)

        assert w.latency_ms > 0

    def test_multiple_responses_accumulate(self, ollama_client, model, tmp_db):
        with watch(db_path=tmp_db) as w:
            for prompt in ["What is 1+1?", "What is 2+2?", "What is 3+3?"]:
                resp = ollama_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                w.add_response(resp, db_path=tmp_db)

        assert w.call_count    == 3
        assert w.tokens_used   > 0
        assert w.input_tokens  > 0
        assert w.output_tokens > 0

    def test_all_responses_written_to_db(self, ollama_client, model, tmp_db):
        with watch(db_path=tmp_db) as w:
            for _ in range(2):
                resp = ollama_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "hi"}]
                )
                w.add_response(resp, db_path=tmp_db)

        assert len(query_logs(db_path=tmp_db)) == 2

    def test_tag_forwarded_to_db(self, ollama_client, model, tmp_db):
        with watch(tag="ctx-integration", db_path=tmp_db) as w:
            resp = ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}]
            )
            w.add_response(resp, db_path=tmp_db)

        assert query_logs(db_path=tmp_db)[0].tag == "ctx-integration"

    def test_success_true_on_clean_block(self, ollama_client, model, tmp_db):
        with watch(db_path=tmp_db) as w:
            resp = ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}]
            )
            w.add_response(resp, db_path=tmp_db)

        assert w.success is True

    def test_exception_propagates(self, ollama_client, tmp_db):
        with pytest.raises(Exception):
            with watch(db_path=tmp_db) as w:
                ollama_client.chat.completions.create(
                    model="nonexistent-model:latest",
                    messages=[{"role": "user", "content": "hi"}]
                )

        assert w.success is False
