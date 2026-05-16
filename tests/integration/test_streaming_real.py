# tests/integration/test_streaming_real.py
"""
Integration tests for streaming response support with real Ollama calls.
Requires: ollama serve && ollama pull qwen2.5:0.5b
"""
from __future__ import annotations

import pytest
from infertrack import watchdog
from infertrack.core.streaming import StreamingWrapper
from infertrack.storage.db import query_logs


@pytest.mark.integration
class TestStreamingRealCalls:

    def test_stream_returns_wrapper(self, ollama_client, model, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask_stream(prompt):
            return ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True},
            )

        result = ask_stream("hi")
        assert isinstance(result, StreamingWrapper)

    def test_stream_yields_content(self, ollama_client, model, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask_stream(prompt):
            return ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True},
            )

        chunks = list(ask_stream("Say only the word hello."))
        content = "".join(
            c.choices[0].delta.content or ""
            for c in chunks
            if c.choices and c.choices[0].delta.content
        )
        assert len(content) > 0

    def test_stream_logged_after_exhaustion(self, ollama_client, model, tmp_db):
        """DB must be empty before stream is consumed, written after."""
        @watchdog(db_path=tmp_db)
        def ask_stream(prompt):
            return ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True},
            )

        wrapper = ask_stream("hi")
        assert len(query_logs(db_path=tmp_db)) == 0   # not logged yet

        list(wrapper)                                   # exhaust stream
        assert len(query_logs(db_path=tmp_db)) == 1    # now logged

    def test_stream_tokens_logged(self, ollama_client, model, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask_stream(prompt):
            return ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True},
            )

        list(ask_stream("What is 2 + 2?"))
        log = query_logs(db_path=tmp_db)[0]
        assert log.input_tokens  > 0
        assert log.output_tokens > 0

    def test_stream_cost_zero_ollama(self, ollama_client, model, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask_stream(prompt):
            return ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True},
            )

        list(ask_stream("hi"))
        assert query_logs(db_path=tmp_db)[0].cost_usd == 0.0

    def test_stream_model_logged(self, ollama_client, model, tmp_db):
        @watchdog(db_path=tmp_db)
        def ask_stream(prompt):
            return ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True},
            )

        list(ask_stream("hi"))
        assert query_logs(db_path=tmp_db)[0].model == model

    def test_stream_without_usage_still_logged(self, ollama_client, model, tmp_db):
        """Without stream_options, tokens=0 but call still recorded."""
        @watchdog(db_path=tmp_db)
        def ask_stream(prompt):
            return ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                # No stream_options -- tokens will be 0
            )

        list(ask_stream("hi"))
        logs = query_logs(db_path=tmp_db)
        assert len(logs) == 1
        assert logs[0].input_tokens  == 0
        assert logs[0].output_tokens == 0
