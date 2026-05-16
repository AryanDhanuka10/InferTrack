# tests/integration/test_budget_real.py
"""
Integration tests for Budget() with real Ollama calls.
Ollama is free so BudgetExceeded won't trigger on normal calls.
We test that spend accumulates correctly and DB is updated per call.
Requires: ollama serve && ollama pull qwen2.5:0.5b
"""
from __future__ import annotations

import pytest
from infertrack import Budget, BudgetExceeded
from infertrack.storage.db import query_logs, get_total_cost


@pytest.mark.integration
class TestBudgetRealCalls:

    def test_free_calls_never_exceed_budget(self, ollama_client, model, tmp_db):
        """Ollama calls cost $0.00 — budget must never be exceeded."""
        with Budget(max_usd=0.001, db_path=tmp_db) as b:
            for _ in range(3):
                resp = ollama_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "hi"}]
                )
                b.add_response(resp)

        assert b.call_count   == 3
        assert b.spent_usd    == 0.0
        assert b.remaining_usd == 0.001

    def test_calls_logged_to_db(self, ollama_client, model, tmp_db):
        with Budget(max_usd=1.0, db_path=tmp_db) as b:
            for _ in range(2):
                resp = ollama_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "hi"}]
                )
                b.add_response(resp)

        assert len(query_logs(db_path=tmp_db)) == 2

    def test_user_id_stored_on_logs(self, ollama_client, model, tmp_db):
        with Budget(max_usd=1.0, user_id="real-alice", db_path=tmp_db) as b:
            resp = ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}]
            )
            b.add_response(resp)

        assert query_logs(db_path=tmp_db)[0].user_id == "real-alice"

    def test_get_total_cost_after_calls(self, ollama_client, model, tmp_db):
        """get_total_cost should return 0.0 after Ollama calls."""
        with Budget(max_usd=1.0, user_id="bob", db_path=tmp_db) as b:
            resp = ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}]
            )
            b.add_response(resp)

        total = get_total_cost(db_path=tmp_db, user_id="bob")
        assert total == 0.0

    def test_period_session_starts_at_zero(self, ollama_client, model, tmp_db):
        """period='session' must start from 0 even with prior DB history."""
        from infertrack.storage.models import CallLog
        from infertrack.storage.db import insert_log

        # Insert fake prior spend for the user
        insert_log(CallLog(
            provider="openai", model="gpt-4o",
            input_tokens=1000, output_tokens=500,
            cost_usd=0.99, latency_ms=100.0,
            success=True, user_id="session-user",
        ), db_path=tmp_db)

        # Session budget ignores prior spend
        with Budget(max_usd=0.001, user_id="session-user",
                    period="session", db_path=tmp_db) as b:
            assert b.spent_usd == 0.0
            resp = ollama_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}]
            )
            b.add_response(resp)

        # Still 0 because Ollama is free
        assert b.spent_usd == 0.0
