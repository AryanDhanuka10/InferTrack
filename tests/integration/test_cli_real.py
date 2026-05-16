# tests/integration/test_cli_real.py
"""
Integration tests for CLI commands after real Ollama calls.
Verifies that watchdog summary/tail/top/export reflect real logged data.
Requires: ollama serve && ollama pull qwen2.5:0.5b
"""
from __future__ import annotations

import csv
import io
import json
import pytest
from click.testing import CliRunner

from infertrack import watchdog
from infertrack.cli.__main__ import cli
from infertrack.storage.db import init_db


@pytest.fixture
def populated_real_db(ollama_client, model, tmp_db):
    """Make 3 real Ollama calls with different tags and users, return db path."""

    @watchdog(tag="summarise", user_id="alice", db_path=tmp_db)
    def ask_alice(prompt):
        return ollama_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

    @watchdog(tag="search", user_id="bob", db_path=tmp_db)
    def ask_bob(prompt):
        return ollama_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

    ask_alice("What is 1+1?")
    ask_alice("What is 2+2?")
    ask_bob("What is the capital of France?")

    return tmp_db


@pytest.mark.integration
class TestCLIRealData:

    def test_summary_shows_real_calls(self, populated_real_db):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "summary", "--last", "all", "--db", str(populated_real_db)
        ])
        assert result.exit_code == 0
        assert "3" in result.output          # 3 total calls
        assert "%" in result.output          # success rate shown

    def test_summary_shows_model(self, populated_real_db, model):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "summary", "--last", "all", "--db", str(populated_real_db)
        ])
        assert model in result.output

    def test_tail_shows_real_entries(self, populated_real_db):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "tail", "-n", "10", "--db", str(populated_real_db)
        ])
        assert result.exit_code == 0
        assert "$" in result.output          # cost column shown
        assert "ms" in result.output or "s" in result.output  # latency shown

    def test_tail_filter_by_tag(self, populated_real_db):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "tail", "--tag", "summarise", "--db", str(populated_real_db)
        ])
        assert result.exit_code == 0
        assert "summarise" in result.output

    def test_tail_filter_by_user(self, populated_real_db):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "tail", "--user", "alice", "--db", str(populated_real_db)
        ])
        assert result.exit_code == 0
        assert "alice" in result.output

    def test_top_by_cost_no_crash(self, populated_real_db):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "top", "--by", "cost", "--group", "user",
            "--last", "all", "--db", str(populated_real_db)
        ])
        assert result.exit_code == 0

    def test_top_by_calls_shows_alice(self, populated_real_db):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "top", "--by", "calls", "--group", "user",
            "--last", "all", "--db", str(populated_real_db)
        ])
        assert result.exit_code == 0
        assert "alice" in result.output    # alice made 2 calls

    def test_export_csv_correct_count(self, populated_real_db):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "export", "--format", "csv",
            "--last", "all", "--db", str(populated_real_db)
        ])
        assert result.exit_code == 0
        reader = csv.DictReader(io.StringIO(result.output))
        rows = list(reader)
        assert len(rows) == 3

    def test_export_json_correct_count(self, populated_real_db):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "export", "--format", "json",
            "--last", "all", "--db", str(populated_real_db)
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 3

    def test_export_json_tokens_nonzero(self, populated_real_db):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "export", "--format", "json",
            "--last", "all", "--db", str(populated_real_db)
        ])
        data = json.loads(result.output)
        for record in data:
            assert record["total_tokens"] > 0

    def test_export_filter_by_user(self, populated_real_db):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "export", "--format", "json", "--user", "bob",
            "--last", "all", "--db", str(populated_real_db)
        ])
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["user_id"] == "bob"
