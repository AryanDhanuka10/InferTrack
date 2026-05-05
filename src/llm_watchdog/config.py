"""
config.py — Central configuration for llm-watchdog
====================================================
Single source of truth for all runtime settings.

Priority order (highest → lowest):
    1. Environment variables  (WATCHDOG_*)
    2. Values set via configure() in user code
    3. Defaults defined here

Why this file exists:
    Without config.py, settings like DB path, default tags, and
    pricing overrides get hardcoded in 5 different files. When a
    user wants to change the DB path, they'd have no obvious place
    to look. config.py is that obvious place.

Usage:
    from llm_watchdog.config import config

    # Read a setting
    db_path = config.db_path

    # Override at runtime (e.g. in tests)
    from llm_watchdog.config import configure
    configure(db_path=":memory:", default_tag="test")
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WatchdogConfig:
    """All runtime configuration in one place."""

    # ── Storage ──────────────────────────────────────────────────────
    db_path: str = field(
        default_factory=lambda: os.environ.get(
            "WATCHDOG_DB_PATH",
            str(Path.home() / ".llm-watchdog" / "logs.db")
        )
    )
    # Why: Users may want to change DB location (CI, Docker, multi-project)
    # Env var lets them do it without touching code.

    # ── Pricing ──────────────────────────────────────────────────────
    prices_path: str = field(
        default_factory=lambda: os.environ.get(
            "WATCHDOG_PRICES_PATH",
            ""   # empty = use bundled prices.json
        )
    )
    # Why: Power users may maintain their own pricing file with
    # enterprise/custom model costs. This lets them plug it in.

    # ── Default Tags ─────────────────────────────────────────────────
    default_tag: str = field(
        default_factory=lambda: os.environ.get(
            "WATCHDOG_DEFAULT_TAG",
            "default"
        )
    )
    # Why: In a large app, every call should have a tag.
    # If user forgets, this prevents NULL tag chaos in the DB.

    # ── Budget ───────────────────────────────────────────────────────
    global_budget_usd: float = field(
        default_factory=lambda: float(
            os.environ.get("WATCHDOG_GLOBAL_BUDGET_USD", "0")
        )
    )
    # Why: 0 means no global budget. Positive value = hard global cap.
    # Useful for CI environments where you want a safety net.

    # ── Retry ────────────────────────────────────────────────────────
    default_retry_count: int = field(
        default_factory=lambda: int(
            os.environ.get("WATCHDOG_RETRY_COUNT", "0")
        )
    )
    default_retry_backoff: str = field(
        default_factory=lambda: os.environ.get(
            "WATCHDOG_RETRY_BACKOFF",
            "exponential"   # or "linear"
        )
    )

    # ── Logging ──────────────────────────────────────────────────────
    silent: bool = field(
        default_factory=lambda: os.environ.get(
            "WATCHDOG_SILENT", "false"
        ).lower() == "true"
    )
    # Why: In production apps, you may not want any stdout output.
    # silent=True suppresses all watchdog prints.


# ── Singleton ────────────────────────────────────────────────────────
# One global config instance used across the entire package.
# Import this, don't instantiate WatchdogConfig yourself.
config = WatchdogConfig()


def configure(**kwargs) -> None:
    """
    Override config values at runtime.

    Example:
        from llm_watchdog.config import configure
        configure(db_path=":memory:", default_tag="pytest")

    Useful for:
        - Tests (use in-memory DB, never touch ~/.llm-watchdog/)
        - Multi-tenant apps (switch DB per request)
        - CI/CD (set silent=True, global_budget_usd=0.10)
    """
    for key, value in kwargs.items():
        if not hasattr(config, key):
            raise ValueError(
                f"Unknown config key: '{key}'. "
                f"Valid keys: {list(config.__dataclass_fields__.keys())}"
            )
        setattr(config, key, value)


def reset_config() -> None:
    """
    Reset all config to defaults.
    Call this in test teardown to avoid state leakage between tests.

    Example (conftest.py):
        @pytest.fixture(autouse=True)
        def clean_config():
            yield
            reset_config()
    """
    new = WatchdogConfig()
    for key in config.__dataclass_fields__:
        setattr(config, key, getattr(new, key))
