"""
llm-watchdog
============
Zero-config LLM call interceptor.
Track cost, latency, and token usage. Enforce budgets. No cloud required.

Quickstart:
    from llm_watchdog import watchdog

    @watchdog()
    def ask(prompt):
        return client.chat.completions.create(...)
"""

__version__ = "0.1.0"

# Public API — populated as days progress
# Day 3: from llm_watchdog.core.decorator import watchdog
# Day 3: from llm_watchdog.core.context import watch
# Day 4: from llm_watchdog.core.budget import Budget
# Day 4: from llm_watchdog.exceptions import BudgetExceeded
# Day 7: from llm_watchdog.core.interceptor import intercept
