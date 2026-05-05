"""
exceptions.py — All custom exceptions for llm-watchdog
"""


class WatchdogError(Exception):
    """Base exception for all llm-watchdog errors."""


class BudgetExceeded(WatchdogError):
    """
    Raised when a Budget context manager's spending limit is crossed.

    Attributes:
        spent:   How much was spent (USD) at time of exception
        limit:   The configured budget limit (USD)
        user_id: The user/session that exceeded the budget
    """

    def __init__(self, spent: float, limit: float, user_id: str = "global") -> None:
        self.spent = spent
        self.limit = limit
        self.user_id = user_id
        super().__init__(
            f"Budget exceeded for '{user_id}': "
            f"spent ${spent:.6f}, limit ${limit:.6f}"
        )


class ProviderNotDetected(WatchdogError):
    """
    Raised when watchdog cannot identify the LLM provider
    from the response object.
    """

    def __init__(self, response_type: str) -> None:
        self.response_type = response_type
        super().__init__(
            f"Cannot detect provider from response type: '{response_type}'. "
            f"Supported: OpenAI ChatCompletion, Anthropic Message."
        )


class PricingModelNotFound(WatchdogError):
    """
    Raised when a model name has no entry in prices.json.
    Cost will be logged as 0.0 with a warning instead of crashing.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(
            f"No pricing data for model '{model}'. "
            f"Cost logged as $0.00. Add it to prices.json to track cost."
        )
