"""Same-sample-budget accounting.

Bandit / e-PS procedures spend one observation per pull. A later P0/P1
comparison is only valid when methods share the same budget and the same
evaluation times. This helper refuses overspend; it does not invent
sample-complexity numbers.
"""

from __future__ import annotations


class SampleBudget:
    """Hard cap on the number of arm pulls / observations."""

    def __init__(self, total: int) -> None:
        if isinstance(total, bool) or not isinstance(total, int):
            raise TypeError("total must be an int")
        if total < 0:
            raise ValueError("total must be >= 0")
        self.total = int(total)
        self.spent = 0

    @property
    def remaining(self) -> int:
        return self.total - self.spent

    def can_spend(self, n: int = 1) -> bool:
        if n < 0:
            raise ValueError("n must be >= 0")
        return self.spent + n <= self.total

    def spend(self, n: int = 1) -> int:
        """Consume ``n`` observations. Raises if this would exceed ``total``."""
        if n < 0:
            raise ValueError("n must be >= 0")
        if n == 0:
            return self.spent
        if self.spent + n > self.total:
            raise RuntimeError(
                f"sample budget overspend: spent={self.spent} + {n} > total={self.total}"
            )
        self.spent += n
        return self.spent

    def spend_up_to(self, n: int = 1) -> int:
        """Spend ``min(n, remaining)`` and return the actual amount spent."""
        if n < 0:
            raise ValueError("n must be >= 0")
        take = min(int(n), self.remaining)
        if take:
            self.spent += take
        return take

    def assert_not_overspent(self) -> None:
        if self.spent > self.total:
            raise AssertionError(
                f"sample budget overspent: spent={self.spent} total={self.total}"
            )
