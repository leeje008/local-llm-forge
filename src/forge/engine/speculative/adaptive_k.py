from __future__ import annotations

from dataclasses import dataclass

# ===========================================================================
# Adaptive K — Dynamic Draft Length (Phase 7.3)
#
# Automatically adjusts the number of draft tokens based on observed
# acceptance rate. High acceptance → more drafts. Low → fewer.
# ===========================================================================

@dataclass
class AdaptiveKConfig:
    """Configuration for adaptive draft length."""

    initial_k: int = 3             # Starting draft length
    min_k: int = 1                 # Minimum draft length
    max_k: int = 10                # Maximum draft length
    target_acceptance: float = 0.7 # Target acceptance rate
    increase_threshold: float = 0.8 # Increase K if acceptance > this
    decrease_threshold: float = 0.4 # Decrease K if acceptance < this
    window_size: int = 10           # Rolling window for acceptance tracking
    step_size: int = 1              # K adjustment step


class AdaptiveKController:
    """Dynamically adjusts speculative decoding draft length.

    Tracks rolling acceptance rate and increases K when the draft model
    is performing well (high acceptance), decreases when it's struggling.

    This prevents wasting compute on long drafts that will be rejected,
    while maximizing throughput when the draft model is accurate.
    """

    def __init__(self, config: AdaptiveKConfig | None = None):
        self.config = config or AdaptiveKConfig()
        self._current_k = self.config.initial_k
        self._history: list[float] = []  # Recent acceptance rates

    @property
    def current_k(self) -> int:
        return self._current_k

    def reset(self):
        self._current_k = self.config.initial_k
        self._history.clear()

    def record_round(self, drafted: int, accepted: int):
        """Record the result of a speculative round and adjust K.

        Args:
            drafted: Number of tokens drafted.
            accepted: Number of tokens accepted by target model.
        """
        if drafted == 0:
            return

        rate = accepted / drafted
        self._history.append(rate)

        # Keep only recent window
        if len(self._history) > self.config.window_size:
            self._history = self._history[-self.config.window_size:]

        # Adjust K based on rolling average
        avg_rate = sum(self._history) / len(self._history)

        if avg_rate >= self.config.increase_threshold:
            self._current_k = min(
                self._current_k + self.config.step_size,
                self.config.max_k,
            )
        elif avg_rate <= self.config.decrease_threshold:
            self._current_k = max(
                self._current_k - self.config.step_size,
                self.config.min_k,
            )

    def get_stats(self) -> dict:
        avg_rate = sum(self._history) / max(len(self._history), 1)
        return {
            "current_k": self._current_k,
            "avg_acceptance_rate": round(avg_rate, 3),
            "window_size": len(self._history),
        }
