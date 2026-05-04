from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 1e-3


class TrainingLoop:
    """Thin orchestrator so notebooks call one API instead of ad-hoc code per cell."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def run(self) -> dict[str, float]:
        # This method intentionally stays framework-light until MiniFASNet architecture
        # and dataset loaders are connected. It provides a stable interface now.
        return {
            'epochs': float(self.config.epochs),
            'learning_rate': float(self.config.learning_rate),
        }
