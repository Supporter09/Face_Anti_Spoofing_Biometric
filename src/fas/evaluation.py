from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalMetrics:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        if total == 0:
            return 0.0
        return (self.tp + self.tn) / total

    @property
    def apcer(self) -> float:
        denominator = self.fp + self.tn
        if denominator == 0:
            return 0.0
        return self.fp / denominator

    @property
    def bpcer(self) -> float:
        denominator = self.tp + self.fn
        if denominator == 0:
            return 0.0
        return self.fn / denominator

    @property
    def acer(self) -> float:
        return (self.apcer + self.bpcer) / 2.0


def evaluate_binary_predictions(
    live_scores: list[float], labels: list[int], threshold: float
) -> EvalMetrics:
    tp = fp = tn = fn = 0

    for score, label in zip(live_scores, labels, strict=True):
        predicted_live = score >= threshold
        is_live = label == 1
        if predicted_live and is_live:
            tp += 1
        elif predicted_live and not is_live:
            fp += 1
        elif not predicted_live and not is_live:
            tn += 1
        else:
            fn += 1

    return EvalMetrics(tp=tp, fp=fp, tn=tn, fn=fn)
