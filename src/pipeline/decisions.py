from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Decision:
    is_fake: bool
    score: float
    reason: str


def make_decision(prob_fake: float, threshold: float) -> Decision:
    is_fake = prob_fake >= threshold
    reason = "prob >= threshold" if is_fake else "prob < threshold"
    return Decision(is_fake=is_fake, score=prob_fake, reason=reason)


