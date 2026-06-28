from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EffectivenessThresholds:
    """Configurable thresholds that define what constitutes a meaningful change
    in loss or throughput between consecutive milestone recordings.

    loss_improvement:      minimum decrease in loss (absolute) to be classified
                           as 'positive_loss_trend'. Default 0.01 — below this
                           the move is within noise for typical LM pretraining.
    loss_degradation:      increase in loss that triggers 'negative_loss_trend'.
                           Default 0.25 — a quarter-unit spike is reliably bad
                           in fp16/bf16 runs; smaller moves may be batch noise.
    throughput_improvement: tokens/s gain to be classified as
                            'positive_throughput_trend'. Default 100 — roughly
                            1-2% of a typical GPU throughput at batch=4/seq=128,
                            i.e. a detectable but not trivial improvement.
    consecutive_negative_for_action: how many consecutive 'negative_loss_trend'
                                     recordings before suggest_action() returns
                                     a non-'none' suggestion. Default 2.
    """
    loss_improvement: float = 0.01
    loss_degradation: float = 0.25
    throughput_improvement: float = 100.0
    consecutive_negative_for_action: int = 2


@dataclass
class DecisionEffect:
    step: int
    directive: str
    lr: float
    gradient_clip_norm: float
    loss: Optional[float]
    tokens_processed: int
    steps_per_second: float
    tokens_per_second: float
    effectiveness: str = "pending"
    loss_delta_since_prev: Optional[float] = None
    throughput_delta_since_prev: Optional[float] = None
    ts: float = field(default_factory=time.time)


class AdaptiveRuntimeMemory:
    """Persistent decision-effect memory for v89.

    Records the outcome of each executive directive at milestone steps and
    exposes suggest_action() so the training loop can consult the history
    and make a local adjustment when the API directive is underperforming.

    Design constraints:
    - Local and deterministic — no additional API calls.
    - LocalPolicy remains authoritative; suggestions here are advisory only.
    - The API may provide moderate directives, but this memory records whether
      previous directives were actually useful.
    """

    def __init__(
        self,
        evidence_dir: str | Path | None,
        thresholds: EffectivenessThresholds | None = None,
    ) -> None:
        self.evidence_dir = Path(evidence_dir) if evidence_dir else None
        self.thresholds = thresholds or EffectivenessThresholds()
        self.effects: List[DecisionEffect] = []
        self._last_loss: Optional[float] = None
        self._last_tps: Optional[float] = None
        self._write_failures: int = 0
        if self.evidence_dir:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Optional[Path]:
        return (self.evidence_dir / "adaptive_memory.jsonl") if self.evidence_dir else None

    @property
    def summary_path(self) -> Optional[Path]:
        return (self.evidence_dir / "adaptive_memory_summary.json") if self.evidence_dir else None

    def record(
        self,
        *,
        step: int,
        directive: str,
        lr: float,
        gradient_clip_norm: float,
        loss: Optional[float],
        tokens_processed: int,
        steps_per_second: float,
        tokens_per_second: float,
    ) -> Dict[str, Any]:
        t = self.thresholds
        loss_delta = None if self._last_loss is None or loss is None else float(loss - self._last_loss)
        tps_delta = None if self._last_tps is None else float(tokens_per_second - self._last_tps)

        effectiveness = "neutral"
        if loss_delta is not None and loss_delta < -t.loss_improvement:
            effectiveness = "positive_loss_trend"
        elif loss_delta is not None and loss_delta > t.loss_degradation:
            effectiveness = "negative_loss_trend"
        elif tps_delta is not None and tps_delta > t.throughput_improvement:
            effectiveness = "positive_throughput_trend"

        eff = DecisionEffect(
            step=int(step),
            directive=str(directive),
            lr=float(lr),
            gradient_clip_norm=float(gradient_clip_norm),
            loss=None if loss is None else float(loss),
            tokens_processed=int(tokens_processed),
            steps_per_second=float(steps_per_second),
            tokens_per_second=float(tokens_per_second),
            effectiveness=effectiveness,
            loss_delta_since_prev=loss_delta,
            throughput_delta_since_prev=tps_delta,
        )
        self.effects.append(eff)
        self._last_loss = loss if loss is not None else self._last_loss
        self._last_tps = tokens_per_second
        payload = asdict(eff)
        if self.path:
            try:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(payload) + "\n")
            except Exception:
                self._write_failures += 1
        if self.summary_path:
            try:
                self.summary_path.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
            except Exception:
                self._write_failures += 1
        return payload

    def suggest_action(self) -> str:
        """Return a local advisory action based on recent effectiveness history.

        This is consulted by the training loop at each milestone and gives the
        runner a chance to tighten the gradient clip or reduce the LR locally
        when the API directive is not helping. LocalPolicy bounds are never
        bypassed — this is advisory only.

        Returns one of:
          'reduce_lr'       — consecutive negative loss trends detected; LR
                              may be too high for current chaos environment.
          'increase_clip'   — same signal but gradient clip is already relaxed;
                              tightening may stabilise the run faster.
          'none'            — insufficient history or no concerning pattern.
        """
        t = self.thresholds
        if len(self.effects) < t.consecutive_negative_for_action:
            return "none"
        recent = self.effects[-t.consecutive_negative_for_action:]
        all_negative = all(e.effectiveness == "negative_loss_trend" for e in recent)
        if not all_negative:
            return "none"
        # If gradient clip is already tight (<=0.5), prefer an LR reduction.
        # Otherwise tightening the clip is the cheaper first move.
        last_clip = recent[-1].gradient_clip_norm
        return "reduce_lr" if last_clip <= 0.5 else "increase_clip"

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for eff in self.effects:
            counts[eff.effectiveness] = counts.get(eff.effectiveness, 0) + 1
        return {
            "adaptive_memory_enabled": True,
            "decision_effects_recorded": len(self.effects),
            "effectiveness_counts": counts,
            "suggested_action": self.suggest_action(),
            "write_failures": self._write_failures,
            "last_effect": asdict(self.effects[-1]) if self.effects else None,
        }
