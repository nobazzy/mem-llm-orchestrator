# Lanes — MEM Orchestrator v89

Lanes are predefined execution profiles used by the sustained controller.

## Main lane families

```txt
aggressive: higher throughput target, higher runtime pressure
fast: balanced throughput and stability
safe: conservative execution profile for recovery/stability
```

## Validated lane examples

```txt
safe_seq256
fast_seq256_zero0_gacc4
aggressive_seq256_zero0_gacc4
```

## Lane interpretation

A lane being safer does not mean it is always healthier for a long run.

```txt
safe = lower runtime risk / recovery-oriented
fast = balanced sustained productivity
aggressive = higher throughput attempt with higher pressure
```

In particular, `safe_seq256` should be treated as a recovery/stability lane, not as a lane that can remain active forever regardless of throughput.

---

## Same-lane refresh

Before switching lanes, MEM can attempt to recover productivity inside the current lane. This is useful when degradation appears temporary or when switching would add unnecessary instability.

Same-lane refresh should not override mandatory escape rules when floor conditions are violated.

---

## Controlled lane switching

Lane switching is used when the current lane is no longer suitable according to sustained degradation and guardrail signals.

Common switch reasons include:

```txt
sustained_throughput_drop
slow_degradation
optimizer_ratio_high
dynamic_grad_accum_needed
low_gpu_utilization
gacc_weak_lane
proactive_throughput_guard
```

---

## Absolute safe-lane escape

`mem_v3.2-absolute-safe-escape` adds a mandatory escape rule for the known `safe_seq256` throughput-prison case:

```txt
safe_seq256 + tokens/s < 25000 + 1 bad window
= forced escape to fast_seq256_zero0_gacc4
```

This prevents `safe_seq256` from remaining active when it is stable but unproductive.

Expected target:

```txt
fast_seq256_zero0_gacc4
```

Expected events:

```txt
safe_seq256_absolute_escape_armed
safe_seq256_hard_escape_override
safe_seq256_absolute_25k_floor_escape
lane_switch_applied
```

---

## API boundary for lane decisions

API Light may assist the controller, but it must not veto the absolute `safe_seq256` escape condition.

For this specific case:

```txt
API keep_current_lane cannot keep safe_seq256 active
API switch_lane with lane=None cannot block the escape
cooldown/recovery hold should not block the escape
fallback is forced to fast_seq256_zero0_gacc4
```

---

## Lane policy note

The controller should avoid unnecessary lane churn. It should not switch lanes only because another lane might be briefly faster.

However, when a lane violates a hard productivity floor, the controller should prioritize long-run completion and sustained throughput over conservatism.
