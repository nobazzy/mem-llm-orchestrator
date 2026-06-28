# Controller Policy — MEM Orchestrator v89

The v89 controller prioritizes:

```txt
Safety
Stability
Long-run completion
Sustained throughput
Short-term peak throughput
```

`mem_v3.2-absolute-safe-escape` keeps this priority order, but adds one narrow hard rule: `safe_seq256` must not remain active when it becomes a throughput prison.

---

## Decision signals

The controller observes runtime signals such as:

```txt
tokens/s
steps/s
loss
GPU utilization
VRAM
RAM
optimizer/sync pressure
data wait
bad windows
recent events
lane age
restart index
```

For the absolute safe-lane escape, the most important signals are:

```txt
current lane
tokens/s
bad windows
optimizer/sync pressure
steps/s
local fallback lane
degradation reasons
```

---

## Decision types

```txt
observe
same-lane refresh
controlled lane switch
block unsafe change
mandatory escape when floor conditions are violated
```

---

## Absolute safe-lane escape

`mem_v3.2-absolute-safe-escape` adds a mandatory escape condition for the known `safe_seq256` degradation case:

```txt
safe_seq256 + tokens/s < 25000 + 1 bad window
= forced escape to fast_seq256_zero0_gacc4
```

This rule exists because a lane can be stable but still unproductive. `safe_seq256` is safe against risk, but it should not be allowed to destroy sustained throughput during a long run.

Expected event sequence:

```txt
safe_seq256_absolute_escape_armed
safe_seq256_hard_escape_override
safe_seq256_absolute_25k_floor_escape
lane_switch_applied -> fast_seq256_zero0_gacc4
```

---

## API authority boundary

API Light is a controller aid, not the final authority for critical local safety/productivity rules.

For the absolute `safe_seq256` escape condition:

```txt
API keep_current_lane cannot veto the escape
API switch_lane with lane=None cannot block the escape
cooldown/recovery hold should not block the escape
target lane is forced to fast_seq256_zero0_gacc4
```

The local controller rule wins when the absolute floor condition is met.

---

## Guardrails

The controller should not switch lanes just because another lane might be faster for a short period. It should require evidence of sustained degradation, health risk or productivity loss.

The absolute safe-lane escape is intentionally narrow and should only apply to the known degraded `safe_seq256` condition.

Changes to controller policy should be treated as high-risk and must be revalidated.
