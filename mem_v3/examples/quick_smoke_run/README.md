# Quick Smoke Run

This example provides a fast sanity check for the MEM v3 runtime and lane controller.

It exercises the real runtime execution path, memory sensing, and dataset batching behavior without requiring long-running benchmark iterations or external API keys.

## Recommended Smoke Run Approach

```bash
cd mem_v3
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# 1. Run static architecture and invariant validation
python scripts/v89_static_validation.py

# 2. Run automated test suite (38/38 passing)
pytest -v

# 3. Optional: Execute a short 50-step native PyTorch smoke run
python scripts/run_live_training.py --steps 50 --batch-size 4 --dataset tinystories
```

For a true short runtime check, use a deliberately small step count to verify that GPU memory sensing and checkpointing cycle properly without launching an extended endurance session.
