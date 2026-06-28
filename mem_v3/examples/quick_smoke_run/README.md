# Quick smoke run

This example is intended for a short sanity check of the v89 controller path.

It should use the real runtime path and real dataset/cache behavior. Do not replace it with synthetic fake-token benchmarks.

Recommended smoke-run approach:

```bash
cd mem_orchestrator_v89
source .venv/bin/activate
export OPENAI_API_KEY="PASTE_YOUR_OPENAI_API_KEY_HERE"
export API_KEY="$OPENAI_API_KEY"
python scripts/v89_static_validation.py
```

For a true short runtime check, use a deliberately small target only when you understand that it is not the official 300k validation run.
