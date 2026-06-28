# API Light — MEM Orchestrator v89

API Light is the API-assisted controller path used by MEM v89 for selected high-level decisions.

## Purpose

API Light helps evaluate controller-level actions such as whether to keep observing, refresh the current lane or approve a controlled lane switch.

It is not intended to call an external API for every training step.

## Requirements

```bash
OPENAI_API_KEY
```

Recommended compatibility alias:

```bash
API_KEY
```

## Safety

Never commit API keys, tokens or local secrets.

The local controller policy and safety guardrails remain important. API-assisted decisions should not bypass hard safety constraints.
