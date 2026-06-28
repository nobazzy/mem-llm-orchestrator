#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== CHECKPOINT RESUME EVENTS ==="
grep -R -iE "checkpoint_resume_selected|checkpoint_resume_not_found|checkpoint_resume_failed|load_checkpoint|checkpoint_resume" evidence evidence_packets logs reports 2>/dev/null | tail -120 || true

echo
echo "=== FIRST LOSS PER SESSION ==="
python - <<'PY'
import json
from pathlib import Path
rows=[]
for p in sorted(Path('evidence').glob('*/runtime_milestones.jsonl')):
    first=None
    last=None
    for line in p.read_text(errors='ignore').splitlines():
        try: d=json.loads(line)
        except Exception: continue
        if d.get('loss') is not None:
            if first is None: first=d
            last=d
    if first:
        rows.append((p.parent.name, first.get('step'), first.get('loss'), last.get('step') if last else None, last.get('loss') if last else None))
for r in rows[-40:]:
    print(f"{r[0]} first_step={r[1]} first_loss={r[2]} last_step={r[3]} last_loss={r[4]}")
PY
