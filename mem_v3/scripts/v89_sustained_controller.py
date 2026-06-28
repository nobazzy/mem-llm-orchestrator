#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

VERSION = "v89.0.0"
CONFIRM = "I_UNDERSTAND_V89_RECOVERY_CONTROL"

@dataclass(frozen=True)
class Lane:
    name: str
    batch: int
    seq: int
    precision: str = "fp16"
    grad_accum: int = 2
    zero_stage: int = 1
    min_tokens: float = 0.0
    description: str = ""

LANES: Dict[str, Lane] = {
    # v89 golden rule: only the lanes that earned their keep survive.
    # Standard three-lane control: aggressive, fast fallback, safe last resort.
    "aggressive_seq256_zero0_gacc4": Lane(
        "aggressive_seq256_zero0_gacc4", batch=16, seq=256, grad_accum=4, zero_stage=0,
        min_tokens=18000, description="primary zero0 gacc4 lane"
    ),
    "fast_seq256_zero0_gacc4": Lane(
        "fast_seq256_zero0_gacc4", batch=10, seq=256, grad_accum=4, zero_stage=0,
        min_tokens=18000, description="stable fast zero0 gacc4 fallback"
    ),
    "safe_seq256": Lane(
        "safe_seq256", batch=8, seq=256, grad_accum=2, zero_stage=0,
        min_tokens=12000, description="last resort only; promote out when stable"
    ),
}
DEFAULT_ORDER = [
    "aggressive_seq256_zero0_gacc4",
    "fast_seq256_zero0_gacc4",
    "safe_seq256",
]
ALLOWED_ACTIONS = {"keep_current_lane", "restart_same_lane", "switch_lane", "stop_and_preserve_evidence"}


def now() -> float:
    return time.time()


def run(cmd: List[str], *, cwd: Path, env: Optional[Dict[str, str]] = None, stdout=None, stderr=None) -> subprocess.Popen:
    return subprocess.Popen(cmd, cwd=str(cwd), env=env or os.environ.copy(), stdout=stdout, stderr=stderr, text=True, preexec_fn=os.setsid)


def latest_progress(project: Path, *, after_ts: float = 0.0) -> Optional[Path]:
    """Return the active lane progress with the highest real step.

    Root cause fixed:
    - If runtime_progress_latest.json is missing/stale, the controller builds a
      fallback from adaptive_memory.jsonl.
    - That fallback file can then look "fresh" forever while frozen at step 1.
    - So every sample must compare runtime_progress_latest.json against the
      latest adaptive_memory.jsonl line and choose the higher step.
    """
    best_runtime_path: Optional[Path] = None
    best_runtime_data: Dict[str, Any] = {}
    best_runtime_step = -1
    best_runtime_mtime = 0.0

    for rp in (project / "evidence").glob("*/runtime_progress_latest.json"):
        try:
            if not rp.exists() or rp.stat().st_mtime < after_ts:
                continue
            data = json.loads(rp.read_text(encoding="utf-8"))
            if str(data.get("version")) != VERSION:
                continue
            step = int(data.get("step") or 0)
            mt = rp.stat().st_mtime
            if step > best_runtime_step or (step == best_runtime_step and mt > best_runtime_mtime):
                best_runtime_path = rp
                best_runtime_data = data
                best_runtime_step = step
                best_runtime_mtime = mt
        except Exception:
            continue

    best_adaptive_dir: Optional[Path] = None
    best_adaptive_data: Dict[str, Any] = {}
    best_adaptive_step = -1
    best_adaptive_mtime = 0.0

    for af in (project / "evidence").glob("*/adaptive_memory.jsonl"):
        try:
            if not af.exists() or af.stat().st_mtime < after_ts:
                continue
            last = ""
            with af.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.strip():
                        last = line.strip()
            if not last:
                continue
            data = json.loads(last)
            step = int(data.get("step") or 0)
            if step <= 0:
                continue
            mt = af.stat().st_mtime
            if step > best_adaptive_step or (step == best_adaptive_step and mt > best_adaptive_mtime):
                payload = dict(data)
                payload["version"] = VERSION
                payload["source"] = "adaptive_memory_fallback"
                payload.setdefault("ts", time.time())
                best_adaptive_dir = af.parent
                best_adaptive_data = payload
                best_adaptive_step = step
                best_adaptive_mtime = mt
        except Exception:
            continue

    runtime_is_fallback = str(best_runtime_data.get("source") or "") == "adaptive_memory_fallback"
    use_adaptive = False
    if best_adaptive_dir is not None:
        if best_runtime_path is None:
            use_adaptive = True
        elif best_adaptive_step > best_runtime_step:
            use_adaptive = True
        elif runtime_is_fallback and best_adaptive_mtime > best_runtime_mtime:
            use_adaptive = True

    if use_adaptive and best_adaptive_dir is not None:
        rp = best_adaptive_dir / "runtime_progress_latest.json"
        tmp = rp.with_suffix(rp.suffix + ".tmp")
        tmp.write_text(json.dumps(best_adaptive_data, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        os.replace(tmp, rp)
        return rp

    return best_runtime_path


def has_real_progress(metrics: Dict[str, Any], *, min_step: int = 1) -> bool:
    try:
        step = int(metrics.get("step") or 0)
    except Exception:
        step = 0
    if step < min_step:
        return False
    # A true progress heartbeat should include at least one real performance
    # field. This protects the controller from acting on empty metrics or stale
    # placeholder files before DeepSpeed has produced its first heartbeat.
    return any(k in metrics for k in ("tokens_per_second", "steps_per_second", "loss", "profiler"))




def tail_file(path: Path, lines: int = 80, max_chars: int = 12000) -> str:
    try:
        if not path.exists():
            return ""
        data = path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        out = "\n".join(data)
        return out[-max_chars:]
    except Exception as exc:
        return f"tail_failed:{type(exc).__name__}:{str(exc)[:240]}"


def latest_checkpoint_for_resume(project: Path) -> Optional[Path]:
    import os
    import json
    import time
    import re
    from pathlib import Path

    def _event(event, **payload):
        try:
            out = Path("evidence_packets")
            out.mkdir(parents=True, exist_ok=True)
            payload["event"] = event
            payload["ts"] = time.time()
            with (out / "v89_sustained_control_events.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception:
            pass

    def _read_json(path):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return {}

    def _walk_find_model(obj):
        found = []
        def walk(x):
            if isinstance(x, dict):
                for k, v in x.items():
                    kl = str(k).lower()
                    if kl in {"model_preset", "preset", "model_name", "model"} and v is not None:
                        found.append(str(v))
                    walk(v)
            elif isinstance(x, list):
                for y in x:
                    walk(y)
        walk(obj)
        return found

    def _extract_step(obj):
        best = 0
        def walk(x):
            nonlocal best
            if isinstance(x, dict):
                for k, v in x.items():
                    kl = str(k).lower()
                    if kl in {"global_step", "step", "train_step", "global_steps", "steps"}:
                        try:
                            best = max(best, int(float(v)))
                        except Exception:
                            pass
                    walk(v)
            elif isinstance(x, list):
                for y in x:
                    walk(y)
        walk(obj)
        return best

    def _size_tag(text):
        m = re.search(r"(\d+)\s*m", str(text).lower())
        return m.group(1) if m else ""

    project_root = Path(__file__).resolve().parents[1]
    cwd = Path.cwd()

    target_preset = (
        os.environ.get("MEM_MODEL_PRESET")
        or os.environ.get("MEM_V89_MODEL_PRESET")
        or os.environ.get("MODEL_PRESET")
        or ""
    )
    target_size = _size_tag(target_preset)

    roots = []
    for raw in [
        os.environ.get("MEM_CHECKPOINT_DIR", ""),
        "checkpoints",
        str(project_root / "checkpoints"),
        str(cwd / "checkpoints"),
    ]:
        if raw:
            r = Path(raw)
            if not r.is_absolute():
                roots.append(cwd / r)
                roots.append(project_root / r)
            else:
                roots.append(r)

    # dedup roots
    uniq_roots = []
    seen_roots = set()
    for r in roots:
        try:
            key = str(r.resolve())
        except Exception:
            key = str(r)
        if key not in seen_roots:
            seen_roots.add(key)
            uniq_roots.append(r)

    candidates = []

    for root in uniq_roots:
        try:
            latest = root / "v89_latest.txt"
            if latest.exists():
                raw = latest.read_text(encoding="utf-8", errors="ignore").strip()
                if raw:
                    q = Path(raw)
                    if not q.is_absolute():
                        q1 = root / q
                        q2 = project_root / q
                        q3 = cwd / q
                        for qq in [q1, q2, q3]:
                            candidates.append(qq / "mem_model_optimizer.pt" if qq.is_dir() else qq)
                    else:
                        candidates.append(q / "mem_model_optimizer.pt" if q.is_dir() else q)
        except Exception:
            pass

        try:
            for d in root.glob("v89_live_*"):
                if d.is_dir():
                    candidates.append(d / "mem_model_optimizer.pt")
        except Exception:
            pass

        try:
            for q in root.rglob("mem_model_optimizer.pt"):
                candidates.append(q)
        except Exception:
            pass

    # dedup candidates
    uniq = []
    seen = set()
    for q in candidates:
        q = Path(q)
        try:
            key = str(q.resolve())
        except Exception:
            key = str(q)
        if key not in seen:
            seen.add(key)
            uniq.append(q)

    valid = []
    skipped = []

    for q in uniq:
        try:
            if not q.exists():
                skipped.append({"path": str(q), "reason": "missing"})
                continue
            if not q.is_file():
                skipped.append({"path": str(q), "reason": "not_file"})
                continue

            meta_path = q.parent / "metadata.json"
            meta = _read_json(meta_path)
            models = _walk_find_model(meta)

            # Só rejeita se a metadata disser explicitamente outro tamanho.
            mismatch = False
            for m in models:
                ms = _size_tag(m)
                if target_size and ms and ms != target_size:
                    mismatch = True

            if mismatch:
                skipped.append({
                    "path": str(q),
                    "reason": "model_size_mismatch",
                    "target_preset": target_preset,
                    "metadata_models": models,
                })
                continue

            step = _extract_step(meta)
            mtime = q.stat().st_mtime
            size = q.stat().st_size

            valid.append({
                "path": q,
                "meta": meta_path,
                "step": step,
                "mtime": mtime,
                "size": size,
                "models": models,
            })

        except Exception as e:
            skipped.append({"path": str(q), "reason": f"exception:{type(e).__name__}:{str(e)[:120]}"})

    if not valid:
        _event(
            "checkpoint_resume_not_found_starting_from_scratch",
            target_preset=target_preset,
            target_size=target_size,
            roots=[str(x) for x in uniq_roots],
            candidates=[str(x) for x in uniq],
            skipped=skipped[-40:],
        )
        return None

    # Escolhe maior step; se metadata não tiver step, escolhe mais recente por mtime.
    valid.sort(key=lambda x: (int(x["step"]), float(x["mtime"]), int(x["size"])), reverse=True)
    best = valid[0]
    selected = str(best["path"])

    _event(
        "checkpoint_resume_selected",
        checkpoint_path=selected,
        metadata_path=str(best["meta"]),
        step=int(best["step"]),
        size=int(best["size"]),
        target_preset=target_preset,
        target_size=target_size,
        metadata_models=best["models"],
        candidates=len(valid),
    )

    return selected
def clear_runtime_progress(project: Path) -> None:
    # v89 fresh-progress gate: prevent a new lane from reading stale metrics
    # from the previous lane/restart. Evidence is packed before this runs.
    for p in (project / "evidence").glob("*/runtime_progress_latest.json"):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            try:
                p.rename(p.with_suffix(p.suffix + f".stale_{int(now())}"))
            except Exception:
                pass


def cleanup_project_processes(project: Path, *, aggressive_python_dash: bool = True) -> None:
    # Last-resort cleanup for children that escape the lane process group.
    # The observed failure mode in v89 left `python -` workers burning GPU
    # after the controller exited. This is intentionally scoped to known MEM
    # run patterns and chaos sidecar patterns.
    patterns = [
        "main.py.*--deepspeed",
        "main.py --llm",
        "deepspeed",
        "sha256sum chaos_tmp",
        "dd if=/dev/urandom",
        "io_pressure",
        "real_chaos",
    ]
    if aggressive_python_dash:
        patterns.append("^python -$")
    for sig in ("TERM", "KILL"):
        for pat in patterns:
            try:
                subprocess.run(["pkill", f"-{sig}", "-f", pat], cwd=str(project), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            except Exception:
                pass
        if sig == "TERM":
            time.sleep(3)
    try:
        subprocess.run(["bash", "scripts/stop_real_chaos_hard_sidecar.sh"], cwd=str(project), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    except Exception:
        pass


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc)}


def sample_gpu() -> Dict[str, Any]:
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=5).strip().splitlines()[0]
        parts = [x.strip() for x in out.split(",")]
        return {
            "gpu_utilization_percent": float(parts[0]),
            "gpu_memory_used_mib": float(parts[1]),
            "gpu_memory_total_mib": float(parts[2]),
            "gpu_temperature_c": float(parts[3]),
            "gpu_power_w": float(parts[4]),
        }
    except Exception as exc:
        return {"gpu_sample_error": str(exc)}


def sample_swap() -> Dict[str, Any]:
    try:
        out = subprocess.check_output(["bash", "-lc", "free -m | awk '/Swap:/ {print $2, $3, $4}'"], text=True, timeout=5).strip()
        total, used, free = [float(x) for x in out.split()]
        return {"swap_total_mib": total, "swap_used_mib": used, "swap_free_mib": free}
    except Exception as exc:
        return {"swap_sample_error": str(exc)}


def event_log(project: Path, payload: Dict[str, Any]) -> None:
    out = project / "evidence_packets" / "v89_sustained_control_events.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("ts", round(now(), 3))
    payload.setdefault("version", VERSION)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")




def controller_state_path(project: Path) -> Path:
    label = os.environ.get("MEM_CHECKPOINT_LABEL", "v89").strip() or "v89"
    return project / "checkpoints" / f"{label}_controller_state.json"


def save_controller_state(project: Path, payload: Dict[str, Any]) -> None:
    """Atomic controller progress state.

    This is separate from model/optimizer checkpoint because the runner owns
    micro_train_steps_completed, while the controller owns global_step.
    """
    path = controller_state_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = dict(payload)
    data["controller_state_label"] = os.environ.get("MEM_CHECKPOINT_LABEL", "v89").strip() or "v89"
    data["controller_state_saved_at"] = time.time()
    tmp.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_controller_state(project: Path) -> Dict[str, Any]:
    path = controller_state_path(project)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}

def write_controller_status(project: Path, *, state: str, lane: Optional[Lane] = None, reason: str = "",
                            restart_index: int = 0, global_step: int = 0, target_steps: int = 0,
                            lane_pid: int = 0, progress_age: float = -1.0, extra: Optional[Dict[str, Any]] = None) -> None:
    """v89 live status: monitor must not infer strong/running from stale progress."""
    payload: Dict[str, Any] = {
        "version": VERSION,
        "state": state,
        "lane": lane.name if lane else "",
        "reason": reason,
        "restart_index": int(restart_index),
        "global_step": int(global_step),
        "target_global_steps": int(target_steps),
        "lane_pid": int(lane_pid or 0),
        "progress_age_seconds": round(float(progress_age), 3) if progress_age is not None else None,
        "ts": now(),
    }
    if extra:
        payload.update(extra)
    out = project / "evidence" / "v89_controller_status_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def mark_global_progress_transition(project: Path, *, state: str, reason: str, lane: Lane,
                                    global_step: int, target_steps: int, restart_index: int) -> None:
    out = project / "evidence" / "v89_controller_global_progress_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": VERSION,
        "controller_state": state,
        "lane": lane.name,
        "global_step": int(global_step),
        "target_global_steps": int(target_steps),
        "restart_index": int(restart_index),
        "reason": reason,
        "tokens_per_second": 0,
        "steps_per_second": 0,
        "stale_progress_invalidated": True,
        "ts": now(),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def pack_evidence(project: Path, lane: Lane, reason: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = project / "evidence_packets" / f"v89_{lane.name}_{reason}_{ts}.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    latest = latest_progress(project)
    if latest:
        paths.append(latest.parent)
    for log in (project / "logs").glob(f"v89_*{lane.name}*.log"):
        paths.append(log)
    try:
        with tarfile.open(out, "w:gz") as tar:
            for p in paths:
                if p.exists():
                    tar.add(p, arcname=str(p.relative_to(project)))
        return str(out)
    except Exception as exc:
        return f"pack_failed:{exc}"


def api_lane_advisor(metrics: Dict[str, Any], gpu: Dict[str, Any], lane: Lane, allowed_lanes: List[str], local_reason: str) -> Dict[str, Any]:
    """v89 API Light: consulted only before non-critical lane changes.

    If API is unavailable, invalid, or conservative, LocalPolicy keeps the
    current lane. This prevents silent local churn when the lane is productive.
    """
    project = Path.cwd()
    if os.environ.get("MEM_V89_ENABLE_API", "1") != "1":
        event_log(project, {"event": "api_light_skipped", "reason": "MEM_V89_ENABLE_API_disabled", "lane": lane.name, "local_reason": local_reason})
        return {"source": "api_disabled_keep_current", "action": "keep_current_lane", "lane": lane.name, "reason": local_reason}
    if not os.environ.get("OPENAI_API_KEY"):
        event_log(project, {"event": "api_light_skipped", "reason": "OPENAI_API_KEY_missing", "lane": lane.name, "local_reason": local_reason})
        return {"source": "api_missing_keep_current", "action": "keep_current_lane", "lane": lane.name, "reason": local_reason}
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(timeout=float(os.environ.get("MEM_V89_API_TIMEOUT_SECONDS", "8")))
        payload = {
            "version": VERSION,
            "current_lane": asdict(lane),
            "allowed_lanes": allowed_lanes,
            "local_reasons": local_reason,
            "metrics": {
                "step": metrics.get("step"),
                "target_steps": metrics.get("target_steps"),
                "loss": metrics.get("loss"),
                "tokens_per_second": metrics.get("tokens_per_second"),
                "steps_per_second": metrics.get("steps_per_second"),
                "bottleneck": metrics.get("bottleneck") or metrics.get("bottleneck_classification"),
                "profiler": metrics.get("profiler"),
                "nan_or_inf_detected": metrics.get("nan_or_inf_detected"),
                "grad_nonfinite_detected": metrics.get("grad_nonfinite_detected"),
                "early_stop_reason": metrics.get("early_stop_reason"),
            },
            "gpu": gpu,
            "policy": (
                "Return JSON only. Allowed actions: keep_current_lane, restart_same_lane, switch_lane, stop_and_preserve_evidence. "
                "Approve switch_lane only if the current lane is clearly harmful, stalled, unsafe, or repeatedly underperforming. "
                "V89 mandatory floor escape: if local_reasons contains below_lane_min_tokens together with optimizer_ratio_force_zero0, "
                "or below_lane_min_tokens together with optimizer_ratio_high and low_gpu_utilization, choose switch_lane to the best allowed fallback lane. "
                "Do not answer keep_current_lane merely because there is no critical error when the lane is below its productivity floor. "
                "If current lane is still above its lane minimum, making good throughput/progress, and has no critical error, choose keep_current_lane."
            ),
        }
        event_log(project, {"event": "api_light_called", "call_type": "lane_change_approval", "lane": lane.name, "local_reason": local_reason})
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a conservative runtime lane-change approver. Return compact JSON only."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0,
            max_tokens=220,
            timeout=float(os.environ.get("MEM_V89_API_TIMEOUT_SECONDS", "8")),
        )
        text = resp.choices[0].message.content or "{}"
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start:end + 1] if start >= 0 and end >= start else text)
        data["source"] = "openai_lane_change_approval"
        event_log(project, {"event": "api_light_response", "call_type": "lane_change_approval", "lane": lane.name, "response": {k: data.get(k) for k in ("action", "lane", "reason", "source")}})
        return data
    except Exception as exc:
        event_log(project, {"event": "api_light_error", "call_type": "lane_change_approval", "lane": lane.name, "error_type": type(exc).__name__, "error": str(exc)[:300]})
        return {"source": "api_error_keep_current", "action": "keep_current_lane", "lane": lane.name, "reason": f"{local_reason}; api_error={type(exc).__name__}"}

def sustained_efficiency_score(tokens: float, optimizer_ratio: float, gpu_util: float, data_wait_ratio: float, best_tokens: float = 0.0, grad_accum: int = 2) -> float:
    """Score for comparing lane quality over noisy samples.

    Higher is better. It rewards sustained tokens/s and useful GPU work while
    penalizing optimizer/sync overhead, data wait, and large drop from the best
    observed throughput of the current lane.
    """
    tokens_component = min(tokens / 31000.0, 1.25)
    gpu_component = min(max(gpu_util, 0.0) / 70.0, 1.0) * 0.18
    optimizer_penalty = max(optimizer_ratio, 0.0) * 1.28
    data_penalty = max(data_wait_ratio, 0.0) * 0.78
    # v89: trend/drop matters here. We act before a lane slides
    # from 25k into the 18k-20k band for multiple windows.
    drop_penalty = 0.0
    if best_tokens > 0 and tokens > 0:
        drop_penalty = max(0.0, 1.0 - (tokens / best_tokens)) * 0.92
    gacc_penalty = 0.0
    if grad_accum <= 2:
        gacc_penalty = 0.34
    elif grad_accum == 3:
        gacc_penalty = 0.12
    return round(tokens_component + gpu_component - optimizer_penalty - data_penalty - drop_penalty - gacc_penalty, 6)


def choose_local_fallback(reasons: List[str], current_lane: Lane) -> str:
    """v89 simple fallback policy.

    No gacc3/gacc5 probes, no seq512 probes, no API lane creativity.
    The controller rotates only between primary, fast fallback, and safe.
    """
    r = set(reasons)
    name = current_lane.name

    if "swap_pressure" in r or "progress_stall_critical" in r or "oom_or_process_exit" in r:
        return "safe_seq256"

    if name == "aggressive_seq256_zero0_gacc4":
        return "fast_seq256_zero0_gacc4"
    if name == "fast_seq256_zero0_gacc4":
        if {"data_wait_severe", "progress_stall", "heavy_chaos_pressure"} & r:
            return "safe_seq256"
        return "aggressive_seq256_zero0_gacc4"
    if name == "safe_seq256":
        return "fast_seq256_zero0_gacc4"

    return "fast_seq256_zero0_gacc4"


def local_policy_validate(candidate: Dict[str, Any], current_lane: Lane, fallback_lane: str, allowed_lanes: List[str]) -> Dict[str, Any]:
    """Validate API/local lane decisions deterministically.

    MEM v3.2.2 local-policy fix:
    - API `switch_lane` with lane=None/"None"/"null"/"" is not a keep.
      It becomes the local fallback when that fallback is valid.
    - API `switch_lane` pointing back to the current lane is not silently kept
      when a different local fallback exists.
    - LocalPolicy remains the final authority over allowed lanes.
    """
    action = str(candidate.get("action", "keep_current_lane"))
    if action not in ALLOWED_ACTIONS:
        action = "keep_current_lane"

    raw_lane = candidate.get("lane", None)
    raw_lane_text = "" if raw_lane is None else str(raw_lane).strip()
    lane_missing = raw_lane is None or raw_lane_text.lower() in {"", "none", "null"}

    if action == "keep_current_lane":
        target = current_lane.name
    elif action == "switch_lane" and lane_missing:
        target = fallback_lane if fallback_lane in allowed_lanes else current_lane.name
    else:
        target = raw_lane_text or current_lane.name
        if target not in allowed_lanes:
            target = fallback_lane if fallback_lane in allowed_lanes else current_lane.name

    # Deterministic fallback: switch_lane with missing/invalid/current target
    # should use a valid local fallback whenever one exists. It must not become
    # keep_current_lane merely because the API omitted the lane field.
    api_switch_without_usable_target = action == "switch_lane" and (lane_missing or target == current_lane.name)
    if api_switch_without_usable_target and fallback_lane in allowed_lanes and fallback_lane != current_lane.name:
        target = fallback_lane
    elif action == "switch_lane" and target == current_lane.name:
        action = "keep_current_lane"

    if action == "restart_same_lane":
        target = current_lane.name

    return {
        "source": candidate.get("source", "unknown"),
        "action": action,
        "lane": target,
        "validated_by_local_policy": True,
        "local_policy_mode": "mem_v3_2_2_deterministic_local_fallback",
        "api_switch_without_usable_target": bool(api_switch_without_usable_target),
        "fallback_lane": fallback_lane,
        "raw_candidate_redacted": {k: v for k, v in candidate.items() if k not in {"raw", "prompt"}},
    }




def v89_mandatory_floor_escape(reasons: List[str], fallback_lane: str, current_lane: Lane, active_allowed: List[str]) -> bool:
    """V89 fix: API cannot keep a lane that is below its productivity floor.

    The previous failure mode was: local controller detected below_lane_min_tokens
    plus optimizer/sync pressure, but the API repeatedly answered
    keep_current_lane because there was no crash/NaN/OOM. In v89, these
    repeated bad-window cases are no longer treated as optional churn.
    """
    r = set(reasons)
    if fallback_lane == current_lane.name or fallback_lane not in active_allowed:
        return False
    if "below_lane_min_tokens" not in r:
        return False
    if "optimizer_ratio_force_zero0" in r:
        return True
    if "optimizer_ratio_high" in r and "low_gpu_utilization" in r:
        return True
    if "sustained_throughput_drop" in r and "slow_degradation" in r and "optimizer_ratio_high" in r:
        return True
    return False



def mem_v3_throughput_priority_override(d: Dict[str, Any], reasons: List[str], fallback_lane: str, current_lane: Lane, active_allowed: List[str], args: argparse.Namespace) -> tuple[bool, str]:
    """MEM v3 throughput-first local sovereignty.

    Churn is acceptable for this workload because checkpoint/resume preserves
    loss. Therefore API keep_current_lane must not trap the run in a lane that
    is repeatedly below its productivity floor while a valid fallback exists.

    This does not touch checkpointing, DeepSpeed, dataset, or resume logic. It
    only upgrades the final local policy decision from keep_current_lane to a
    fallback switch when throughput evidence is already bad.
    """
    if fallback_lane == current_lane.name or fallback_lane not in active_allowed:
        return False, "no_valid_different_fallback"

    tokens = float(d.get("tokens") or d.get("tokens_per_second") or 0.0)
    steps_s = float(d.get("steps_s") or d.get("steps_per_second") or 0.0)
    opt = float(d.get("optimizer_ratio") or 0.0)
    gpu_util = float(d.get("gpu_util") or 0.0)
    bad = int(d.get("bad_count") or d.get("bad_windows") or d.get("controller_bad_count") or 0)
    r = set(reasons or []) | set(d.get("reasons") or [])

    if tokens <= 0:
        return False, "waiting_real_tokens"
    if bad < max(1, int(args.local_throughput_override_bad_windows)):
        return False, "waiting_repeated_bad_windows"

    pressure = {
        "below_lane_min_tokens",
        "slow_degradation",
        "sustained_throughput_drop",
        "optimizer_ratio_high",
        "optimizer_ratio_force_zero0",
        "dynamic_grad_accum_needed",
        "proactive_throughput_guard",
        "low_gpu_utilization",
        "high_vram_low_compute",
        "gacc2_weak_lane",
        "gacc_le3_optimizer_escape",
    }
    has_pressure = bool(r & pressure)

    # Current lane's own contract is sovereign. If a lane is below its floor and
    # pressure is present, API keep_current_lane is treated as advisory only.
    if current_lane.min_tokens > 0 and tokens < float(current_lane.min_tokens) and has_pressure:
        return True, "below_current_lane_productivity_floor"

    # Throughput-first floor for this 1M run: when the model is making low
    # throughput and optimizer/sync or low-compute pressure is visible, rotate.
    if tokens < float(args.local_throughput_override_tokens):
        if opt >= float(args.local_throughput_override_optimizer_ratio):
            return True, "throughput_floor_plus_optimizer_pressure"
        if steps_s > 0 and steps_s < float(args.local_throughput_override_steps_per_second):
            return True, "throughput_floor_plus_low_steps_per_second"
        if has_pressure and gpu_util < float(args.local_throughput_override_gpu_util):
            return True, "throughput_floor_plus_low_gpu_utilization"
        if {"sustained_throughput_drop", "slow_degradation"} <= r:
            return True, "repeated_slow_throughput_degradation"

    return False, "throughput_override_not_required"

def safe_seq256_hard_escape_override(d: Dict[str, Any], fallback_lane: str, current_lane: Lane, active_allowed: List[str], args: argparse.Namespace) -> tuple[bool, str]:
    """MEM v3 absolute fix: safe_seq256 must never trap the run below 25k tokens/s.

    This is deliberately less conservative than the earlier patch.

    Contract:
    - only applies to safe_seq256;
    - one bad window is enough;
    - tokens/s below 25k is below the acceptable floor for the validated 8GB run;
    - API keep_current_lane, cooldown, and recovery-hold gates must not block it;
    - target is forced later to fast_seq256_zero0_gacc4.
    """
    if current_lane.name != "safe_seq256":
        return False, "not_safe_seq256"

    tokens = float(d.get("tokens") or d.get("tokens_per_second") or 0.0)
    opt = float(d.get("optimizer_ratio") or 0.0)
    bad = int(d.get("bad_count") or d.get("bad_windows") or d.get("controller_bad_count") or 0)
    steps_s = float(d.get("steps_s") or d.get("steps_per_second") or 0.0)
    reasons = set(d.get("reasons") or [])

    if tokens <= 0:
        return False, "no_real_tokens_yet"

    if bad < max(1, int(args.safe_hard_escape_bad_windows)):
        return False, "waiting_first_bad_window"

    # Absolute user-defined floor: any confirmed bad safe_seq256 window below
    # 25k is unacceptable for the 1M endurance run.
    if tokens < float(args.safe_hard_escape_tokens):
        return True, "safe_seq256_absolute_25k_floor_escape"

    # Secondary escape: if tokens are close to the floor but optimizer/sync is
    # clearly hot, rotate early instead of waiting for metric damage.
    optimizer_reasons = {
        "optimizer_ratio_high",
        "optimizer_ratio_force_zero0",
        "dynamic_grad_accum_needed",
        "gacc2_weak_lane",
        "gacc_le3_optimizer_escape",
        "optimizer_or_sync_bound",
    }
    if tokens < float(args.safe_hard_escape_attention_tokens) and (opt >= float(args.safe_hard_escape_hot_optimizer_ratio) or optimizer_reasons & reasons):
        return True, "safe_seq256_hot_optimizer_25k_attention_escape"

    if steps_s > 0 and steps_s < float(args.safe_hard_escape_steps_per_second) and opt >= float(args.safe_hard_escape_optimizer_ratio):
        return True, "safe_seq256_low_steps_absolute_escape"

    return False, "safe_seq256_above_absolute_floor"


def is_lane_banned(lane_name: str, global_step: int, banned_until: Dict[str, int]) -> bool:
    return int(banned_until.get(lane_name, 0)) > int(global_step)


def filter_allowed_lanes(base: List[str], global_step: int, banned_until: Dict[str, int]) -> List[str]:
    allowed = [x for x in base if not is_lane_banned(x, global_step, banned_until)]
    return allowed or ["fast_seq256_zero0_gacc4", "safe_seq256"]


def enforce_lane_cooldown(target: str, fallback: str, allowed: List[str]) -> str:
    if target in allowed:
        return target
    if fallback in allowed:
        return fallback
    if "fast_seq256_zero0_gacc4" in allowed:
        return "fast_seq256_zero0_gacc4"
    if "safe_seq256" in allowed:
        return "safe_seq256"
    return allowed[0]


def lane_step(metrics: Dict[str, Any]) -> int:
    try:
        return int(metrics.get("step") or 0)
    except Exception:
        return 0


def write_global_progress(project: Path, lane: Lane, metrics: Dict[str, Any], global_completed: int, lane_start_global: int, lane_restart_index: int, target_global_steps: int, decision_state: Optional[Dict[str, Any]] = None) -> None:
    step = lane_step(metrics)
    payload = {
        "version": VERSION,
        "lane": lane.name,
        "lane_batch": lane.batch,
        "lane_seq": lane.seq,
        "lane_zero_stage": lane.zero_stage,
        "lane_grad_accum": lane.grad_accum,
        "lane_step": step,
        "lane_start_global_step": lane_start_global,
        "global_step": global_completed + step,
        "target_global_steps": int(target_global_steps),
        "loss": metrics.get("loss"),
        "tokens_per_second": metrics.get("tokens_per_second"),
        "steps_per_second": metrics.get("steps_per_second"),
        "bottleneck": metrics.get("bottleneck") or metrics.get("bottleneck_classification"),
        "api_executive_enabled": metrics.get("api_executive_enabled"),
        "heartbeat_writes": metrics.get("heartbeat_writes"),
        "restart_index": lane_restart_index,
        "sustained_efficiency_score": (decision_state or {}).get("sustained_efficiency_score"),
        "bad_windows": (decision_state or {}).get("bad_count"),
        "best_tokens_per_second": (decision_state or {}).get("best_tokens"),
        "optimizer_ratio": (decision_state or {}).get("optimizer_ratio"),
        "data_wait_ratio": (decision_state or {}).get("data_wait_ratio"),
        "real_chaos_score": (decision_state or {}).get("chaos_score"),
        "host_cpu_percent": (decision_state or {}).get("host_cpu_percent"),
        "reasons": (decision_state or {}).get("reasons"),
        "recommended_fallback": (decision_state or {}).get("recommended_fallback"),
        "controller_state": "RUNNING" if not (decision_state or {}).get("reasons") else "DEGRADED_OBSERVED",
        "ts": now(),
    }
    out = project / "evidence" / "v89_controller_global_progress_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def is_critical_degradation(reasons: List[str]) -> bool:
    # v89: normal progress_stall is degraded, but not enough to bypass the
    # hard minimum lane window. Only critical stall, swap pressure, OOM-like
    # exits, or explicit safety faults may bypass the 10k hard gate.
    return any(r in {"progress_stall_critical", "swap_pressure", "oom_or_process_exit", "nan_or_inf"} for r in reasons)

def hard_degradation_override(d: Dict[str, Any], lane: Lane, args: argparse.Namespace, current_lane_step: int, tokens: float, best_tokens: float) -> tuple[bool, str]:
    """Bypass fixed min-step gates when a lane is objectively bad.

    prior evidence showed LocalPolicy could hold `fast_seq256` until the
    minimum proof window even at ~14k tokens/s, optimizer_ratio ~0.47, and
    sustained_score < -0.8. v89 keeps long proof windows for healthy lanes,
    but degradation overrides are sovereign for objectively bad lanes.
    """
    opt = float(d.get("optimizer_ratio") or 0.0)
    score = float(d.get("sustained_efficiency_score") or 0.0)
    steps_s = float(d.get("steps_s") or 0.0)
    gpu_util = float(d.get("gpu_util") or 0.0)
    bottleneck = str(d.get("bottleneck") or "")

    # Do not act on the first tiny warmup samples unless the signal is extreme.
    if current_lane_step < args.hard_override_min_step:
        if opt >= args.hard_override_optimizer_ratio + 0.08 or tokens < args.hard_override_tokens - 2500:
            return True, "extreme_degradation_during_warmup"
        return False, "warmup_protected"

    # Universal emergency performance override.
    if tokens > 0 and tokens < args.hard_override_tokens:
        return True, "tokens_below_hard_override"
    if opt >= args.hard_override_optimizer_ratio:
        return True, "optimizer_ratio_above_hard_override"
    if score <= args.hard_override_score:
        return True, "sustained_score_below_hard_override"
    if steps_s > 0 and steps_s < args.hard_override_steps_per_second:
        return True, "steps_per_second_below_hard_override"
    if gpu_util < args.hard_override_gpu_util and bottleneck == "optimizer_or_sync_bound" and tokens < args.proactive_min_tokens:
        return True, "low_gpu_optimizer_bound_hard_override"

    # gacc2/gacc3 lanes are not allowed to consume long proof windows when weak.
    if lane.grad_accum <= 2 and (tokens < args.gacc2_escape_tokens or opt >= args.gacc2_escape_optimizer_ratio):
        return True, "gacc2_weak_lane_hard_override"
    if lane.grad_accum <= 3 and opt >= args.gacc_le3_escape_optimizer_ratio:
        return True, "gacc_le3_optimizer_hard_override"

    # Primary lane is protected longer, but not if it becomes clearly bad.
    if lane.name == "fast_seq256_zero0_gacc4":
        if tokens < args.primary_hard_override_tokens or opt >= args.primary_hard_override_optimizer_ratio:
            return True, "primary_lane_objectively_degraded"

    return False, "no_hard_override"

def build_command(project: Path, lane: Lane, steps: int, port: int) -> List[str]:
    candidates = [project / ".venv312" / "bin" / "python", project / ".venv" / "bin" / "python"]
    py = next((str(p) for p in candidates if p.exists()), sys.executable)

    # mem_v3: workload configuration is intentionally injected at the launcher
    # boundary, not inside controller policy. This preserves the validated v89
    # core while allowing custom datasets/tokenizers through environment vars.
    dataset_name = os.environ.get("MEM_DATASET_NAME", "HuggingFaceFW/fineweb-edu")
    dataset_config = os.environ.get("MEM_DATASET_CONFIG", "sample-10BT")
    dataset_split = os.environ.get("MEM_DATASET_SPLIT", "train")
    dataset_fallback = os.environ.get("MEM_DATASET_FALLBACK_NAME", "roneneldan/TinyStories")
    dataset_mix = os.environ.get("MEM_DATASET_MIX", "HuggingFaceFW/fineweb-edu:sample-10BT,roneneldan/TinyStories:")
    tokenizer_name = os.environ.get("MEM_TOKENIZER_NAME", "gpt2")
    model_preset = os.environ.get("MEM_MODEL_PRESET", "tiny_decoder")
    chaos_profile = os.environ.get("MEM_CHAOS_PROFILE", "real_desktop_contention")
    load_checkpoint = os.environ.get("MEM_DEEPSPEED_LOAD_CHECKPOINT", "").strip()
    if not load_checkpoint and os.environ.get("MEM_AUTO_RESUME_CHECKPOINT", "1") == "1":
        latest = latest_checkpoint_for_resume(project)
        if latest is not None:
            load_checkpoint = str(latest)
            event_log(project, {"event": "checkpoint_resume_selected", "checkpoint_path": load_checkpoint, "checkpoint_label": os.environ.get("MEM_CHECKPOINT_LABEL", "v89")})
        else:
            event_log(project, {"event": "checkpoint_resume_not_found_starting_from_scratch", "checkpoint_label": os.environ.get("MEM_CHECKPOINT_LABEL", "v89")})

    cmd = [
        py, "main.py",
        "--deepspeed-wsl-accelerated", "--deepspeed-aggressive-bounded",
        "--deepspeed-real-micro-train", "--deepspeed-real-limited-apply", "--deepspeed-persistent-checkpoint",
        "--real-dataset", "--dataset-name", dataset_name, "--dataset-config", dataset_config,
        "--dataset-split", dataset_split,
        "--dataset-fallback-name", dataset_fallback,
        "--dataset-mix", dataset_mix,
        "--tokenizer-name", tokenizer_name, "--sequence-length", str(lane.seq), "--model-preset", model_preset,
        "--chaos-profile", chaos_profile, "--guardrail-mode", "sampled", "--guardrail-sample-interval", "8",
        "--gradient-audit", "--operator", "--benchmark-mode", "mem_real_chaos",
        "--deepspeed-max-steps", str(steps), "--deepspeed-batch-size", str(lane.batch),
        "--deepspeed-zero-stage", str(lane.zero_stage), "--deepspeed-precision", lane.precision,
        "--deepspeed-gradient-accumulation-steps", str(lane.grad_accum),
        "--confirm-deepspeed-accelerated", CONFIRM,
        "--llm", "--api-executive-mode",
    ]
    if load_checkpoint:
        cmd += ["--deepspeed-load-checkpoint", load_checkpoint]
    return cmd


def terminate(proc: subprocess.Popen, timeout: int = 15) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.terminate()
    deadline = now() + timeout
    while now() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        proc.kill()


def decide(metrics: Dict[str, Any], gpu: Dict[str, Any], swap: Dict[str, Any], lane: Lane, best_tokens: float, bad_count: int, args: argparse.Namespace, lane_start_ts: float) -> Dict[str, Any]:
    age = 10**9
    progress_path = latest_progress(Path.cwd(), after_ts=lane_start_ts)
    if progress_path:
        age = now() - progress_path.stat().st_mtime
    step = int(metrics.get("step") or 0)
    tokens = float(metrics.get("tokens_per_second") or 0.0)
    steps_s = float(metrics.get("steps_per_second") or 0.0)
    bottleneck = str(metrics.get("bottleneck") or metrics.get("bottleneck_classification") or "unknown")
    profiler = metrics.get("profiler") or {}
    optimizer_ratio = float(profiler.get("optimizer_ratio") or 0.0)
    data_wait_ratio = float(profiler.get("data_wait_ratio") or 0.0)
    gpu_util = float(gpu.get("gpu_utilization_percent") or 0.0)
    gpu_mem = float(gpu.get("gpu_memory_used_mib") or 0.0)
    swap_used = float(swap.get("swap_used_mib") or 0.0)
    chaos_env = metrics.get("chaos_environment") or {}
    chaos_score = float(metrics.get("real_chaos_score") or 0.0)
    host_cpu = float(chaos_env.get("cpu_percent") or 0.0)
    efficiency_score = sustained_efficiency_score(tokens, optimizer_ratio, gpu_util, data_wait_ratio, best_tokens, lane.grad_accum)
    reasons: List[str] = []

    if age > args.stall_seconds:
        reasons.append("progress_stall")
    if age > getattr(args, "hard_stall_seconds", 180):
        reasons.append("progress_stall_critical")
    if swap_used > args.max_swap_mib:
        reasons.append("swap_pressure")
    if bool(metrics.get("nan_or_inf_detected")):
        reasons.append("nan_or_inf")
    if bool(metrics.get("grad_nonfinite_detected")):
        reasons.append("nonfinite_gradient")

    if step >= args.min_degrade_step:
        # v89: lane min is enforced alongside best*drop_ratio,
        # so a bad lane could observe forever at 12k–13k tokens/s.
        if tokens > 0 and tokens < lane.min_tokens:
            reasons.append("below_lane_min_tokens")
        if best_tokens > 0 and tokens < best_tokens * args.drop_ratio:
            reasons.append("sustained_throughput_drop")
        # v89 slow degradation: catch gradual decline before a hard collapse.
        # This fires only after the proof window and requires either optimizer heat
        # or low GPU utilization, avoiding switches on harmless loss noise.
        if best_tokens > 0 and tokens > 0 and tokens < best_tokens * args.slow_degradation_ratio and (optimizer_ratio >= args.slow_degradation_optimizer_ratio or gpu_util < args.min_gpu_util):
            reasons.append("slow_degradation")
        if bottleneck == "optimizer_or_sync_bound" and steps_s < args.min_steps_per_second:
            reasons.append("slow_optimizer_bound")
        if bottleneck == "optimizer_or_sync_bound" and optimizer_ratio >= args.max_optimizer_ratio:
            reasons.append("optimizer_ratio_high")
        # Strong v89 condition: optimizer_ratio > 0.42 for repeated windows should force zero0.
        if bottleneck == "optimizer_or_sync_bound" and optimizer_ratio >= args.force_zero0_optimizer_ratio:
            reasons.append("optimizer_ratio_force_zero0")
        if data_wait_ratio >= args.data_wait_ratio_high:
            reasons.append("data_wait_high")
        if data_wait_ratio >= args.data_wait_ratio_severe:
            reasons.append("data_wait_severe")
        if bottleneck == "optimizer_or_sync_bound" and optimizer_ratio >= args.dynamic_gacc_optimizer_ratio and lane.grad_accum <= 4:
            reasons.append("dynamic_grad_accum_needed")
        # v89 gacc2 protection: weak gacc2 lanes escape quickly
        # when they fall below 18k tokens/s or cross optimizer_ratio > 0.42.
        if lane.grad_accum <= 2 and (tokens < args.gacc2_escape_tokens or optimizer_ratio > args.gacc2_escape_optimizer_ratio):
            reasons.append("gacc2_weak_lane")
        # v89 strong rule: gacc <= 3 should not be held under hot optimizer/sync.
        if lane.grad_accum <= 3 and optimizer_ratio > args.gacc_le3_escape_optimizer_ratio:
            reasons.append("gacc_le3_optimizer_escape")
        # v89 proactive guard: act before a healthy 25k lane slides into the low-20k band.
        if tokens > 0 and tokens < args.proactive_min_tokens and step >= args.min_degrade_step and lane.grad_accum <= 4:
            reasons.append("proactive_throughput_guard")
        # v89 condition: low GPU utilization for repeated windows is treated as efficiency loss.
        if gpu_util < args.min_gpu_util and tokens > 0 and step >= args.min_degrade_step:
            reasons.append("low_gpu_utilization")
        if gpu_mem >= args.high_vram_mib and gpu_util < args.min_gpu_util:
            reasons.append("high_vram_low_compute")
        if host_cpu >= args.max_host_cpu_percent and gpu_util < args.min_gpu_util:
            reasons.append("host_cpu_starving_gpu")
        if chaos_score >= args.max_chaos_score and steps_s < args.min_steps_per_second:
            reasons.append("heavy_chaos_pressure")

    # MEM v89 recovery mode: during plateau recovery, do not churn lanes on
    # harmless desktop telemetry noise. Keep only safety/true-failure reasons.
    # This is controlled by run_v89_50m_recovery.sh and does not affect normal v89 runs.
    if os.environ.get("MEM_V89_RECOVERY_STABILITY_LOCK", "0").strip().lower() in {"1", "true", "yes", "on"}:
        ignored = {
            "low_gpu_utilization",
            "sustained_throughput_drop",
            "slow_degradation",
            "proactive_throughput_guard",
            "below_lane_min_tokens",
            "high_vram_low_compute",
            "host_cpu_starving_gpu",
        }
        filtered = [r for r in reasons if r not in ignored]
        if len(filtered) != len(reasons):
            try:
                event_log(Path.cwd(), {
                    "event": "recovery_stability_lock_filtered_reasons",
                    "lane": lane.name,
                    "step": step,
                    "ignored_reasons": [r for r in reasons if r in ignored],
                    "kept_reasons": filtered,
                    "tokens": tokens,
                    "steps_s": steps_s,
                    "gpu_util": gpu_util,
                    "optimizer_ratio": optimizer_ratio,
                    "data_wait_ratio": data_wait_ratio,
                })
            except Exception:
                pass
        reasons = filtered

    degraded = bool(reasons)
    bad_count = bad_count + 1 if degraded else 0
    return {
        "degraded": degraded,
        "bad_count": bad_count,
        "reasons": reasons,
        "age": age,
        "step": step,
        "tokens": tokens,
        "best_tokens": best_tokens,
        "steps_s": steps_s,
        "gpu_util": gpu_util,
        "gpu_mem": gpu_mem,
        "bottleneck": bottleneck,
        "optimizer_ratio": optimizer_ratio,
        "data_wait_ratio": data_wait_ratio,
        "chaos_score": chaos_score,
        "host_cpu_percent": host_cpu,
        "sustained_efficiency_score": efficiency_score,
        "recommended_fallback": choose_local_fallback(reasons, lane) if reasons else "observe",
    }



def v89_midband_switch_blocked(d: Dict[str, Any], lane: Lane, args: argparse.Namespace) -> tuple[bool, str]:
    """V89 minimal guard over the proven recovery-control base.

    Protect the aggressive lane from switching too early in the 27k-30k band
    when signals are only medium. This does not touch metric reading,
    bootstrap, checkpoint, or global progress accounting.
    """
    if lane.name != "aggressive_seq256_zero0_gacc4":
        return False, "not_aggressive_lane"
    tokens = float(d.get("tokens") or 0.0)
    if tokens < args.midband_protect_min_tokens or tokens > args.midband_protect_max_tokens:
        return False, "outside_midband"
    reasons = set(d.get("reasons") or [])
    if "below_lane_min_tokens" in reasons or "optimizer_ratio_force_zero0" in reasons:
        return False, "strong_escape_reason_present"
    opt = float(d.get("optimizer_ratio") or 0.0)
    bad_count = int(d.get("bad_count") or 0)
    if opt < args.midband_required_optimizer_ratio or bad_count < args.midband_required_bad_windows:
        return True, "midband_signals_not_strong_enough"
    return False, "midband_allowed_by_strong_signals"

def productive_lane_guard(d: Dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    """Return True when the lane is useful enough that switching is unnecessary.

    This is the central v89 fix: lane switches are evidence-driven, not step-driven.
    A lane that is producing healthy tokens/steps and has no critical safety issue
    must be preserved even if GPU utilization, optimizer ratio, or score is noisy.
    """
    tokens = float(d.get("tokens") or 0.0)
    steps_s = float(d.get("steps_s") or 0.0)
    age = float(d.get("age") or 0.0)
    opt = float(d.get("optimizer_ratio") or 0.0)
    data_wait = float(d.get("data_wait_ratio") or 0.0)
    chaos = float(d.get("chaos_score") or 0.0)
    reasons = set(d.get("reasons") or [])
    critical = {"progress_stall_critical", "swap_pressure", "oom_or_process_exit", "nonfinite_gradient", "nan_or_inf"}
    if reasons & critical:
        return False, "critical_reason_present"
    if age > getattr(args, "stall_seconds", 120):
        return False, "progress_age_too_old"
    if tokens >= args.productive_min_tokens and steps_s >= args.productive_min_steps_per_second:
        if opt <= args.productive_max_optimizer_ratio and data_wait <= args.productive_max_data_wait_ratio:
            return True, "healthy_throughput"
        if tokens >= args.productive_strong_tokens and steps_s >= args.productive_strong_steps_per_second:
            return True, "strong_throughput_overrides_noisy_ratios"
    if tokens >= args.productive_strong_tokens and steps_s >= args.productive_min_steps_per_second and chaos < args.max_chaos_score:
        return True, "strong_tokens_with_progress"
    return False, "not_productive_enough"


def absolute_health_action(d: Dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    """Classify degradation using absolute health bands before lane switching.

    V89 philosophy: a drop from peak is not automatically a sick lane. On this
    desktop/WSL runtime, 27k-34k tokens/s is usable and often historically good.
    Return values:
      - block: keep current lane; relative degradation is acceptable.
      - refresh: restart the same lane once/twice before cross-lane switching.
      - allow_switch: degradation is severe enough for normal switch logic.
    """
    tokens = float(d.get("tokens") or 0.0)
    steps_s = float(d.get("steps_s") or 0.0)
    opt = float(d.get("optimizer_ratio") or 0.0)
    bad = int(d.get("bad_count") or 0)
    reasons = set(d.get("reasons") or [])
    critical = {"progress_stall_critical", "swap_pressure", "oom_or_process_exit", "nonfinite_gradient", "nan_or_inf"}
    if reasons & critical:
        return "allow_switch", "critical_reason_present"
    if tokens <= 0 or steps_s <= 0:
        return "block", "waiting_real_metrics"
    # Excellent / good: do not switch because of relative peak drop.
    if tokens >= args.health_good_tokens:
        return "block", "absolute_health_good_or_excellent"
    # Acceptable band: hold unless repeated bad windows plus very hot optimizer.
    if tokens >= args.health_acceptable_tokens:
        if bad >= args.health_required_bad_windows and opt >= args.health_attention_optimizer_ratio:
            return "refresh", "acceptable_band_but_hot_optimizer_refresh_same_lane"
        return "block", "absolute_health_acceptable"
    # Attention band: prefer same-lane refresh before cross-lane switching.
    if tokens >= args.health_attention_tokens:
        return "refresh", "attention_band_refresh_same_lane_first"
    # Bad band: normal switch/fallback logic can proceed.
    return "allow_switch", "absolute_health_bad"


def requires_api_approval(reasons: List[str]) -> bool:
    # Critical safety failures can be handled locally. Everything else needs API approval.
    critical = {"progress_stall_critical", "swap_pressure", "oom_or_process_exit", "nonfinite_gradient", "nan_or_inf"}
    return not bool(set(reasons) & critical)


def safe_lane_promotion_triggered(d: Dict[str, Any], lane: Lane, args: argparse.Namespace, current_lane_step: int, current_global_step: int, tokens: float) -> tuple[bool, str]:
    """v89: safe_seq256 is an emergency lane, not a prison.

    If safe is making real progress and the machine has margin, promote back to a
    stronger zero0/gacc lane instead of waiting for long recovery holds.
    """
    if lane.name != "safe_seq256":
        return False, "not_safe_lane"
    if current_lane_step < args.safe_promotion_min_lane_step:
        return False, "safe_warmup_hold"
    opt = float(d.get("optimizer_ratio") or 0.0)
    data_wait = float(d.get("data_wait_ratio") or 0.0)
    gpu_util = float(d.get("gpu_util") or 0.0)
    steps_s = float(d.get("steps_s") or 0.0)
    if tokens >= args.safe_promotion_min_tokens and steps_s >= args.safe_promotion_min_steps_per_second and opt < args.safe_promotion_max_optimizer_ratio and data_wait < args.safe_promotion_max_data_wait_ratio and gpu_util < args.safe_promotion_max_gpu_util:
        return True, "safe_lane_stable_promote_to_fast_zero0_gacc4"
    return False, "safe_lane_not_stable_enough"


def recovery_escape_triggered(d: Dict[str, Any], lane: Lane, args: argparse.Namespace, current_lane_step: int, current_global_step: int, tokens: float, best_tokens: float) -> bool:
    if lane.name not in {"fast_seq256_zero0_gacc4", "safe_seq256", "fast_seq256_zero0_gacc4", "fast_seq256_zero0_gacc4", "fast_seq256_zero0_gacc4", "fast_seq256_zero0_gacc4", "fast_seq256_zero0_gacc4", "fast_seq256_zero0_gacc4", "fast_seq256_zero0_gacc4", "aggressive_seq256_zero0_gacc4", "fast_seq256_zero0_gacc4", "fast_seq256_zero0_gacc4", "fast_seq256_zero0_gacc4"}:
        return False
    if current_global_step < args.recovery_escape_global_step and current_lane_step < args.recovery_escape_lane_step:
        return False
    steps_s = float(d.get("steps_s") or 0.0)
    # v89: gacc2 recovery lanes should not be held for long if they are clearly
    # underperforming; this prevents the v89 14k–18k gacc2 drag.
    if lane.grad_accum <= 2 and (tokens < args.gacc2_escape_tokens or float(d.get("optimizer_ratio") or 0.0) > args.gacc2_escape_optimizer_ratio):
        return True
    if tokens < args.recovery_escape_min_tokens:
        return True
    if steps_s < args.recovery_escape_min_steps_per_second:
        return True
    if best_tokens > 0 and tokens < best_tokens * args.recovery_escape_drop_ratio:
        return True
    return False


def checkpoint_step_from_path(project: Path, checkpoint_path: str) -> int:
    """Return metadata step/global_step for a live checkpoint path.

    Used by recovery resume so the controller's global accounting and LR schedule
    continue from the checkpoint step instead of restarting warmup at zero.
    """
    raw = str(checkpoint_path or "").strip()
    if not raw:
        return 0
    q = Path(raw)
    if not q.is_absolute():
        q = project / q
    meta = q.parent / "metadata.json"
    try:
        data = json.loads(meta.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return 0

    best = 0
    def walk(x: Any) -> None:
        nonlocal best
        if isinstance(x, dict):
            for k, v in x.items():
                kl = str(k).lower()
                if kl in {"global_step", "step", "train_step", "global_steps", "steps"}:
                    try:
                        best = max(best, int(float(v)))
                    except Exception:
                        pass
                walk(v)
        elif isinstance(x, list):
            for y in x:
                walk(y)
    walk(data)
    return max(0, int(best))

def main() -> int:
    ap = argparse.ArgumentParser(description="MEM v89 input-pipeline + safe-lane recovery sustained supervisor")
    ap.add_argument("--start-lane", default="fast_seq256_zero0_gacc4", choices=list(LANES))
    ap.add_argument("--target-steps", type=int, default=300000)
    # v89 long-run mode: no hard max_restarts cap. This argument is accepted
    # only for backward compatibility and is intentionally ignored. Long runs
    # continue until target_global_steps or a critical safety condition.
    ap.add_argument("--max-restarts", type=int, default=0, help="ignored for v89 long runs")
    ap.add_argument("--sample-seconds", type=int, default=20)
    ap.add_argument("--min-degrade-step", type=int, default=1500)
    ap.add_argument("--min-lane-steps-before-switch", type=int, default=int(os.environ.get("MEM_V89_MIN_LANE_STEPS_BEFORE_SWITCH", "50000")))
    ap.add_argument("--min-recovery-hold-steps", type=int, default=11000)
    ap.add_argument("--recovery-escape-global-step", type=int, default=18000)
    ap.add_argument("--recovery-escape-lane-step", type=int, default=12000)
    ap.add_argument("--recovery-escape-min-tokens", type=float, default=11000.0)
    ap.add_argument("--recovery-escape-min-steps-per-second", type=float, default=5.0)
    ap.add_argument("--recovery-escape-drop-ratio", type=float, default=0.60)
    ap.add_argument("--seq512-cooldown-steps", type=int, default=30000)
    ap.add_argument("--recovery-cooldown-steps", type=int, default=26000)
    ap.add_argument("--drop-ratio", type=float, default=0.70)
    ap.add_argument("--bad-windows", type=int, default=2)
    ap.add_argument("--midband-protect-min-tokens", type=float, default=18000)
    ap.add_argument("--midband-protect-max-tokens", type=float, default=23000)
    ap.add_argument("--midband-required-bad-windows", type=int, default=4)
    ap.add_argument("--midband-required-optimizer-ratio", type=float, default=0.45)
    # V89 absolute health bands: relative peak drop is lower priority than absolute usefulness.
    ap.add_argument("--health-good-tokens", type=float, default=19000)
    ap.add_argument("--health-acceptable-tokens", type=float, default=17500)
    ap.add_argument("--health-attention-tokens", type=float, default=16000)
    ap.add_argument("--health-attention-optimizer-ratio", type=float, default=0.45)
    ap.add_argument("--health-required-bad-windows", type=int, default=4)
    ap.add_argument("--same-lane-refresh-max-attempts", type=int, default=2)
    ap.add_argument("--same-lane-refresh-eval-steps", type=int, default=5000)
    ap.add_argument("--productive-min-tokens", type=float, default=18000)
    ap.add_argument("--productive-min-steps-per-second", type=float, default=12.0)
    ap.add_argument("--productive-strong-tokens", type=float, default=20500)
    ap.add_argument("--productive-strong-steps-per-second", type=float, default=16.0)
    ap.add_argument("--productive-max-optimizer-ratio", type=float, default=0.44)
    ap.add_argument("--productive-max-data-wait-ratio", type=float, default=0.32)
    ap.add_argument("--stall-seconds", type=int, default=90)
    ap.add_argument("--hard-stall-seconds", type=int, default=180)
    ap.add_argument("--fresh-progress-grace-seconds", type=int, default=180)
    ap.add_argument("--first-progress-hard-seconds", type=int, default=420)
    ap.add_argument("--first-progress-min-step", type=int, default=1)
    ap.add_argument("--min-gpu-util", type=float, default=40.0)
    ap.add_argument("--min-steps-per-second", type=float, default=4.5)
    ap.add_argument("--max-swap-mib", type=float, default=128.0)
    ap.add_argument("--high-vram-mib", type=float, default=7400.0)
    ap.add_argument("--max-optimizer-ratio", type=float, default=0.36)
    ap.add_argument("--force-zero0-optimizer-ratio", type=float, default=0.40)
    ap.add_argument("--gacc2-escape-tokens", type=float, default=18000.0)
    ap.add_argument("--gacc2-escape-optimizer-ratio", type=float, default=0.42)
    ap.add_argument("--gacc-le3-escape-optimizer-ratio", type=float, default=0.39)
    ap.add_argument("--proactive-min-tokens", type=float, default=17000)
    ap.add_argument("--slow-degradation-ratio", type=float, default=0.91)
    ap.add_argument("--slow-degradation-optimizer-ratio", type=float, default=0.30)
    ap.add_argument("--primary-hold-min-steps", type=int, default=15000)
    ap.add_argument("--primary-min-tokens", type=float, default=18000)
    ap.add_argument("--primary-max-optimizer-ratio", type=float, default=0.38)
    ap.add_argument("--primary-min-gpu-util", type=float, default=35.0)
    ap.add_argument("--max-chaos-score", type=float, default=82.0)
    ap.add_argument("--safe-promotion-min-lane-step", type=int, default=1500)
    ap.add_argument("--safe-promotion-min-tokens", type=float, default=17000)
    ap.add_argument("--safe-promotion-min-steps-per-second", type=float, default=7.0)
    ap.add_argument("--safe-promotion-max-optimizer-ratio", type=float, default=0.38)
    ap.add_argument("--safe-promotion-max-data-wait-ratio", type=float, default=0.25)
    ap.add_argument("--safe-promotion-max-gpu-util", type=float, default=65.0)
    ap.add_argument("--max-host-cpu-percent", type=float, default=96.0)
    ap.add_argument("--loss-plateau-delta", type=float, default=0.030)
    ap.add_argument("--data-wait-ratio-high", type=float, default=0.18)
    ap.add_argument("--data-wait-ratio-severe", type=float, default=0.35)
    ap.add_argument("--dynamic-gacc-optimizer-ratio", type=float, default=0.37)
    ap.add_argument("--weak-lane-cooldown-steps", type=int, default=42000)
    # v89 hard-degradation override: min_lane_steps/min_recovery_hold are not sovereign
    # once a lane is objectively bad. Healthy primary/fallback lanes still get long windows.
    ap.add_argument("--hard-override-min-step", type=int, default=2500)
    ap.add_argument("--hard-override-tokens", type=float, default=16000.0)
    ap.add_argument("--hard-override-optimizer-ratio", type=float, default=0.45)
    ap.add_argument("--hard-override-score", type=float, default=-0.50)
    ap.add_argument("--hard-override-steps-per-second", type=float, default=7.0)
    ap.add_argument("--hard-override-gpu-util", type=float, default=35.0)
    ap.add_argument("--primary-hard-override-tokens", type=float, default=15000)
    ap.add_argument("--primary-hard-override-optimizer-ratio", type=float, default=0.43)
    # MEM v3: do not let API keep safe_seq256 when it is clearly dragging metrics.
    ap.add_argument("--safe-hard-escape-bad-windows", type=int, default=1)
    ap.add_argument("--safe-hard-escape-tokens", type=float, default=25000.0)
    ap.add_argument("--safe-hard-escape-attention-tokens", type=float, default=27000.0)
    ap.add_argument("--safe-hard-escape-optimizer-ratio", type=float, default=0.35)
    ap.add_argument("--safe-hard-escape-hot-optimizer-ratio", type=float, default=0.40)
    ap.add_argument("--safe-hard-escape-steps-per-second", type=float, default=10.0)
    # MEM v3.2.4 throughput-first override: API remains consultive, local policy is sovereign.
    ap.add_argument("--local-throughput-override-bad-windows", type=int, default=2)
    ap.add_argument("--local-throughput-override-tokens", type=float, default=15000)
    ap.add_argument("--local-throughput-override-optimizer-ratio", type=float, default=0.37)
    ap.add_argument("--local-throughput-override-steps-per-second", type=float, default=10.0)
    ap.add_argument("--local-throughput-override-gpu-util", type=float, default=55.0)
    # MEM v3.2.1 anti-churn: safe_seq256 is recovery-only.
    # It must not refresh itself repeatedly, and after leaving safe it is
    # temporarily banned so the controller does not bounce back into it.
    ap.add_argument("--safe-max-lane-steps", type=int, default=5000)
    # MEM v3.3: safe_seq256 is recovery-only, not a sustained training lane.
    # These thresholds bypass the generic warmup/bad-window path when safe is
    # alive but clearly unproductive under high VRAM/GPU pressure.
    ap.add_argument("--safe-recovery-min-lane-step", type=int, default=500)
    ap.add_argument("--safe-recovery-exit-tokens", type=float, default=17000)
    ap.add_argument("--safe-recovery-exit-steps-per-second", type=float, default=3.5)
    ap.add_argument("--safe-recovery-exit-target", default="aggressive_seq256_zero0_gacc4", choices=list(LANES))
    ap.add_argument("--safe-hard-escape-ban-safe-steps", type=int, default=50000)
    ap.add_argument("--allow-safe-same-lane-refresh", action="store_true")
    ap.add_argument("--no-chaos", action="store_true")
    args = ap.parse_args()

    project = Path.cwd()
    (project / "logs").mkdir(exist_ok=True)
    (project / "evidence").mkdir(exist_ok=True)
    (project / "evidence_packets").mkdir(exist_ok=True)
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. v89 keeps API candidate mandatory for MEM sustained control.", file=sys.stderr)
        return 2

    allowed = DEFAULT_ORDER[:]
    lane_name = args.start_lane
    restarts = 0
    global_completed_steps = 0

    controller_state = load_controller_state(project)
    if controller_state:
        restored_global = int(controller_state.get("global_completed_steps") or controller_state.get("current_global_step") or 0)
        if restored_global > 0:
            global_completed_steps = restored_global
            event_log(project, {
                "event": "controller_global_step_restored",
                "global_step": global_completed_steps,
                "source": str(controller_state_path(project)),
                "controller_state": controller_state,
            })
    # Recovery resume bootstrap:
    # If a checkpoint is provided/auto-selected but controller state was cleaned,
    # restore global_completed_steps from checkpoint metadata. This keeps
    # MEM_V89_GLOBAL_STEP_START and the LR schedule aligned with the loaded weights.
    if global_completed_steps <= 0:
        recovery_ckpt = os.environ.get("MEM_DEEPSPEED_LOAD_CHECKPOINT", "").strip()
        if not recovery_ckpt and os.environ.get("MEM_AUTO_RESUME_CHECKPOINT", "1") == "1":
            latest = latest_checkpoint_for_resume(project)
            if latest is not None:
                recovery_ckpt = str(latest)
                os.environ["MEM_DEEPSPEED_LOAD_CHECKPOINT"] = recovery_ckpt
        recovery_step = checkpoint_step_from_path(project, recovery_ckpt)
        if recovery_step > 0:
            global_completed_steps = int(recovery_step)
            event_log(project, {
                "event": "controller_global_step_bootstrapped_from_checkpoint",
                "global_step": global_completed_steps,
                "checkpoint_path": recovery_ckpt,
                "reason": "controller_state_missing_or_cleaned",
            })
            save_controller_state(project, {
                "global_completed_steps": int(global_completed_steps),
                "current_global_step": int(global_completed_steps),
                "lane_step": 0,
                "lane": lane_name,
                "restart_index": int(restarts),
                "target_steps": int(args.target_steps),
                "source": "checkpoint_metadata_bootstrap",
            })

    lane_banned_until: Dict[str, int] = {}
    lane_history: List[Dict[str, Any]] = []
    same_lane_refresh_attempts: Dict[str, int] = {}
    # v89: no hard restart ceiling for long runs. The supervisor continues
    # until target_global_steps is reached, or until a critical safety stop is
    # explicitly accepted. This avoids early run termination when
    # the system is still healthy but needs multiple lane rotations.
    while global_completed_steps < args.target_steps:
        lane = LANES[lane_name]
        lane_start_global = global_completed_steps
        remaining_steps = max(1, args.target_steps - global_completed_steps)
        cleanup_project_processes(project, aggressive_python_dash=True)
        clear_runtime_progress(project)
        lane_start_ts = now()
        env = os.environ.copy()
        env.setdefault("MASTER_PORT", str(30000 + restarts))
        log_path = project / "logs" / f"v89_sustained_{lane.name}_r{restarts}.log"
        event_log(project, {"event": "lane_start", "lane": asdict(lane), "restart_index": restarts, "log": str(log_path), "fresh_progress_after_ts": lane_start_ts, "global_step_start": lane_start_global, "remaining_global_steps": remaining_steps, "banned_until": lane_banned_until})
        write_controller_status(project, state="STARTING", lane=lane, reason="lane_start_waiting_first_progress", restart_index=restarts, global_step=global_completed_steps, target_steps=args.target_steps)
        mark_global_progress_transition(project, state="STARTING", reason="lane_start_waiting_first_progress", lane=lane, global_step=global_completed_steps, target_steps=args.target_steps, restart_index=restarts)
        env["MEM_V89_GLOBAL_STEP_START"] = str(global_completed_steps)
        env["MEM_V89_TARGET_STEPS"] = str(args.target_steps)
        env["MEM_V89_LANE_NAME"] = str(lane.name)
        env.setdefault("MEM_V89_LR_SCHEDULE", "1")
        env.setdefault("MEM_V89_BASE_LR_REAL", "1.2e-5")
        env.setdefault("MEM_V89_LR_WARMUP_STEPS", "10000")
        env.setdefault("MEM_V89_LR_MIN_MULT", "0.05")
        chaos_proc = None
        if not args.no_chaos and (project / "scripts" / "start_real_chaos_hard_sidecar.sh").exists():
            chaos_proc = subprocess.Popen(["bash", "scripts/start_real_chaos_hard_sidecar.sh"], cwd=str(project), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
            time.sleep(5)
        with log_path.open("w", encoding="utf-8") as log:
            proc = run(build_command(project, lane, remaining_steps, 30000 + restarts), cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT)
            (project / "logs" / "v89_sustained_controller_current.pid").write_text(str(proc.pid), encoding="utf-8")
            write_controller_status(project, state="STARTING", lane=lane, reason="child_process_started_waiting_first_progress", restart_index=restarts, global_step=global_completed_steps, target_steps=args.target_steps, lane_pid=proc.pid)
            best_tokens = 0.0
            bad = 0
            last_step = 0
            last_controller_heartbeat_ts = 0.0
            first_progress_observed = False
            while proc.poll() is None:
                time.sleep(args.sample_seconds)
                progress = latest_progress(project, after_ts=lane_start_ts)
                metrics = read_json(progress) if progress else {}
                gpu = sample_gpu()
                swap = sample_swap()
                real_progress = has_real_progress(metrics, min_step=args.first_progress_min_step)
                if real_progress:
                    first_progress_observed = True
                    current_runtime_step_for_status = int(metrics.get("step") or 0)
                    write_controller_status(project, state="RUNNING", lane=lane, reason="real_progress_observed", restart_index=restarts, global_step=global_completed_steps + current_runtime_step_for_status, target_steps=args.target_steps, lane_pid=proc.pid, progress_age=0.0)
                    # P0.2: lightweight controller heartbeat. This proves the
                    # supervisor loop is alive even when no lane change is taken.
                    # It prevents the monitor from looking frozen while runtime
                    # progress/checkpoints continue normally.
                    if now() - last_controller_heartbeat_ts >= 60:
                        last_controller_heartbeat_ts = now()
                        event_log(project, {
                            "event": "controller_heartbeat",
                            "lane": lane.name,
                            "restart_index": restarts,
                            "global_step": global_completed_steps + current_runtime_step_for_status,
                            "runtime_step": current_runtime_step_for_status,
                            "tokens_per_second": metrics.get("tokens_per_second"),
                            "steps_per_second": metrics.get("steps_per_second"),
                            "loss": metrics.get("loss"),
                        })
                if not first_progress_observed:
                    age_since_lane_start = now() - lane_start_ts
                    # v89 FIRST-PROGRESS LOCK: never call decide(), never switch,
                    # and never pack degradation evidence before at least one real
                    # progress heartbeat exists. GPU/VRAM activity during this period
                    # means the child is likely still loading dataset/model or warming
                    # DeepSpeed, not that the lane is degraded.
                    write_controller_status(project, state="STARTING", lane=lane, reason="first_progress_lock_wait", restart_index=restarts, global_step=global_completed_steps, target_steps=args.target_steps, lane_pid=proc.pid, progress_age=age_since_lane_start)
                    event_log(project, {
                        "event": "first_progress_lock_wait",
                        "lane": lane.name,
                        "age_since_lane_start": round(age_since_lane_start, 3),
                        "fresh_progress_grace_seconds": args.fresh_progress_grace_seconds,
                        "first_progress_hard_seconds": args.first_progress_hard_seconds,
                        "has_metrics_file": bool(metrics),
                        "metrics_step": int(metrics.get("step") or 0) if metrics else 0,
                        "gpu": gpu,
                        "swap": swap,
                    })
                    if age_since_lane_start < args.first_progress_hard_seconds:
                        continue
                    # Only after the hard timeout do we treat this as a startup failure.
                    # This is not a performance degradation and must not be classified
                    # as slow throughput loss. Preserve the log and try the next safe
                    # startup lane.
                    event_log(project, {
                        "event": "first_progress_lock_timeout",
                        "lane": lane.name,
                        "age_since_lane_start": round(age_since_lane_start, 3),
                        "returncode": proc.poll(),
                        "log_tail": tail_file(log_path),
                        "gpu": gpu,
                        "swap": swap,
                    })
                    pack = pack_evidence(project, lane, "no_first_progress")
                    event_log(project, {"event": "evidence_packed", "path": pack, "reason": "no_first_progress"})
                    terminate(proc)
                    cleanup_project_processes(project, aggressive_python_dash=True)
                    active_allowed = filter_allowed_lanes(allowed, global_completed_steps, lane_banned_until)
                    if lane.name in active_allowed and len(active_allowed) > 1:
                        active_allowed = [x for x in active_allowed if x != lane.name]
                    lane_name = enforce_lane_cooldown("aggressive_seq256_zero0_gacc4", "fast_seq256_zero0_gacc4", active_allowed)
                    restarts += 1
                    break
                tokens = float(metrics.get("tokens_per_second") or 0.0)
                best_tokens = max(best_tokens, tokens)
                d = decide(metrics, gpu, swap, lane, best_tokens, bad, args, lane_start_ts)
                if not hasattr(main, "_loss_history"):
                    main._loss_history = []  # type: ignore[attr-defined]
                try:
                    loss_val = float(metrics.get("loss"))
                    main._loss_history.append(loss_val)  # type: ignore[attr-defined]
                    main._loss_history = main._loss_history[-8:]  # type: ignore[attr-defined]
                    hist = list(main._loss_history)  # type: ignore[attr-defined]
                    if len(hist) >= 5 and abs(hist[-1] - hist[-5]) < args.loss_plateau_delta:
                        d.setdefault("reasons", []).append("loss_plateau_observed")
                        d["loss_plateau_delta_5"] = round(hist[-1] - hist[-5], 6)
                except Exception:
                    pass
                bad = int(d["bad_count"])
                write_global_progress(project, lane, metrics, global_completed_steps, lane_start_global, restarts, args.target_steps, d)
                write_controller_status(project, state="RUNNING" if not d.get("reasons") else "DEGRADED_OBSERVED", lane=lane, reason=",".join(d.get("reasons") or ["healthy"]), restart_index=restarts, global_step=global_completed_steps + int(metrics.get("step") or 0), target_steps=args.target_steps, lane_pid=proc.pid, progress_age=float(d.get("age") or 0.0), extra={"tokens_per_second": d.get("tokens"), "steps_per_second": d.get("steps_s"), "optimizer_ratio": d.get("optimizer_ratio"), "data_wait_ratio": d.get("data_wait_ratio"), "real_chaos_score": d.get("chaos_score"), "sustained_efficiency_score": d.get("sustained_efficiency_score")})
                step = int(metrics.get("step") or 0)
                if step != last_step or bad:
                    event_log(project, {"event": "sample", "lane": lane.name, "decision_state": d, "best_tokens": best_tokens, "metrics": metrics, "gpu": gpu, "swap": swap})
                    last_step = step
                # MEM v3.3 recovery-only safe lane:
                # safe_seq256 may be operationally alive while producing unusable
                # throughput. Do not wait for the generic min_degrade_step or for
                # bad_windows to grow naturally; once safe has a small proof window,
                # treat low tokens/s or low steps/s as THROUGHPUT_DEGRADED and
                # force the normal switch path toward the configured target.
                if lane.name == "safe_seq256" and step >= int(args.safe_recovery_min_lane_step) and tokens > 0:
                    safe_steps_s = float(d.get("steps_s") or 0.0)
                    safe_drag = tokens < float(args.safe_recovery_exit_tokens) or (safe_steps_s > 0 and safe_steps_s < float(args.safe_recovery_exit_steps_per_second))
                    if safe_drag:
                        d.setdefault("reasons", [])
                        for reason in ("safe_seq256_recovery_only_metric_drag", "throughput_degraded", "safe_seq256_recovery_exit_ready"):
                            if reason not in d["reasons"]:
                                d["reasons"].append(reason)
                        bad = max(int(bad or 0), int(args.bad_windows))
                        d["bad_count"] = max(int(d.get("bad_count") or 0), int(bad))
                        d["controller_bad_count"] = int(bad)
                        event_log(project, {
                            "event": "safe_seq256_recovery_metric_drag_armed",
                            "lane": lane.name,
                            "target_lane": args.safe_recovery_exit_target,
                            "lane_step": step,
                            "global_step": global_completed_steps + step,
                            "tokens": tokens,
                            "steps_s": safe_steps_s,
                            "optimizer_ratio": d.get("optimizer_ratio"),
                            "gpu_util": d.get("gpu_util"),
                            "vram_mib": d.get("gpu_mem"),
                            "bad_windows": d.get("bad_count"),
                            "reason": "safe_alive_but_throughput_degraded",
                        })
                # MEM v3.2.1 anti-churn: safe_seq256 is a temporary recovery lane,
                # not a long-lived performance lane. Once it has produced a real
                # proof window, force normal switch handling toward fast_seq256.
                if lane.name == "safe_seq256" and step >= int(args.safe_max_lane_steps) and tokens > 0:
                    d.setdefault("reasons", [])
                    if "safe_seq256_max_dwell_exit_ready" not in d["reasons"]:
                        d["reasons"].append("safe_seq256_max_dwell_exit_ready")
                    bad = max(int(bad or 0), int(args.bad_windows))
                    d["bad_count"] = max(int(d.get("bad_count") or 0), int(bad))
                    d["controller_bad_count"] = int(bad)
                    event_log(project, {
                        "event": "safe_seq256_max_dwell_exit_armed",
                        "lane": lane.name,
                        "lane_step": step,
                        "global_step": global_completed_steps + step,
                        "tokens": tokens,
                        "steps_s": d.get("steps_s"),
                        "max_lane_steps": args.safe_max_lane_steps,
                        "target_lane": args.safe_recovery_exit_target,
                    })
                if bad >= args.bad_windows:
                    reasons = list(d.get("reasons") or [])
                    critical = is_critical_degradation(reasons)
                    current_lane_step = int(d.get("step") or 0)
                    current_global_step = global_completed_steps + current_lane_step
                    save_controller_state(project, {
                        "global_completed_steps": int(current_global_step),
                        "current_global_step": int(current_global_step),
                        "lane_step": int(current_lane_step),
                        "lane": lane.name,
                        "restart_index": int(restarts),
                        "target_steps": int(args.target_steps),
                    })
                    productive, productive_reason = productive_lane_guard(d, args)
                    if productive and not critical:
                        event_log(project, {"event": "productive_lane_switch_blocked", "lane": lane.name, "lane_step": current_lane_step, "global_step": current_global_step, "tokens": tokens, "steps_s": d.get("steps_s"), "optimizer_ratio": d.get("optimizer_ratio"), "data_wait_ratio": d.get("data_wait_ratio"), "gpu_util": d.get("gpu_util"), "reasons": reasons, "productive_reason": productive_reason})
                        bad = 0
                        continue
                    hard_override, hard_override_reason = hard_degradation_override(d, lane, args, current_lane_step, tokens, best_tokens)
                    # MEM v3 absolute safe_seq256 escape: arm hard_override before
                    # v89 health/midband/min-hold gates, so the 25k floor cannot be
                    # blocked by conservative recovery protection. The actual switch
                    # is still executed through the normal lane-switch path below.
                    d["bad_count"] = max(int(d.get("bad_count") or 0), int(bad or 0))
                    d["controller_bad_count"] = int(bad or 0)
                    safe_abs_escape_armed, safe_abs_escape_reason = safe_seq256_hard_escape_override(
                        d, "fast_seq256_zero0_gacc4", lane, ["fast_seq256_zero0_gacc4", "aggressive_seq256_zero0_gacc4", "safe_seq256"], args
                    )
                    if safe_abs_escape_armed:
                        hard_override = True
                        hard_override_reason = safe_abs_escape_reason
                        reasons = list(dict.fromkeys(reasons + ["safe_seq256_absolute_escape_ready", safe_abs_escape_reason]))
                        event_log(project, {"event": "safe_seq256_absolute_escape_armed", "lane": lane.name, "target_lane": args.safe_recovery_exit_target, "lane_step": current_lane_step, "global_step": current_global_step, "tokens": tokens, "steps_s": d.get("steps_s"), "optimizer_ratio": d.get("optimizer_ratio"), "bad_windows": d.get("bad_count"), "reasons": reasons, "override_reason": safe_abs_escape_reason})
                    if hard_override:
                        event_log(project, {"event": "hard_degradation_override", "lane": lane.name, "lane_step": current_lane_step, "global_step": current_global_step, "tokens": tokens, "optimizer_ratio": d.get("optimizer_ratio"), "sustained_efficiency_score": d.get("sustained_efficiency_score"), "reason": hard_override_reason, "reasons": reasons})
                    safe_promote, safe_promote_reason = safe_lane_promotion_triggered(d, lane, args, current_lane_step, current_global_step, tokens)
                    if safe_promote:
                        hard_override = True
                        hard_override_reason = safe_promote_reason
                        reasons = list(dict.fromkeys(reasons + ["safe_lane_promotion_ready"]))
                        event_log(project, {"event": "safe_lane_promotion_ready", "lane": lane.name, "lane_step": current_lane_step, "global_step": current_global_step, "tokens": tokens, "steps_s": d.get("steps_s"), "optimizer_ratio": d.get("optimizer_ratio"), "data_wait_ratio": d.get("data_wait_ratio"), "gpu_util": d.get("gpu_util"), "reason": safe_promote_reason})
                    critical_or_hard = critical or hard_override
                    # MEM v3.2.3 deterministic local policy: performance-triggered
                    # same-lane refresh is not allowed to beat a clear fallback switch.
                    # Refresh remains reserved for real operational failures; optimizer
                    # pressure + sustained degradation must rotate lanes predictably.
                    local_forced_fallback_switch = False
                    local_forced_fallback_reason = ""

                    # V89: absolute health bands are sovereign over relative peak drop.
                    # If the lane is still useful, do not switch. If it is moderately
                    # degraded, refresh the same lane before trying another lane.
                    health_action, health_reason = absolute_health_action(d, args)
                    if lane.name == "safe_seq256" and current_lane_step >= int(args.safe_max_lane_steps) and tokens > 0 and not critical:
                        hard_override = True
                        hard_override_reason = "safe_seq256_max_dwell_force_fast"
                        critical_or_hard = True
                        health_action = "allow_switch"
                        reasons = list(dict.fromkeys(reasons + ["safe_seq256_max_dwell_exit_ready", hard_override_reason]))
                        event_log(project, {
                            "event": "safe_seq256_max_dwell_force_fast",
                            "lane": lane.name,
                            "target_lane": args.safe_recovery_exit_target,
                            "lane_step": current_lane_step,
                            "global_step": current_global_step,
                            "tokens": tokens,
                            "steps_s": d.get("steps_s"),
                            "optimizer_ratio": d.get("optimizer_ratio"),
                            "max_lane_steps": args.safe_max_lane_steps,
                            "health_reason": health_reason,
                            "reasons": reasons,
                        })
                    strong_switch_reasons = set(reasons)
                    if (
                        not critical_or_hard
                        and lane.name != "safe_seq256"
                        and health_action == "refresh"
                        and "optimizer_ratio_force_zero0" in strong_switch_reasons
                        and ("sustained_throughput_drop" in strong_switch_reasons or "slow_degradation" in strong_switch_reasons)
                    ):
                        local_forced_fallback_switch = True
                        local_forced_fallback_reason = "strong_degradation_bypasses_same_lane_refresh"
                        critical_or_hard = True
                        health_action = "allow_switch"
                        reasons = list(dict.fromkeys(reasons + [local_forced_fallback_reason]))
                        event_log(project, {
                            "event": "v89_same_lane_refresh_bypassed_for_fallback_switch",
                            "lane": lane.name,
                            "lane_step": current_lane_step,
                            "global_step": current_global_step,
                            "tokens": tokens,
                            "steps_s": d.get("steps_s"),
                            "optimizer_ratio": d.get("optimizer_ratio"),
                            "bad_windows": d.get("bad_count"),
                            "health_reason": health_reason,
                            "reasons": reasons,
                            "forced_reason": local_forced_fallback_reason,
                        })
                    if lane.name == "safe_seq256" and not args.allow_safe_same_lane_refresh and not critical_or_hard and health_action == "refresh":
                        hard_override = True
                        hard_override_reason = "safe_seq256_same_lane_refresh_blocked_force_fast"
                        critical_or_hard = True
                        health_action = "allow_switch"
                        reasons = list(dict.fromkeys(reasons + ["safe_seq256_absolute_escape_ready", hard_override_reason]))
                        event_log(project, {
                            "event": "safe_seq256_same_lane_refresh_blocked_force_fast",
                            "lane": lane.name,
                            "target_lane": args.safe_recovery_exit_target,
                            "lane_step": current_lane_step,
                            "global_step": current_global_step,
                            "tokens": tokens,
                            "steps_s": d.get("steps_s"),
                            "optimizer_ratio": d.get("optimizer_ratio"),
                            "bad_windows": d.get("bad_count"),
                            "health_reason": health_reason,
                            "reasons": reasons,
                        })
                    if not critical_or_hard and health_action == "block":
                        event_log(project, {
                            "event": "v89_absolute_health_switch_blocked",
                            "lane": lane.name,
                            "lane_step": current_lane_step,
                            "global_step": current_global_step,
                            "tokens": tokens,
                            "steps_s": d.get("steps_s"),
                            "optimizer_ratio": d.get("optimizer_ratio"),
                            "bad_windows": d.get("bad_count"),
                            "reasons": reasons,
                            "health_reason": health_reason,
                        })
                        bad = 0
                        continue
                    if not critical_or_hard and health_action == "refresh":
                        attempts = int(same_lane_refresh_attempts.get(lane.name, 0))
                        if attempts < args.same_lane_refresh_max_attempts:
                            same_lane_refresh_attempts[lane.name] = attempts + 1
                            event_log(project, {
                                "event": "v89_same_lane_refresh_decision",
                                "lane": lane.name,
                                "lane_step": current_lane_step,
                                "global_step": current_global_step,
                                "tokens": tokens,
                                "steps_s": d.get("steps_s"),
                                "optimizer_ratio": d.get("optimizer_ratio"),
                                "bad_windows": d.get("bad_count"),
                                "attempt": attempts + 1,
                                "max_attempts": args.same_lane_refresh_max_attempts,
                                "health_reason": health_reason,
                                "reasons": reasons,
                            })
                            write_controller_status(project, state="REFRESHING_SAME_LANE", lane=lane, reason="same_lane_refresh:" + health_reason, restart_index=restarts, global_step=current_global_step, target_steps=args.target_steps, lane_pid=proc.pid, progress_age=float(d.get("age") or 0.0), extra={"action": "restart_same_lane"})
                            mark_global_progress_transition(project, state="REFRESHING_SAME_LANE", reason="same_lane_refresh:" + health_reason, lane=lane, global_step=current_global_step, target_steps=args.target_steps, restart_index=restarts)
                            pack = pack_evidence(project, lane, "same_lane_refresh")
                            event_log(project, {"event": "evidence_packed", "path": pack, "reason": "same_lane_refresh"})
                            terminate(proc)
                            cleanup_project_processes(project, aggressive_python_dash=True)
                            completed_delta = max(0, int(d.get("step") or 0))
                            global_completed_steps += completed_delta
                            lane_history.append({"lane": lane.name, "lane_step": completed_delta, "global_step": global_completed_steps, "reasons": reasons, "decision": {"action": "restart_same_lane", "source": "v89_absolute_health_same_lane_refresh", "attempt": attempts + 1}})
                            lane_name = lane.name
                            restarts += 1
                            break
                        else:
                            event_log(project, {
                                "event": "v89_same_lane_refresh_exhausted",
                                "lane": lane.name,
                                "lane_step": current_lane_step,
                                "global_step": current_global_step,
                                "tokens": tokens,
                                "optimizer_ratio": d.get("optimizer_ratio"),
                                "attempts": attempts,
                                "health_reason": health_reason,
                                "reasons": reasons,
                            })

                    midband_blocked, midband_reason = v89_midband_switch_blocked(d, lane, args)
                    if not critical_or_hard and midband_blocked:
                        event_log(project, {
                            "event": "productive_lane_switch_blocked_v89_midband",
                            "lane": lane.name,
                            "lane_step": current_lane_step,
                            "global_step": current_global_step,
                            "tokens": tokens,
                            "best_tokens": best_tokens,
                            "optimizer_ratio": d.get("optimizer_ratio"),
                            "gpu_util": d.get("gpu_util"),
                            "bad_windows": d.get("bad_count"),
                            "reasons": reasons,
                            "block_reason": midband_reason,
                        })
                        bad = 0
                        continue

                    # v89 primary protection: aggressive_seq256_gacc4 was the best historically strong lane.
                    # Do not abandon it just because it is no longer at peak if it still delivers
                    # useful sustained throughput and has not crossed the primary proof window.
                    # Hard-degradation override is the only performance path that can bypass this.
                    if (not critical_or_hard and lane.name == "fast_seq256_zero0_gacc4" and current_lane_step < args.primary_hold_min_steps
                        and tokens >= args.primary_min_tokens
                        and float(d.get("optimizer_ratio") or 0.0) < args.primary_max_optimizer_ratio
                        and float(d.get("gpu_util") or 0.0) >= args.primary_min_gpu_util):
                        event_log(project, {"event": "primary_lane_preserved", "lane": lane.name, "lane_step": current_lane_step, "global_step": current_global_step, "tokens": tokens, "optimizer_ratio": d.get("optimizer_ratio"), "gpu_util": d.get("gpu_util"), "reasons": reasons})
                        bad = 0
                        continue
                    # v89: avoid anxious switching/restart loops. Hold lanes long enough to
                    # make progress unless this is a hard failure/stall/swap situation.
                    if not critical_or_hard and current_lane_step < args.min_lane_steps_before_switch:
                        # v89 HARD GATE: API/LocalPolicy cannot bypass the 10k window
                        # for performance-only degradation. Critical failures and v89 hard-degradation overrides bypass it.
                        event_log(project, {"event": "hard_gate_denied_switch", "lane": lane.name, "lane_step": current_lane_step, "global_step": current_global_step, "min_lane_steps_before_switch": args.min_lane_steps_before_switch, "reasons": reasons})
                        bad = 0
                        continue
                    escape_recovery = recovery_escape_triggered(d, lane, args, current_lane_step, current_global_step, tokens, best_tokens)
                    if escape_recovery:
                        event_log(project, {"event": "recovery_escape_triggered", "lane": lane.name, "lane_step": current_lane_step, "global_step": current_global_step, "tokens": tokens, "steps_s": d.get("steps_s"), "best_tokens": best_tokens, "reasons": reasons})
                    if not critical_or_hard and lane.name in {"fast_seq256_zero0_gacc4", "safe_seq256", "fast_seq256_zero0_gacc4", "fast_seq256_zero0_gacc4", "fast_seq256_zero0_gacc4", "fast_seq256_zero0_gacc4", "fast_seq256_zero0_gacc4", "fast_seq256_zero0_gacc4", "fast_seq256_zero0_gacc4", "aggressive_seq256_zero0_gacc4", "fast_seq256_zero0_gacc4", "fast_seq256_zero0_gacc4", "fast_seq256_zero0_gacc4"} and current_lane_step < args.min_recovery_hold_steps and not escape_recovery:
                        event_log(project, {"event": "hold_recovery_lane", "lane": lane.name, "lane_step": current_lane_step, "global_step": current_global_step, "min_recovery_hold_steps": args.min_recovery_hold_steps, "tokens": tokens, "steps_s": d.get("steps_s"), "best_tokens": best_tokens, "reasons": reasons})
                        bad = 0
                        continue
                    safe_exit_reasons = {"safe_lane_promotion_ready", "safe_seq256_absolute_escape_ready", "safe_seq256_max_dwell_exit_ready", "safe_seq256_recovery_exit_ready"}
                    fallback = str(args.safe_recovery_exit_target) if (lane.name == "safe_seq256" or bool(safe_exit_reasons & set(reasons))) else str(d.get("recommended_fallback") or choose_local_fallback(reasons, lane))
                    active_allowed = filter_allowed_lanes(allowed, current_global_step, lane_banned_until)
                    if lane.name == "safe_seq256" and fallback not in active_allowed:
                        active_allowed = [fallback] + list(active_allowed)
                    # v89: once a lane has crossed its minimum proof window and is
                    # degraded, do not let API simply restart the same poor lane unless
                    # no viable alternative exists. This encourages controlled rotation
                    # across aggressive/zero0/gacc lanes instead of repeated restarts.
                    if not critical and lane.name in active_allowed and len(active_allowed) > 1:
                        active_allowed = [x for x in active_allowed if x != lane.name]
                        event_log(project, {"event": "current_lane_temporarily_removed_from_allowed", "lane": lane.name, "global_step": current_global_step, "reasons": reasons, "allowed": active_allowed})
                    if lane.name == "safe_seq256":
                        fallback = str(args.safe_recovery_exit_target)
                        if fallback not in active_allowed:
                            active_allowed = [fallback] + list(active_allowed)
                    else:
                        fallback = enforce_lane_cooldown(fallback, "fast_seq256_zero0_gacc4", active_allowed)
                    # MEM v3 fix: make the hard-escape override see the real
                    # controller bad-window counter from this loop. This guards
                    # against stale/missing decision-state counters and against
                    # API keep_current_lane cycles that reset pressure.
                    d["bad_count"] = max(int(d.get("bad_count") or 0), int(bad or 0))
                    d["controller_bad_count"] = int(bad or 0)
                    safe_hard_escape, safe_hard_escape_reason = safe_seq256_hard_escape_override(d, fallback, lane, active_allowed, args)
                    if safe_hard_escape:
                        fallback = str(args.safe_recovery_exit_target)
                        if fallback not in active_allowed:
                            active_allowed = [fallback] + list(active_allowed)
                        decision = {
                            "source": "mem_v3_safe_seq256_hard_escape_override",
                            "action": "switch_lane",
                            "lane": fallback,
                            "validated_by_local_policy": True,
                            "api_bypass": True,
                            "override_reason": safe_hard_escape_reason,
                            "local_policy_mode": "mem_v3_safe_seq256_metric_drag_escape",
                        }
                        event_log(project, {
                            "event": "safe_seq256_hard_escape_override",
                            "lane": lane.name,
                            "target_lane": fallback,
                            "lane_step": current_lane_step,
                            "global_step": current_global_step,
                            "tokens": tokens,
                            "steps_s": d.get("steps_s"),
                            "optimizer_ratio": d.get("optimizer_ratio"),
                            "bad_windows": d.get("bad_count"),
                            "reasons": reasons,
                            "override_reason": safe_hard_escape_reason,
                        })
                    else:
                        candidate = api_lane_advisor(metrics, gpu, lane, active_allowed, ";".join(reasons))
                        decision = local_policy_validate(candidate, lane, fallback, active_allowed)
                    mandatory_floor_escape = v89_mandatory_floor_escape(reasons, fallback, lane, active_allowed)
                    throughput_priority_override, throughput_priority_reason = mem_v3_throughput_priority_override(d, reasons, fallback, lane, active_allowed, args)
                    if hard_override and fallback in active_allowed and fallback != lane.name and not critical:
                        throughput_priority_override = True
                        throughput_priority_reason = "hard_degradation_override_requires_fallback:" + str(hard_override_reason)
                    if decision.get("action") == "switch_lane" and decision.get("api_switch_without_usable_target"):
                        event_log(project, {
                            "event": "api_switch_lane_without_target_using_local_fallback",
                            "lane": lane.name,
                            "target_lane": decision.get("lane"),
                            "fallback_lane": fallback,
                            "lane_step": current_lane_step,
                            "global_step": current_global_step,
                            "reasons": reasons,
                            "decision": decision,
                        })
                    if mandatory_floor_escape and decision.get("lane") != fallback:
                        original_decision = dict(decision)
                        decision["action"] = "switch_lane"
                        decision["lane"] = fallback
                        decision["source"] = "v89_local_policy_mandatory_floor_escape"
                        decision["api_decision_overridden"] = True
                        decision["override_reason"] = "mandatory_floor_escape_requires_local_fallback"
                        event_log(project, {
                            "event": "v89_mandatory_floor_escape_forced_local_fallback",
                            "lane": lane.name,
                            "target_lane": fallback,
                            "lane_step": current_lane_step,
                            "global_step": current_global_step,
                            "tokens": tokens,
                            "optimizer_ratio": d.get("optimizer_ratio"),
                            "gpu_util": d.get("gpu_util"),
                            "reasons": reasons,
                            "original_decision": original_decision,
                        })
                    if local_forced_fallback_switch and decision.get("lane") != fallback:
                        original_decision = dict(decision)
                        decision["action"] = "switch_lane"
                        decision["lane"] = fallback
                        decision["source"] = "mem_v3_2_3_local_forced_fallback_switch"
                        decision["api_decision_overridden"] = True
                        decision["override_reason"] = local_forced_fallback_reason
                        event_log(project, {
                            "event": "v89_strong_local_fallback_switch_forced",
                            "lane": lane.name,
                            "target_lane": fallback,
                            "lane_step": current_lane_step,
                            "global_step": current_global_step,
                            "tokens": tokens,
                            "steps_s": d.get("steps_s"),
                            "optimizer_ratio": d.get("optimizer_ratio"),
                            "gpu_util": d.get("gpu_util"),
                            "reasons": reasons,
                            "original_decision": original_decision,
                            "forced_reason": local_forced_fallback_reason,
                        })
                    if throughput_priority_override and decision.get("lane") != fallback:
                        original_decision = dict(decision)
                        decision["action"] = "switch_lane"
                        decision["lane"] = fallback
                        decision["source"] = "mem_v3_2_4_throughput_priority_override"
                        decision["api_decision_overridden"] = True
                        decision["override_reason"] = throughput_priority_reason
                        event_log(project, {
                            "event": "mem_v3_throughput_priority_forced_local_fallback",
                            "lane": lane.name,
                            "target_lane": fallback,
                            "lane_step": current_lane_step,
                            "global_step": current_global_step,
                            "tokens": tokens,
                            "steps_s": d.get("steps_s"),
                            "best_tokens": best_tokens,
                            "optimizer_ratio": d.get("optimizer_ratio"),
                            "gpu_util": d.get("gpu_util"),
                            "bad_windows": d.get("bad_count"),
                            "reasons": reasons,
                            "original_decision": original_decision,
                            "forced_reason": throughput_priority_reason,
                        })
                    if requires_api_approval(reasons) and not safe_hard_escape:
                        if decision.get("source") != "openai_lane_change_approval" and not mandatory_floor_escape and not local_forced_fallback_switch and not throughput_priority_override:
                            event_log(project, {"event": "noncritical_switch_blocked_no_api_approval", "lane": lane.name, "lane_step": current_lane_step, "global_step": current_global_step, "reasons": reasons, "decision": decision})
                            bad = 0
                            continue
                        if decision.get("action") != "switch_lane":
                            if mandatory_floor_escape or local_forced_fallback_switch or throughput_priority_override:
                                original_decision = dict(decision)
                                decision["action"] = "switch_lane"
                                decision["lane"] = fallback
                                if throughput_priority_override:
                                    decision["source"] = "mem_v3_2_4_throughput_priority_after_api"
                                    decision["override_reason"] = throughput_priority_reason
                                else:
                                    decision["source"] = "v89_local_policy_floor_escape_after_api" if mandatory_floor_escape else "mem_v3_2_3_local_forced_fallback_after_api"
                                    decision["override_reason"] = "below_lane_min_tokens_plus_optimizer_pressure" if mandatory_floor_escape else local_forced_fallback_reason
                                decision["api_decision_overridden"] = True
                                event_log(project, {
                                    "event": "mem_v3_api_keep_overridden_by_throughput_priority" if throughput_priority_override else ("v89_api_keep_overridden_by_floor_escape" if mandatory_floor_escape else "v89_api_keep_overridden_by_strong_local_fallback"),
                                    "lane": lane.name,
                                    "target_lane": fallback,
                                    "lane_step": current_lane_step,
                                    "global_step": current_global_step,
                                    "tokens": tokens,
                                    "best_tokens": best_tokens,
                                    "optimizer_ratio": d.get("optimizer_ratio"),
                                    "gpu_util": d.get("gpu_util"),
                                    "reasons": reasons,
                                    "api_decision": original_decision,
                                })
                            else:
                                event_log(project, {"event": "noncritical_switch_blocked_by_api", "lane": lane.name, "lane_step": current_lane_step, "global_step": current_global_step, "reasons": reasons, "decision": decision})
                                bad = 0
                                continue
                    if decision.get("action") in {"keep_current_lane", "restart_same_lane"} and not critical:
                        event_log(project, {"event": "lane_change_not_taken", "lane": lane.name, "lane_step": current_lane_step, "global_step": current_global_step, "reasons": reasons, "decision": decision})
                        bad = 0
                        continue
                    # v89: safe_seq256 is emergency-only. Do not route ordinary throughput
                    # degradation to safe if a stronger gacc4/gacc5 fallback is available.
                    if decision.get("action") == "switch_lane" and decision.get("lane") == "safe_seq256" and not critical and fallback != "safe_seq256":
                        original_decision = dict(decision)
                        decision["lane"] = fallback
                        decision["safe_seq256_denied_for_noncritical_degradation"] = True
                        event_log(project, {"event": "safe_seq256_denied_for_noncritical_degradation", "current_lane": lane.name, "original_decision": original_decision, "new_lane": fallback, "reasons": reasons, "hard_override": hard_override})
                    # v89: API may suggest stop_and_preserve_evidence during ordinary
                    # throughput degradation. LocalPolicy must not end a healthy 200k
                    # run unless the target is reached or a critical failure happened.
                    if decision["action"] == "stop_and_preserve_evidence" and not critical and current_global_step < args.target_steps:
                        original_decision = dict(decision)
                        decision["action"] = "switch_lane"
                        if decision.get("lane") not in active_allowed:
                            decision["lane"] = fallback
                        decision["converted_from_stop_and_preserve_evidence"] = True
                        event_log(project, {"event": "stop_action_converted_to_switch", "current_lane": lane.name, "original_decision": original_decision, "new_decision": decision, "global_step": current_global_step, "target_steps": args.target_steps, "reason": "non_critical_degradation_must_continue"})
                    event_log(project, {"event": "lane_switch_decision", "current_lane": lane.name, "decision": decision, "metrics": metrics, "gpu": gpu, "best_tokens": best_tokens})
                    write_controller_status(project, state="SWITCHING", lane=lane, reason="lane_switch_decision:" + ",".join(reasons), restart_index=restarts, global_step=current_global_step, target_steps=args.target_steps, lane_pid=proc.pid, progress_age=float(d.get("age") or 0.0), extra={"next_lane": decision.get("lane"), "action": decision.get("action")})
                    mark_global_progress_transition(project, state="SWITCHING", reason="lane_switch_decision:" + ",".join(reasons), lane=lane, global_step=current_global_step, target_steps=args.target_steps, restart_index=restarts)
                    pack = pack_evidence(project, lane, "degraded")
                    event_log(project, {"event": "evidence_packed", "path": pack})
                    terminate(proc)
                    cleanup_project_processes(project, aggressive_python_dash=True)
                    completed_delta = max(0, int(d.get("step") or 0))
                    global_completed_steps += completed_delta
                    lane_history.append({"lane": lane.name, "lane_step": completed_delta, "global_step": global_completed_steps, "reasons": reasons, "decision": decision})
                    if lane.name == "safe_seq256" and decision.get("action") == "switch_lane" and decision.get("lane") != "safe_seq256":
                        lane_banned_until["safe_seq256"] = global_completed_steps + int(args.safe_hard_escape_ban_safe_steps)
                        event_log(project, {
                            "event": "safe_seq256_banned_after_exit_anti_churn",
                            "lane": lane.name,
                            "new_lane": decision.get("lane"),
                            "until_global_step": lane_banned_until["safe_seq256"],
                            "global_step": global_completed_steps,
                            "ban_steps": args.safe_hard_escape_ban_safe_steps,
                            "reasons": reasons,
                        })
                    if lane.seq >= 512:
                        lane_banned_until[lane.name] = global_completed_steps + args.seq512_cooldown_steps
                        event_log(project, {"event": "lane_cooldown", "lane": lane.name, "until_global_step": lane_banned_until[lane.name], "global_step": global_completed_steps})
                    elif lane.name not in {"safe_seq256"}:
                        weak_lane = lane.grad_accum <= 3 or float(d.get("tokens") or 0.0) < args.proactive_min_tokens or float(d.get("optimizer_ratio") or 0.0) >= args.force_zero0_optimizer_ratio
                        cooldown_steps = args.weak_lane_cooldown_steps if weak_lane else args.recovery_cooldown_steps
                        lane_banned_until[lane.name] = global_completed_steps + cooldown_steps
                        event_log(project, {"event": "lane_cooldown", "lane": lane.name, "until_global_step": lane_banned_until[lane.name], "global_step": global_completed_steps, "cooldown_type": "weak_lane_aggressive" if weak_lane else "recovery_efficiency", "cooldown_steps": cooldown_steps})
                    if global_completed_steps >= args.target_steps:
                        event_log(project, {"event": "target_global_steps_reached", "global_step": global_completed_steps, "target_steps": args.target_steps, "lane_history": lane_history[-10:]}); write_controller_status(project, state="DONE", reason="target_global_steps_reached", restart_index=restarts, global_step=global_completed_steps, target_steps=args.target_steps)
                        if chaos_proc:
                            terminate(chaos_proc, timeout=5)
                        return 0
                    if decision["action"] == "stop_and_preserve_evidence":
                        # v89: this is only allowed after critical failure handling or target completion.
                        event_log(project, {"event": "stop_and_preserve_evidence_accepted", "lane": lane.name, "global_step": global_completed_steps, "target_steps": args.target_steps, "critical": critical, "reasons": reasons})
                        if chaos_proc:
                            terminate(chaos_proc, timeout=5)
                        return 0
                    lane_name = enforce_lane_cooldown(decision["lane"] if decision["action"] == "switch_lane" else lane.name, fallback, filter_allowed_lanes(allowed, global_completed_steps, lane_banned_until))
                    restarts += 1
                    break
            else:
                pass
        if chaos_proc:
            terminate(chaos_proc, timeout=5)
            subprocess.call(["bash", "scripts/stop_real_chaos_hard_sidecar.sh"], cwd=str(project))
        cleanup_project_processes(project, aggressive_python_dash=True)
        if proc.poll() is not None and bad < args.bad_windows:
            # v89: child exit before global target must not silently end the supervisor.
            # Preserve evidence, account for last progress, then continue with a fallback lane.
            progress = latest_progress(project, after_ts=lane_start_ts)
            metrics = read_json(progress) if progress else {}
            completed_delta = max(0, int(metrics.get("step") or 0))
            if completed_delta:
                global_completed_steps += completed_delta
            write_controller_status(project, state="RECOVERING", lane=lane, reason="unexpected_lane_exit_before_target", restart_index=restarts, global_step=global_completed_steps, target_steps=args.target_steps, lane_pid=0, extra={"returncode": proc.returncode})
            mark_global_progress_transition(project, state="RECOVERING", reason="unexpected_lane_exit_before_target", lane=lane, global_step=global_completed_steps, target_steps=args.target_steps, restart_index=restarts)
            event_log(project, {"event": "unexpected_lane_exit_before_target", "lane": lane.name, "returncode": proc.returncode, "global_step": global_completed_steps, "target_steps": args.target_steps, "last_metrics": metrics, "log_tail": tail_file(log_path)})
            if global_completed_steps >= args.target_steps:
                event_log(project, {"event": "target_global_steps_reached", "global_step": global_completed_steps, "target_steps": args.target_steps, "lane_history": lane_history[-10:]}); write_controller_status(project, state="DONE", reason="target_global_steps_reached", restart_index=restarts, global_step=global_completed_steps, target_steps=args.target_steps)
                return 0
            pack = pack_evidence(project, lane, "unexpected_exit")
            event_log(project, {"event": "evidence_packed", "path": pack})
            if lane.name == "safe_seq256":
                lane_banned_until["safe_seq256"] = global_completed_steps + int(args.safe_hard_escape_ban_safe_steps)
                event_log(project, {
                    "event": "safe_seq256_banned_after_unexpected_exit_anti_churn",
                    "lane": lane.name,
                    "until_global_step": lane_banned_until["safe_seq256"],
                    "global_step": global_completed_steps,
                    "ban_steps": args.safe_hard_escape_ban_safe_steps,
                })
            active_allowed = filter_allowed_lanes(allowed, global_completed_steps, lane_banned_until)
            fallback = enforce_lane_cooldown(choose_local_fallback(["unexpected_lane_exit"], lane), "aggressive_seq256_zero0_gacc4", active_allowed)
            lane_history.append({"lane": lane.name, "lane_step": completed_delta, "global_step": global_completed_steps, "reasons": ["unexpected_lane_exit"], "returncode": proc.returncode})
            lane_name = fallback
            restarts += 1
            continue
    event_log(project, {"event": "controller_loop_completed", "global_step": global_completed_steps, "target_steps": args.target_steps, "restarts": restarts, "lane_history": lane_history[-20:]})
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except KeyboardInterrupt:
        # Manual Ctrl+C / SIGINT is an operator stop, not a controller crash.
        # Record it explicitly so the shell supervisor can tear down any live
        # training lane instead of leaving main.py orphaned or reporting a
        # misleading CONTROLLER_FATAL_ERROR.
        try:
            project = Path.cwd()
            event_log(project, {
                "event": "controller_interrupted",
                "reason": "KeyboardInterrupt",
            })
            write_controller_status(
                project,
                state="CONTROLLER_INTERRUPTED",
                reason="KeyboardInterrupt",
            )
        except Exception:
            pass
        raise SystemExit(130)
    except BaseException as exc:
        try:
            project = Path.cwd()
            import traceback
            event_log(project, {
                "event": "controller_fatal_error",
                "error_type": type(exc).__name__,
                "error": str(exc)[:800],
                "traceback_tail": traceback.format_exc()[-4000:],
            })
            write_controller_status(
                project,
                state="CONTROLLER_FATAL_ERROR",
                reason=f"{type(exc).__name__}:{str(exc)[:240]}",
            )
        except Exception:
            pass
        raise
