from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Enable native Windows SSL truststore
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from core.orchestrator import OrchestratorContext
from domain.models import CONFIRMATION_TOKEN, RuntimeRequest
from runtime.adaptive_lane_runner import AdaptiveLaneRunner, STANDARD_LANES
from runtime.checkpoint_manager import CheckpointManager


def main() -> None:
    parser = argparse.ArgumentParser(description="MEM Live Training Execution with Autonomous Lane Control")
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--target-steps", type=int, default=1000000)
    parser.add_argument("--start-lane", default="aggressive_seq256_zero0_gacc4")
    parser.add_argument("--dataset-name", default="DKYoon/SlimPajama-6B")
    parser.add_argument("--dataset-fallback-name", default="roneneldan/TinyStories")
    parser.add_argument("--model-preset", default="medium_75m")
    parser.add_argument("--precision", default="fp16")
    args = parser.parse_args()

    import torch
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    effective_precision = args.precision if torch.cuda.is_available() else "fp32"

    api_key = os.environ.get("OPENAI_API_KEY", "")
    print("============================================================")
    print("  MEM ORCHESTRATOR — TREINO REAL COM AUTONOMOUS LANE SWITCHING")
    print(f"  Dispositivo: {device_name} (Aceleração AMP: {effective_precision.upper()})")
    print(f"  Target: {args.target_steps:,} steps | Start Lane: {args.start_lane}")
    print(f"  Dataset: {args.dataset_name} (Streaming) [Fallback: {args.dataset_fallback_name}]")
    print(f"  Modelo: {args.model_preset} (~72M parâmetros — SDPA Flash Attention)")
    print(f"  OpenAI API Key: {'PRESENTE (' + api_key[:10] + '...)' if api_key else 'AUSENTE'}")
    print("============================================================\n")

    context = OrchestratorContext(_root)
    req = RuntimeRequest(
        max_steps=args.target_steps,
        batch_size=16,
        zero_stage=0,
        precision=effective_precision,
        persistent_checkpoint=True,
        confirmation=CONFIRMATION_TOKEN,
        operator=True,
        real_micro_train=True,
        real_dataset=True,
        dataset_name=args.dataset_name,
        dataset_fallback_name=args.dataset_fallback_name,
        sequence_length=256,
        model_preset=args.model_preset,
        benchmark_mode="mem_native_pytorch",
        llm_enabled=bool(api_key),
        api_executive_moderate=bool(api_key),
    ).normalized()

    print("[1/3] Consultando LLMPlanner (OpenAI GPT-4o)...")
    plan = context.llm.plan(req)
    print(f"  -> Plano retornado pela IA: {plan.source} | Rationale: {plan.rationale[:80]}...")

    print("[2/3] Avaliando plano com LocalPolicyEngine...")
    env_report = context.doctor.inspect()
    directive = context.llm.executive_directive(req, plan, env_report.to_dict()) if req.api_executive_moderate else None
    decision = context.policy.evaluate(req, plan, env_report, directive)
    print(f"  -> Decisão da Política: Allowed={decision.allowed} | Lane: {decision.lane}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    evidence_dir = Path(_root) / "evidence" / f"v89_wsl_deepspeed_{ts}"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(_root) / "reports"
    reports_dir.mkdir(exist_ok=True)

    api_telemetry = context.llm.telemetry_summary()
    (evidence_dir / "api_usage_summary.json").write_text(json.dumps(api_telemetry, indent=2), encoding="utf-8")
    (reports_dir / "api_usage_summary_latest.json").write_text(json.dumps(api_telemetry, indent=2), encoding="utf-8")

    print("[3/3] Iniciando AdaptiveLaneRunner com troca autônoma de lanes...")
    ckpt_mgr = CheckpointManager(root=Path(_root) / "checkpoints")
    runner = AdaptiveLaneRunner(
        checkpoint_manager=ckpt_mgr,
        evidence_dir=evidence_dir,
        initial_lane=args.start_lane,
    )

    result = runner.train_loop(
        total_steps=args.steps,
        dataset_name=args.dataset_name,
        fallback_name=args.dataset_fallback_name,
        model_preset=args.model_preset,
        checkpoint_interval=500,
        eval_window_steps=40,
    )

    print("\n============================================================")
    print("  TREINO CONCLUÍDO COM SUCESSO!")
    print(f"  Steps completados: {result.get('steps_completed')}")
    print(f"  Tokens processados: {result.get('tokens_processed')}")
    print(f"  Loss inicial: {result.get('loss_first')} -> Loss final: {result.get('loss_last')}")
    print(f"  Histórico de lanes: {len(result.get('lane_history', []))} eventos registrados")
    print("============================================================\n")


if __name__ == "__main__":
    main()
