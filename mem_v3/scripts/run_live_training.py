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
from runtime.checkpoint_manager import CheckpointManager
from runtime.torch_native_runner import PyTorchNativeRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="MEM Live Training Execution")
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--target-steps", type=int, default=1000000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
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
    print("  MEM ORCHESTRATOR — INICIANDO TREINO REAL NA GPU")
    print(f"  Dispositivo: {device_name} (Precisão: {effective_precision.upper()})")
    print(f"  Target: {args.target_steps:,} steps | Executando lote: {args.steps:,} steps")
    print(f"  Dataset: {args.dataset_name} (Streaming) [Fallback: {args.dataset_fallback_name}]")
    print(f"  Modelo: {args.model_preset} (~72M parâmetros)")
    print(f"  OpenAI API Key: {'PRESENTE (' + api_key[:10] + '...)' if api_key else 'AUSENTE'}")
    print("============================================================\n")

    # Initialize context and AI planner
    context = OrchestratorContext(_root)
    req = RuntimeRequest(
        max_steps=args.target_steps,
        batch_size=args.batch_size,
        zero_stage=0,
        precision=effective_precision,
        persistent_checkpoint=True,
        confirmation=CONFIRMATION_TOKEN,
        operator=True,
        real_micro_train=True,
        real_dataset=True,
        dataset_name=args.dataset_name,
        dataset_fallback_name=args.dataset_fallback_name,
        sequence_length=args.sequence_length,
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

    # Evidence directory for dashboard telemetry
    ts = time.strftime("%Y%m%d_%H%M%S")
    evidence_dir = Path(_root) / "evidence" / f"v89_wsl_deepspeed_{ts}"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(_root) / "reports"
    reports_dir.mkdir(exist_ok=True)

    # Save API telemetry and controller state
    api_telemetry = context.llm.telemetry_summary()
    (evidence_dir / "api_usage_summary.json").write_text(json.dumps(api_telemetry, indent=2), encoding="utf-8")
    (reports_dir / "api_usage_summary_latest.json").write_text(json.dumps(api_telemetry, indent=2), encoding="utf-8")

    controller_status = {
        "timestamp": time.time(),
        "lane": decision.lane,
        "allowed": decision.allowed,
        "plan_source": plan.source,
        "rationale": plan.rationale,
        "executive_action": getattr(directive, "action", "stabilize") if directive else "default",
        "target_steps": args.target_steps,
        "steps_in_run": args.steps,
    }
    (Path(_root) / "evidence" / "v89_controller_status_latest.json").write_text(json.dumps(controller_status, indent=2), encoding="utf-8")

    print("[3/3] Iniciando loop de treinamento Causal LM com telemetria ao vivo...")
    ckpt_mgr = CheckpointManager(root=Path(_root) / "checkpoints")
    runner = PyTorchNativeRunner(checkpoint_manager=ckpt_mgr)

    metrics, checkpoint = runner.run(
        steps=args.steps,
        batch_size=args.batch_size,
        zero_stage=0,
        precision=effective_precision,
        persistent_checkpoint=True,
        applied_hyperparams={"batch_size": args.batch_size, "precision": effective_precision, "gradient_accumulation_steps": 1},
        executive_directives=decision.executive_runtime_directives,
        dataset_settings={
            "real_dataset": True,
            "dataset_name": args.dataset_name,
            "dataset_fallback_name": args.dataset_fallback_name,
            "dataset_split": "train",
            "dataset_streaming": True,
            "tokenizer_name": "gpt2",
            "sequence_length": args.sequence_length,
            "model_preset": args.model_preset,
            "evidence_dir": str(evidence_dir),
            "target_steps": args.target_steps,
            "progress_heartbeat_interval": 5,
            "checkpoint_interval": 500,
        },
    )

    print("\n============================================================")
    print("  TREINO CONCLUÍDO COM SUCESSO!")
    print(f"  Steps completados: {metrics.get('micro_train_steps_completed')}")
    print(f"  Tokens processados: {metrics.get('tokens_processed')}")
    print(f"  Loss inicial: {metrics.get('loss_first')} -> Loss final: {metrics.get('loss_last')}")
    print(f"  Throughput: {metrics.get('tokens_per_second')} tokens/s | {metrics.get('steps_per_second')} steps/s")
    print(f"  Checkpoint gravado: {checkpoint.get('checkpoint_written')} | Path: {checkpoint.get('checkpoint_path')}")
    print("============================================================\n")


if __name__ == "__main__":
    main()
