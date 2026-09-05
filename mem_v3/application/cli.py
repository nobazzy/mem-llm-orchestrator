from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Configure Windows native SSL certificates & sanitize cert environment
for _ca_env in ("CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
    _val = os.environ.get(_ca_env)
    if _val and not os.path.exists(_val):
        os.environ.pop(_ca_env, None)

try:
    import truststore
    truststore.inject_into_ssl()
    import urllib3.util.ssl_
    urllib3.util.ssl_.create_urllib3_context = truststore.SSLContext
except Exception:
    pass

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from core.orchestrator import MemOrchestrator, OrchestratorContext
from domain.models import CONFIRMATION_TOKEN, RuntimeRequest, VERSION
from infrastructure.logging import configure_logging, print_json
from runtime.adaptive_lane_runner import AdaptiveLaneRunner
from runtime.checkpoint_manager import CheckpointManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MEM Orchestrator v89/v3 — Unified Autonomous Training & Runtime Control")
    
    # Modern Training Controls
    parser.add_argument("--steps", type=int, default=1000000, help="Target total training steps")
    parser.add_argument("--target-steps", type=int, default=1000000, help="Target total training steps")
    parser.add_argument("--start-lane", default="safe_seq256", help="Initial execution lane")
    parser.add_argument("--dataset-name", default="roneneldan/TinyStories", help="Primary dataset name or HuggingFace ID")
    parser.add_argument("--dataset-config", default="", help="HuggingFace dataset config name")
    parser.add_argument("--dataset-fallback-name", default="roneneldan/TinyStories", help="Fallback dataset name")
    parser.add_argument("--dataset-split", default="train", help="Dataset split (train/validation)")
    parser.add_argument("--tokenizer-name", default="gpt2", help="Tokenizer name or model ID")
    parser.add_argument("--sequence-length", type=int, default=256, help="Sequence length in tokens")
    parser.add_argument("--model-preset", default="large_130m", help="Model preset (large_130m, xlarge_250m, medium_75m, tiny_decoder, etc.)")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"], help="Floating point precision")
    parser.add_argument("--resume-checkpoint", default=None, help="Path to checkpoint directory or 'latest'")
    parser.add_argument("--resume-latest", action="store_true", help="Resume automatically from latest checkpoint")
    parser.add_argument("--live-train", action="store_true", help="Run sustained autonomous training loop")
    
    # Dashboard & Tooling Controls
    parser.add_argument("--dashboard", type=int, nargs="?", const=8089, default=None, help="Start web dashboard on port (default 8089)")
    parser.add_argument("--environment-doctor", action="store_true", help="Run environment inspection and diagnostics")
    parser.add_argument("--version", action="store_true", help="Display orchestrator version")
    parser.add_argument("--json-logs", action="store_true", help="Output machine-readable JSON logs")

    # LLM & Policy Governance Controls
    parser.add_argument("--llm", action="store_true", help="Enable OpenAI LLM planner")
    parser.add_argument("--api-executive-moderate", action="store_true", help="Enable AI executive moderation")
    parser.add_argument("--api-executive-mode", dest="api_executive_moderate", action="store_true", help="Alias for --api-executive-moderate")
    parser.add_argument("--operator", action="store_true", help="Operator interactive mode")
    parser.add_argument("--chaos-profile", default="clean", choices=["clean", "real_streaming_mix", "real_multilingual_noise", "real_checkpoint_pressure", "real_desktop_contention"])
    parser.add_argument("--dataset-mix", default="")
    parser.add_argument("--guardrail-mode", default="sampled", choices=["full", "sampled", "minimal"])
    parser.add_argument("--guardrail-sample-interval", type=int, default=8)
    parser.add_argument("--gradient-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adaptive-memory-apply-suggestions", action="store_true")

    # Legacy DeepSpeed Compatibility Flags
    parser.add_argument("--deepspeed-wsl-accelerated", action="store_true")
    parser.add_argument("--deepspeed-aggressive-bounded", action="store_true")
    parser.add_argument("--deepspeed-real-micro-train", action="store_true")
    parser.add_argument("--deepspeed-persistent-checkpoint", action="store_true")
    parser.add_argument("--deepspeed-max-steps", type=int, default=1000)
    parser.add_argument("--deepspeed-batch-size", type=int, default=1)
    parser.add_argument("--deepspeed-zero-stage", type=int, default=0)
    parser.add_argument("--deepspeed-precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--deepspeed-load-checkpoint", default="")
    parser.add_argument("--deepspeed-real-limited-apply", action="store_true")
    parser.add_argument("--deepspeed-gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--real-dataset", action="store_true")
    parser.add_argument("--confirm-deepspeed-accelerated", default="")
    parser.add_argument("--benchmark-mode", default="mem_native_pytorch")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    configure_logging(args.json_logs)

    if args.version:
        print(VERSION)
        return

    orchestrator = MemOrchestrator()
    if args.environment_doctor:
        print_json(orchestrator.context.doctor.inspect().to_dict())
        return

    if args.dashboard is not None:
        from application.dashboard import run_dashboard_server
        port = args.dashboard if isinstance(args.dashboard, int) and args.dashboard > 0 else 8089
        print(f"Starting MEM Orchestrator Dashboard on http://localhost:{port}...")
        run_dashboard_server(port=port)
        return

    # DeepSpeed legacy runner path
    if args.deepspeed_wsl_accelerated:
        req = RuntimeRequest(
            max_steps=args.deepspeed_max_steps,
            batch_size=args.deepspeed_batch_size,
            zero_stage=args.deepspeed_zero_stage,
            precision=args.deepspeed_precision,
            persistent_checkpoint=args.deepspeed_persistent_checkpoint,
            load_checkpoint=args.deepspeed_load_checkpoint,
            confirmation=args.confirm_deepspeed_accelerated,
            llm_enabled=args.llm,
            api_executive_moderate=args.api_executive_moderate,
            operator=args.operator,
            real_micro_train=args.deepspeed_real_micro_train,
            real_limited_apply=args.deepspeed_real_limited_apply,
            gradient_accumulation_steps=args.deepspeed_gradient_accumulation_steps,
            real_dataset=args.real_dataset,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            dataset_fallback_name=args.dataset_fallback_name,
            tokenizer_name=args.tokenizer_name,
            sequence_length=args.sequence_length,
            model_preset=args.model_preset,
            benchmark_mode=args.benchmark_mode,
            chaos_profile=args.chaos_profile,
            dataset_mix=args.dataset_mix,
            guardrail_mode=args.guardrail_mode,
            guardrail_sample_interval=args.guardrail_sample_interval,
            gradient_audit=args.gradient_audit,
            adaptive_memory_apply_suggestions=args.adaptive_memory_apply_suggestions,
        )
        print("+--------------------------------------------------+")
        print("| MEM ORCHESTRATOR v89.0.0                         |")
        print("| Sustained Runtime Control + API Light            |")
        print("| degradation watch | safe lane switching          |")
        print("+--------------------------------------------------+\n")
        print_json(orchestrator.run_deepspeed(req))
        return

    # Modern Autonomous Training Execution Path (AdaptiveLaneRunner)
    import torch
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    effective_precision = args.precision if torch.cuda.is_available() else "fp32"
    resume_target = args.resume_checkpoint or ("latest" if args.resume_latest else None)
    target_steps = args.target_steps or args.steps or 1000000
    api_key = os.environ.get("OPENAI_API_KEY", "")

    print("============================================================")
    print("  MEM ORCHESTRATOR — TREINO REAL COM AUTONOMOUS LANE SWITCHING")
    print(f"  Dispositivo: {device_name} (Aceleração AMP: {effective_precision.upper()})")
    print(f"  Target: {target_steps:,} steps | Start Lane: {args.start_lane}")
    print(f"  Dataset: {args.dataset_name} (Streaming) [Fallback: {args.dataset_fallback_name}]")
    print(f"  Modelo: {args.model_preset} (SDPA Flash Attention)")
    print(f"  Resume Checkpoint: {resume_target if resume_target else 'Não (Início do zero)'}")
    print(f"  OpenAI API Key: {'PRESENTE (' + api_key[:10] + '...)' if api_key else 'AUSENTE'}")
    print("============================================================\n")

    context = OrchestratorContext(_root)
    req = RuntimeRequest(
        max_steps=target_steps,
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
        sequence_length=args.sequence_length,
        model_preset=args.model_preset,
        benchmark_mode="mem_native_pytorch",
        llm_enabled=bool(api_key or args.llm),
        api_executive_moderate=bool(api_key or args.api_executive_moderate),
    ).normalized()

    print("[1/3] Consultando LLMPlanner...")
    plan = context.llm.plan(req)
    print(f"  -> Plano retornado: {plan.source} | Rationale: {plan.rationale[:80]}...")

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
    try:
        (evidence_dir / "api_usage_summary.json").write_text(json.dumps(api_telemetry, indent=2), encoding="utf-8")
        (reports_dir / "api_usage_summary_latest.json").write_text(json.dumps(api_telemetry, indent=2), encoding="utf-8")
    except Exception:
        pass

    print("[3/3] Iniciando AdaptiveLaneRunner com troca autônoma de lanes...")
    ckpt_mgr = CheckpointManager(root=Path(_root) / "checkpoints")
    runner = AdaptiveLaneRunner(
        checkpoint_manager=ckpt_mgr,
        evidence_dir=evidence_dir,
        initial_lane=args.start_lane,
        model_preset=args.model_preset,
    )

    result = runner.train_loop(
        total_steps=target_steps,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        fallback_name=args.dataset_fallback_name,
        model_preset=args.model_preset,
        checkpoint_interval=500,
        eval_window_steps=50,
        resume_from_checkpoint=resume_target,
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
