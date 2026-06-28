from __future__ import annotations

import argparse

from core.orchestrator import MemOrchestrator
from domain.models import VERSION
from infrastructure.logging import configure_logging, print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MEM Orchestrator v89 — Sustained Runtime Control + API Lane Switching")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--api-executive-moderate", action="store_true")
    parser.add_argument("--api-executive-mode", dest="api_executive_moderate", action="store_true", help="Alias for --api-executive-moderate")
    parser.add_argument("--operator", action="store_true")
    parser.add_argument("--json-logs", action="store_true")
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
    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-fallback-name", default="roneneldan/TinyStories")
    parser.add_argument("--tokenizer-name", default="gpt2")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--model-preset", default="tiny_decoder", choices=["tiny_decoder", "small_decoder", "medium_50m",
            "75m_decoder",
            "decoder_75m",
            "medium_75m", "decoder_50m", "50m_decoder", "medium_100m", "decoder_100m", "100m_decoder"])
    parser.add_argument("--benchmark-mode", default="mem_real_chaos", choices=["mem_real_chaos"])
    parser.add_argument("--chaos-profile", default="real_streaming_mix", choices=["clean", "real_streaming_mix", "real_multilingual_noise", "real_checkpoint_pressure", "real_desktop_contention"])
    parser.add_argument("--dataset-mix", default="HuggingFaceFW/fineweb-edu:sample-10BT,roneneldan/TinyStories:")
    parser.add_argument("--guardrail-mode", default="sampled", choices=["full", "sampled", "minimal"])
    parser.add_argument("--guardrail-sample-interval", type=int, default=8)
    parser.add_argument("--gradient-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adaptive-memory-apply-suggestions", action="store_true")
    parser.add_argument("--confirm-deepspeed-accelerated", default="")
    parser.add_argument("--environment-doctor", action="store_true")
    parser.add_argument("--version", action="store_true")
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
    if args.deepspeed_wsl_accelerated:
        from domain.models import RuntimeRequest

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
    build_parser().print_help()
