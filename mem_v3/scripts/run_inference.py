"""
MEM ORCHESTRATOR - Inference & Sampling Utility.
Loads the latest trained checkpoint and generates text autoregressively.
"""

from __future__ import annotations

import argparse
import os
import sys
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

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from runtime.checkpoint_manager import CheckpointManager
from runtime.lm_model import build_tiny_causal_lm


@torch.no_grad()
def sample_generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    top_k: int = 40,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    model.eval()
    curr_ids = prompt_ids
    for _ in range(max_new_tokens):
        seq_limit = getattr(model, "seq_len", 256)
        cond = curr_ids if curr_ids.size(1) <= seq_limit else curr_ids[:, -seq_limit:]
        logits = model(cond)
        logits = logits[:, -1, :].clone()

        if repetition_penalty != 1.0:
            seen_tokens = set(cond[0].tolist())
            for tok_id in seen_tokens:
                if logits[0, tok_id] > 0:
                    logits[0, tok_id] /= repetition_penalty
                else:
                    logits[0, tok_id] *= repetition_penalty

        logits = logits / max(1e-5, temperature)
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        curr_ids = torch.cat([curr_ids, next_tok], dim=1)
    return curr_ids


def main():
    parser = argparse.ArgumentParser(description="MEM Run Inference on Trained Checkpoint")
    parser.add_argument("--prompt", default="", nargs="?", const="", help="Input prompt for text generation (leave empty or omit for raw unconditional generation)")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint (default: latest in checkpoints/)")
    parser.add_argument("--model-preset", default="large_130m", help="Model preset (large_130m, xlarge_250m)")
    parser.add_argument("--max-tokens", type=int, default=60, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K sampling cutoff")
    parser.add_argument("--repetition-penalty", type=float, default=1.2, help="Penalty applied to repeated tokens (default: 1.2)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for inference (default: cpu to avoid VRAM contention with live training)")
    args = parser.parse_args()

    device = torch.device("cuda:0" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    device_desc = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU (Zero-VRAM Contention Mode)"
    print("\n" + "="*70)
    print("  MEM ORCHESTRATOR - INFERENCE & TEXT GENERATION")
    print(f"  Device:             {device_desc}")
    print(f"  Model Preset:       {args.model_preset}")
    print(f"  Prompt:             \"{args.prompt}\"")
    print(f"  Repetition Penalty: {args.repetition_penalty}")
    print("="*70 + "\n")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ckpt_mgr = CheckpointManager(root=Path(_root) / "checkpoints")

    ckpt_path = args.checkpoint
    if not ckpt_path:
        ckpt_path = ckpt_mgr.latest_checkpoint_path()

    if not ckpt_path or not Path(ckpt_path).exists():
        print(f"[ERROR] No checkpoint found at '{ckpt_path}'. Run training first!")
        return

    print(f"Loading checkpoint from: {ckpt_path}")
    payload = ckpt_mgr.load_torch_checkpoint(ckpt_path, map_location=device)
    meta = payload.get("metadata", {})
    step = meta.get("step", "unknown")
    loss = meta.get("loss", "unknown")
    print(f"Checkpoint info: Step {step} | Loss: {loss}\n")

    model = build_tiny_causal_lm(
        vocab_size=len(tokenizer),
        seq_len=256,
        preset=args.model_preset,
    ).to(device)

    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    encoded = tokenizer.encode(args.prompt) if args.prompt else []
    if not encoded:
        bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id or 50256
        encoded = [bos_id]
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    print("Generating text...\n" + "-"*70)
    out_ids = sample_generate(
        model=model,
        prompt_ids=input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    generated_text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
    try:
        print(generated_text)
    except Exception:
        safe_str = generated_text.encode("ascii", errors="replace").decode("ascii")
        print(safe_str)
    print("-"*70 + "\n")


if __name__ == "__main__":
    main()
