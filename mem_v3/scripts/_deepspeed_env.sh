#!/usr/bin/env bash
# _deepspeed_env.sh — sourced by all run scripts before launching main.py
# Configures the environment for single-GPU DeepSpeed on WSL2 and bare Linux.

# Disable DeepSpeed JIT/fused ops — required for RTX 50xx (compute 12.0+) and WSL2
export DS_BUILD_OPS=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_AIO=0
export DS_BUILD_UTILS=0
export DS_SKIP_CUDA_CHECK=1

# Single-GPU distributed process group — required even for world_size=1
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export LOCAL_RANK=${LOCAL_RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

# WSL2: NCCL cannot enumerate CUDA devices because it uses a different libcuda path
# than PyTorch. For single-GPU there is no benefit to NCCL — use gloo instead.
# DeepSpeed respects TORCH_DISTRIBUTED_DEFAULT_BACKEND when no backend is forced in config.
if grep -qi microsoft /proc/version 2>/dev/null || [ -f /usr/lib/wsl/lib/libcuda.so.1 ]; then
    export TORCH_DISTRIBUTED_DEFAULT_BACKEND=gloo
    export NCCL_SOCKET_IFNAME=lo
    # Expose WSL libcuda to NCCL/other libs in case they need it
    if [ -d /usr/lib/wsl/lib ] && [[ ":${LD_LIBRARY_PATH:-}:" != *":/usr/lib/wsl/lib:"* ]]; then
        export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
fi
