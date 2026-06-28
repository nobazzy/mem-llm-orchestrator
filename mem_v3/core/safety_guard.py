from __future__ import annotations

import importlib
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List

from domain.models import EnvironmentReport


def _run(cmd: list[str], timeout: int = 6) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout, text=True)
        return out.strip()
    except Exception as exc:
        return f"ERROR:{type(exc).__name__}:{exc}"


class EnvironmentDoctor:
    def inspect(self) -> EnvironmentReport:
        recommendations: List[str] = []
        torch_info: Dict[str, Any] = {"import_ok": False}
        ds_info: Dict[str, Any] = {"import_ok": False}
        mpi_info: Dict[str, Any] = {"import_ok": False}
        cuda_home = os.environ.get("CUDA_HOME", "")
        nvcc = shutil.which("nvcc")
        cuda_info: Dict[str, Any] = {
            "CUDA_HOME": cuda_home,
            "CUDA_HOME_exists": bool(cuda_home and os.path.exists(cuda_home)),
            "nvcc_available": bool(nvcc),
            "nvcc_path": nvcc or "",
            "nvcc_output_tail": _run(["bash", "-lc", "which nvcc && nvcc --version | tail -5"]) if nvcc else "",
            "wsl_libcuda_present": os.path.exists("/usr/lib/wsl/lib/libcuda.so.1"),
        }
        try:
            torch = importlib.import_module("torch")
            torch_info.update({
                "import_ok": True,
                "version": getattr(torch, "__version__", "unknown"),
                "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            })
        except Exception as exc:
            torch_info["error"] = repr(exc)
        try:
            ds = importlib.import_module("deepspeed")
            ds_info.update({"import_ok": True, "version": getattr(ds, "__version__", "unknown")})
        except Exception as exc:
            ds_info["error"] = repr(exc)
        try:
            mpi4py = importlib.import_module("mpi4py")
            from mpi4py import MPI  # type: ignore
            mpi_info.update({"import_ok": True, "version": getattr(mpi4py, "__version__", "unknown"), "vendor": MPI.get_vendor()})
        except Exception as exc:
            mpi_info["error"] = repr(exc)
        compat = {
            "python_313_experimental": sys.version_info[:2] >= (3, 13),
            "is_wsl_like": "microsoft" in platform.release().lower() or os.path.exists("/usr/lib/wsl/lib/libcuda.so.1"),
            "recommended_optimizer_path": "torch_adamw_external_no_fused_adam",
            "disable_deepspeed_jit_ops": True,
            "risk_compute_120_or_newer_gpu": False,
        }
        gpu_name = str(torch_info.get("device_name") or "")
        # RTX 50xx series uses compute capability 12.0+ which breaks DeepSpeed
        # fused ops. Match e.g. "RTX 5060", "RTX 5070 Ti", "RTX 5090" but not
        # "RTX 4050" or "RTX 3050" which contain "50" in a different position.
        if re.search(r"\bRTX\s+50[6-9]0\b", gpu_name, re.IGNORECASE):
            compat["risk_compute_120_or_newer_gpu"] = True
            recommendations.append("Disable fused_adam/JIT DeepSpeed ops; use torch AdamW external optimizer.")
        if compat["is_wsl_like"]:
            recommendations.append("WSL2 detected: NCCL cannot enumerate CUDA devices via its own libcuda path. Set TORCH_DISTRIBUTED_DEFAULT_BACKEND=gloo (done automatically by _deepspeed_env.sh) or deepspeed_runner.py will set it at runtime.")
        if not torch_info.get("cuda_available"):
            recommendations.append("CUDA unavailable in torch; run wsl --shutdown and verify /usr/lib/wsl/lib is in LD_LIBRARY_PATH.")
        if not cuda_info["nvcc_available"]:
            recommendations.append("nvcc not found; install CUDA Toolkit or avoid DeepSpeed ops requiring compilation.")
        if not ds_info.get("import_ok"):
            recommendations.append("DeepSpeed is not importable; install with DS_BUILD_OPS=0.")
        if not mpi_info.get("import_ok"):
            recommendations.append("MPI/mpi4py not healthy; install openmpi-bin libopenmpi-dev and reinstall mpi4py.")
        if compat["python_313_experimental"]:
            recommendations.append("Python 3.13 is experimental for DeepSpeed; prefer Python 3.11/3.12 for portable deployments.")
        doctor_status = "PASS" if torch_info.get("cuda_available") and ds_info.get("import_ok") else "NO_GO"
        return EnvironmentReport(
            python_version=sys.version.split()[0],
            platform={"system": platform.system(), "release": platform.release(), "hostname": platform.node()},
            torch=torch_info,
            deepspeed=ds_info,
            cuda=cuda_info,
            mpi=mpi_info,
            compatibility=compat,
            doctor_status=doctor_status,
            recommendations=recommendations,
        )
