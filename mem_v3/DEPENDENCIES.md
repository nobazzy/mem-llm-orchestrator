# Dependency Policy

MEM v3 is a language-model training workload.

Install the Python runtime dependencies from `requirements.txt`, then install `torch`
from a CUDA-compatible PyTorch index appropriate for the target machine.

For CUDA 12.8-compatible environments:

```bash
python -m pip uninstall -y torch torchvision torchaudio torchtext
python -m pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

Install DeepSpeed separately after PyTorch is working:

```bash
DS_BUILD_OPS=0 python -m pip install --upgrade deepspeed
```

Do not install these for MEM v3:

```txt
torchvision
torchaudio
torchtext
```

Reason:

```txt
MEM v3 trains language models.
It does not use image operators.
It does not use audio operators.
It does not need torchvision datasets/models/transforms.
It does not need torchaudio backends.
Keeping the environment smaller reduces binary mismatch risk.
```
