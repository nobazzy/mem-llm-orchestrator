# Release Checklist — mem_v3

This package was cleaned for public Git/release use.

## Included

```txt
source code
configs
docs
sample datasets
README.md
README_PT.md
RELEASE_NOTES.md
EVALUATION.md
DEPENDENCIES.md
requirements.txt
```

## Excluded

```txt
API keys
local secrets
virtual environments
Python caches
runtime logs
runtime reports
evidence folders
checkpoint folders
dataset caches
large checkpoint/model tensor files
machine-specific local paths
backup files
temporary ZIP/log artifacts
```

## Validation performed

```txt
bash -n scripts/*.sh: OK
python -m compileall: OK
python scripts/v89_static_validation.py: PASS
bash scripts/validate_mem_v3_configs.sh: OK
secret/path/artifact scan: clean
```

## Dependency policy

MEM v3 is a language-model workload. Install only `torch` from the CUDA-compatible PyTorch index for the target environment. Do not install `torchvision`, `torchaudio` or `torchtext` for this package.
