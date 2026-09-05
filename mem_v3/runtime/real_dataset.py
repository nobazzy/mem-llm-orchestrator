from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Iterator, List, Tuple, Optional, Callable

import json
import os
import time
import queue
import threading
import hashlib
from pathlib import Path
import torch

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


@dataclass
class DatasetRuntimeInfo:
    requested_dataset: str
    requested_config: str
    active_dataset: str
    active_config: str
    split: str
    streaming: bool
    fallback_used: bool
    fallback_reason: str
    tokenizer_name: str
    vocab_size: int
    sequence_length: int
    task: str = "causal_language_modeling"
    chaos_profile: str = "clean"
    dataset_mix: str = ""
    active_dataset_index: int = 0
    samples_seen: int = 0
    tokens_emitted: int = 0
    cache_mode: str = "off"
    cache_path: str = ""
    cache_reads: int = 0
    cache_writes: int = 0
    cache_tokens_read: int = 0
    cache_tokens_written: int = 0
    iterator_restarts: int = 0
    empty_rows_seen: int = 0
    dataset_exhaustions: int = 0
    dataset_source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RealDatasetBatcher:
    """Robust causal-LM batcher for MEM v89/v3.

    MEM v3 adds configurable datasets while preserving the validated v89
    controller. The batcher must therefore be tolerant of finite local datasets,
    small smoke-test files and streaming Hugging Face iterators. A dataset ending
    is not a training failure; the iterator is restarted and token emission
    continues. If no valid text can be produced after bounded attempts, an
    explicit RuntimeError is raised instead of leaking StopIteration into the
    controller.
    """


    def _mem_v89_cache_pos_file(self):
        from pathlib import Path as _Path
        import re as _re

        cache_dir = _Path(getattr(self, "cache_dir", "dataset_cache"))
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        parts = []
        for attr in ("dataset_name", "dataset_id", "dataset", "name", "split", "config", "seq_len", "tokenizer_name"):
            try:
                val = getattr(self, attr, None)
                if val is not None:
                    parts.append(str(val))
            except Exception:
                pass

        raw = "|".join(parts) or "real_dataset_cache"
        safe = _re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)[:180]
        return cache_dir / f"{safe}_read_pos.txt"

    def _mem_v89_load_cache_read_pos(self):
        try:
            f = self._mem_v89_cache_pos_file()
            if f.exists():
                self.cache_read_pos = max(0, int(float(f.read_text(encoding="utf-8", errors="ignore").strip() or "0")))
                return True
        except Exception:
            pass
        return False

    def _mem_v89_save_cache_read_pos(self):
        try:
            f = self._mem_v89_cache_pos_file()
            f.write_text(str(int(max(0, int(self.cache_read_pos)))), encoding="utf-8")
            return True
        except Exception:
            return False

    def __init__(
        self,
        *,
        dataset_name: str,
        dataset_config: str,
        fallback_name: str,
        split: str,
        streaming: bool,
        tokenizer_name: str,
        sequence_length: int,
        batch_size: int,
        device: torch.device,
        chaos_profile: str = "clean",
        dataset_mix: str = "",
    ) -> None:
        from datasets import load_dataset  # type: ignore
        from transformers import AutoTokenizer  # type: ignore

        self.sequence_length = int(sequence_length)
        self.batch_size = int(batch_size)
        self.device = device
        self.chaos_profile = chaos_profile
        self.dataset_mix = dataset_mix
        self.fallback_name = fallback_name
        self.split = split
        self.streaming = streaming
        self.dataset_type = str(os.environ.get("MEM_DATASET_TYPE", "huggingface")).strip().lower()
        self.text_field = str(os.environ.get("MEM_DATASET_TEXT_FIELD", "")).strip()
        self.data_files = str(os.environ.get("MEM_DATASET_DATA_FILES", "")).strip()
        self._mix_index = 0
        self.buffer: List[int] = []
        self.iterator_restarts = 0
        self.dataset_exhaustions = 0
        self.empty_rows_seen = 0

        try:
            self.prefetch_batches = max(1, min(32, int(os.environ.get("MEM_DATASET_PREFETCH_BATCHES", "12"))))
        except Exception:
            self.prefetch_batches = 12
        try:
            self.max_empty_rows = max(100, int(os.environ.get("MEM_DATASET_MAX_EMPTY_ROWS", "5000")))
        except Exception:
            self.max_empty_rows = 5000
        try:
            self.max_exhaustions = max(1, int(os.environ.get("MEM_DATASET_MAX_EXHAUSTIONS", "200")))
        except Exception:
            self.max_exhaustions = 200

        self.cache_mode = str(os.environ.get("MEM_DATASET_CACHE_MODE", "memmap")).lower()
        self.cache_dir = Path(os.environ.get("MEM_DATASET_CACHE_DIR", "dataset_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = hashlib.sha1((dataset_name + "|" + dataset_config + "|" + self.data_files + "|" + tokenizer_name).encode("utf-8", "ignore")).hexdigest()[:10]
        safe_name = (dataset_name + "_" + dataset_config + "_" + tokenizer_name + "_" + fingerprint).replace("/", "_").replace(":", "_").replace(" ", "_")
        self.cache_path = self.cache_dir / f"{safe_name}_seq{self.sequence_length}_i32.bin"
        self.cache_read_pos = 0
        try:
            self._mem_v89_load_cache_read_pos()
        except Exception:
            pass
        self.cache_reads = 0
        self.cache_writes = 0
        self.cache_tokens_read = 0
        self.cache_tokens_written = 0

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = int(getattr(self.tokenizer, "vocab_size", len(self.tokenizer)))

        active_name = dataset_name
        active_config = dataset_config
        fallback_used = False
        fallback_reason = ""
        dataset_source = self.dataset_type or "huggingface"
        self.datasets: List[Iterable[Dict[str, Any]]] = []
        self.iterators: List[Iterator[Dict[str, Any]]] = []
        self._iterator_factory: Optional[Callable[[], Iterator[Dict[str, Any]]]] = None

        try:
            mix_specs = self._parse_mix(dataset_mix) if chaos_profile != "clean" and dataset_mix and self.dataset_type == "huggingface" else []
            if mix_specs:
                for name, config in mix_specs:
                    ds = self._load_hf(load_dataset, name, config, split, streaming)
                    self.datasets.append(ds)
                self.iterators = [iter(ds) for ds in self.datasets]
                active_name = "real_mix[" + ",".join(name for name, _ in mix_specs) + "]"
                active_config = "mixed"
                dataset_source = "huggingface_mix"
            elif self.dataset_type in {"local_jsonl", "jsonl"} and self.data_files:
                path = Path(self.data_files)
                self._iterator_factory = lambda path=path: self._iter_local_jsonl(path)
                self.iterator = self._iterator_factory()
                active_name = "json"
                active_config = ""
                dataset_source = f"local_jsonl:{path}"
            elif self.dataset_type in {"local_txt", "local_text", "text"} and self.data_files:
                path = Path(self.data_files)
                self._iterator_factory = lambda path=path: self._iter_local_txt(path)
                self.iterator = self._iterator_factory()
                active_name = "text"
                active_config = ""
                dataset_source = f"local_txt:{path}"
            else:
                self.dataset = self._load_hf(load_dataset, dataset_name, dataset_config, split, streaming)
                self.iterator = iter(self.dataset)
                self._iterator_factory = lambda: iter(self.dataset)
                dataset_source = "huggingface"
        except Exception as exc:
            fallback_used = True
            fallback_reason = f"{type(exc).__name__}: {exc}"
            active_name = fallback_name
            active_config = ""
            dataset_source = "huggingface_fallback"
            self.dataset = self._load_hf(load_dataset, fallback_name, "", split, streaming)
            self.iterator = iter(self.dataset)
            self._iterator_factory = lambda: iter(self.dataset)

        if not self.iterators and not hasattr(self, "iterator"):
            self.iterator = iter(self.dataset)
            self._iterator_factory = lambda: iter(self.dataset)

        self.info = DatasetRuntimeInfo(
            requested_dataset=dataset_name,
            requested_config=dataset_config,
            active_dataset=active_name,
            active_config=active_config,
            split=split,
            streaming=streaming,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason[-500:],
            tokenizer_name=tokenizer_name,
            vocab_size=self.vocab_size,
            sequence_length=self.sequence_length,
            chaos_profile=chaos_profile,
            dataset_mix=(dataset_mix or ("data_files=" + self.data_files if self.data_files else "")),
            cache_mode=self.cache_mode,
            cache_path=str(self.cache_path),
            dataset_source=dataset_source,
        )

        self._iterator_lock = threading.Lock()
        self._queue: queue.Queue[List[int]] = queue.Queue(maxsize=200)
        self._stop_event = threading.Event()

        # Pre-fill initial buffer synchronously
        target_prewarm = (self.sequence_length + 1) * self.batch_size * max(8, self.prefetch_batches)
        if len(self.buffer) < target_prewarm:
            self._extend_from_cache(target_prewarm - len(self.buffer))
        while len(self.buffer) < target_prewarm:
            try:
                row = self._next_row()
                text = self._text_from_row(row)
                if not text:
                    continue
                ids = self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max(self.sequence_length + 1, min(2048, self.sequence_length * 12)),
                )
                if ids:
                    ids = [int(x) for x in ids]
                    self.buffer.extend(ids)
                    self._append_to_cache(ids)
                    self.info.samples_seen += 1
                    self._sync_info_counters()
            except Exception:
                break

        self._worker_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._worker_thread.start()

    @staticmethod
    def _parse_mix(spec: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for part in str(spec or "").split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                name, config = part.split(":", 1)
            else:
                name, config = part, ""
            out.append((name.strip(), config.strip()))
        return out

    @staticmethod
    def _load_hf(load_dataset: Any, name: str, config: str, split: str, streaming: bool) -> Iterable[Dict[str, Any]]:
        kwargs: Dict[str, Any] = {"split": split, "streaming": streaming}
        if config:
            try:
                return load_dataset(name, config, **kwargs)
            except (ValueError, KeyError):
                return load_dataset(name, **kwargs)
        return load_dataset(name, **kwargs)

    def _iter_local_jsonl(self, path: Path) -> Iterator[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"local_jsonl dataset not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    self.empty_rows_seen += 1
                    continue
                if isinstance(obj, dict):
                    yield obj
                else:
                    self.empty_rows_seen += 1

    def _iter_local_txt(self, path: Path) -> Iterator[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"local_txt dataset not found: {path}")
        field = self.text_field or "text"
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield {field: line}

    def _restart_single_iterator(self) -> None:
        self.dataset_exhaustions += 1
        self.iterator_restarts += 1
        self.info.dataset_exhaustions = self.dataset_exhaustions
        self.info.iterator_restarts = self.iterator_restarts
        if self._iterator_factory is not None:
            self.iterator = self._iterator_factory()
        elif hasattr(self, "dataset"):
            self.iterator = iter(self.dataset)
        else:
            raise RuntimeError("Dataset iterator exhausted and no restart factory is available")

    def _switch_to_fallback(self, reason: str) -> None:
        with self._iterator_lock:
            self.info.fallback_used = True
            self.info.fallback_reason = str(reason)[-500:]
            self.info.active_dataset = self.fallback_name
            self.info.active_config = ""
            self.info.dataset_source = "huggingface_fallback"
            from datasets import load_dataset
            self.dataset = self._load_hf(load_dataset, self.fallback_name, "", self.split, self.streaming)
            self.iterator = iter(self.dataset)
            self._iterator_factory = lambda: iter(self.dataset)

    def _next_row(self) -> Dict[str, Any]:
        with self._iterator_lock:
            if self.iterators:
                attempts = max(1, len(self.iterators) * 2)
                last_exc: Optional[BaseException] = None
                for _ in range(attempts):
                    idx = self._mix_index % len(self.iterators)
                    self._mix_index += 1
                    self.info.active_dataset_index = idx
                    try:
                        return next(self.iterators[idx])
                    except StopIteration as exc:
                        last_exc = exc
                        self.dataset_exhaustions += 1
                        self.iterator_restarts += 1
                        self.info.dataset_exhaustions = self.dataset_exhaustions
                        self.info.iterator_restarts = self.iterator_restarts
                        self.iterators[idx] = iter(self.datasets[idx])
                        try:
                            return next(self.iterators[idx])
                        except StopIteration as exc2:
                            last_exc = exc2
                            continue
                raise RuntimeError(f"Dataset mix produced no rows after restart attempts: {last_exc}")

            while True:
                try:
                    return next(self.iterator)
                except StopIteration:
                    if self.dataset_exhaustions >= self.max_exhaustions:
                        if not self.info.fallback_used and self.fallback_name:
                            self._switch_to_fallback("Exhaustion limit reached")
                            continue
                        raise RuntimeError(
                            "Dataset iterator exhausted repeatedly before enough valid tokens were produced. "
                            "Check dataset path/text_field or provide more valid text rows."
                        )
                    self._restart_single_iterator()
                except Exception as exc:
                    if not self.info.fallback_used and self.fallback_name:
                        self._switch_to_fallback(f"Streaming network error: {type(exc).__name__}: {exc}")
                        continue
                    self._restart_single_iterator()

    @staticmethod
    def _text_from_row(row: Dict[str, Any]) -> str:
        preferred = str(os.environ.get("MEM_DATASET_TEXT_FIELD", "")).strip()
        if preferred:
            val = row.get(preferred)
            if isinstance(val, str) and val.strip():
                return val
        for key in ("text", "story", "content", "document", "article"):
            val = row.get(key)
            if isinstance(val, str) and val.strip():
                return val
        for val in row.values():
            if isinstance(val, str) and val.strip():
                return val
        return ""

    def _cache_available_tokens(self) -> int:
        if self.cache_mode not in {"memmap", "disk", "aggressive"}:
            return 0
        try:
            if not self.cache_path.exists():
                return 0
            return max(0, self.cache_path.stat().st_size // 4 - self.cache_read_pos)
        except Exception:
            return 0

    def _extend_from_cache(self, need_tokens: int) -> int:
        if self.cache_mode not in {"memmap", "disk", "aggressive"}:
            return 0
        available = self._cache_available_tokens()
        if available <= 0:
            return 0
        take = min(max(0, need_tokens), available)
        if take <= 0:
            return 0
        try:
            import numpy as np
            mm = np.memmap(str(self.cache_path), mode="r", dtype="int32")
            arr = mm[self.cache_read_pos:self.cache_read_pos + take].astype("int64", copy=True)
            self.buffer.extend(int(x) for x in arr.tolist())
            self.cache_read_pos += take
            try:
                self._mem_v89_save_cache_read_pos()
            except Exception:
                pass
            self.cache_reads += 1
            self.cache_tokens_read += int(take)
            return int(take)
        except Exception:
            return 0

    def _append_to_cache(self, ids: List[int]) -> None:
        if self.cache_mode not in {"memmap", "disk", "aggressive"} or not ids:
            return
        try:
            import numpy as np
            arr = np.asarray(ids, dtype="int32")
            with self.cache_path.open("ab") as fh:
                fh.write(arr.tobytes())
            self.cache_writes += 1
            self.cache_tokens_written += int(arr.size)
        except Exception:
            pass

    def _sync_info_counters(self) -> None:
        self.info.cache_reads = self.cache_reads
        self.info.cache_writes = self.cache_writes
        self.info.cache_tokens_read = self.cache_tokens_read
        self.info.cache_tokens_written = self.cache_tokens_written
        self.info.iterator_restarts = self.iterator_restarts
        self.info.dataset_exhaustions = self.dataset_exhaustions
        self.info.empty_rows_seen = self.empty_rows_seen

    def _prefetch_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                row = self._next_row()
                text = self._text_from_row(row)
                if not text:
                    continue
                ids = self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max(self.sequence_length + 1, min(2048, self.sequence_length * 12)),
                )
                if ids:
                    ids = [int(x) for x in ids]
                    while not self._stop_event.is_set():
                        try:
                            self._queue.put(ids, timeout=0.2)
                            break
                        except queue.Full:
                            continue
            except Exception as exc:
                if not self.info.fallback_used and self.fallback_name:
                    self._switch_to_fallback(f"Prefetch worker error: {type(exc).__name__}: {exc}")
                time.sleep(0.1)

    def _extend_buffer(self) -> None:
        target_tokens = (self.sequence_length + 1) * self.batch_size * self.prefetch_batches
        if len(self.buffer) < target_tokens:
            self._extend_from_cache(target_tokens - len(self.buffer))

        # Drain queue non-blockingly
        while not self._queue.empty() and len(self.buffer) < target_tokens:
            try:
                ids = self._queue.get_nowait()
                self.buffer.extend(ids)
                self._append_to_cache(ids)
                self.info.samples_seen += 1
                self._sync_info_counters()
            except queue.Empty:
                break

        # If buffer is under 1 batch, wait on queue or read directly
        min_needed = (self.sequence_length + 1) * self.batch_size
        while len(self.buffer) < min_needed and not self._stop_event.is_set():
            try:
                ids = self._queue.get(timeout=2.0)
                self.buffer.extend(ids)
                self._append_to_cache(ids)
                self.info.samples_seen += 1
                self._sync_info_counters()
            except queue.Empty:
                try:
                    row = self._next_row()
                    text = self._text_from_row(row)
                    if text:
                        ids = self.tokenizer.encode(
                            text,
                            add_special_tokens=False,
                            truncation=True,
                            max_length=max(self.sequence_length + 1, min(2048, self.sequence_length * 12)),
                        )
                        if ids:
                            ids = [int(x) for x in ids]
                            self.buffer.extend(ids)
                            self._append_to_cache(ids)
                            self.info.samples_seen += 1
                            self._sync_info_counters()
                except Exception:
                    break

    def close(self) -> None:
        if hasattr(self, "_stop_event"):
            self._stop_event.set()

    def __del__(self) -> None:
        self.close()

    def prewarm(self, min_batches: Optional[int] = None) -> Dict[str, Any]:
        before = time.perf_counter()
        batches = int(min_batches or self.prefetch_batches)
        batches = max(1, min(64, batches))
        target_tokens = (self.sequence_length + 1) * self.batch_size * batches
        while len(self.buffer) < target_tokens:
            self._extend_buffer()
            if len(self.buffer) >= target_tokens:
                break
        elapsed = time.perf_counter() - before
        self._sync_info_counters()
        return {
            "prewarm_enabled": True,
            "prewarm_batches": batches,
            "prewarm_target_tokens": target_tokens,
            "prewarm_buffer_tokens": len(self.buffer),
            "prewarm_seconds": round(elapsed, 3),
            "cache_mode": self.cache_mode,
            "cache_path": str(self.cache_path),
            "cache_reads": self.cache_reads,
            "cache_writes": self.cache_writes,
            "cache_tokens_read": self.cache_tokens_read,
            "cache_tokens_written": self.cache_tokens_written,
            "iterator_restarts": self.iterator_restarts,
            "dataset_exhaustions": self.dataset_exhaustions,
            "empty_rows_seen": self.empty_rows_seen,
        }

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self._extend_buffer()
        total = self.batch_size * (self.sequence_length + 1)
        if len(self.buffer) < total:
            raise RuntimeError(f"Dataset buffer underfilled: have {len(self.buffer)} tokens, need {total}")
        chunk = self.buffer[:total]
        del self.buffer[:total]
        data = torch.tensor(chunk, dtype=torch.long).view(self.batch_size, self.sequence_length + 1)
        self.info.tokens_emitted += int(data.numel())
        self._sync_info_counters()
        x = data[:, :-1].to(self.device, non_blocking=True)
        y = data[:, 1:].to(self.device, non_blocking=True)
        return x, y
