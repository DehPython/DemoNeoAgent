#!/usr/bin/env python3
"""
LiteASR PT-BR transcription service.

Arquivo único para inferência com o modelo fixo:
    efficient-speech/lite-whisper-large-v3-turbo-acc

Objetivos:
- usar sempre o mesmo checkpoint
- decidir automaticamente o melhor backend (CUDA -> MPS -> CPU)
- ler token do Hugging Face via .env / variável de ambiente
- transcrever áudio em PT-BR
- devolver saída pronta para consumo por LLM

Variáveis de ambiente suportadas:
- HUGGINGFACE_TOKEN=hf_xxx
- HF_TOKEN=hf_xxx
- HUGGING_FACE_TOKEN=hf_xxx
- HF_API_TOKEN=hf_xxx
- LITEASR_HF_TOKEN=hf_xxx
- LITEASR_FORCE_DEVICE=cuda:0|mps|cpu
- LITEASR_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR

Exemplo de .env:
    HUGGINGFACE_TOKEN=hf_seu_token_aqui
    LITEASR_LOG_LEVEL=INFO

Exemplo CLI:
    python liteasr_ptbr_service.py --audio /path/audio.mp3 --json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

import torch
from transformers import AutoModel, AutoProcessor


MODEL_ID = "efficient-speech/lite-whisper-large-v3-turbo-acc"
PROCESSOR_ID = "openai/whisper-large-v3"
SAMPLE_RATE = 16_000
DEFAULT_CHUNK_SECONDS = 28.0
DEFAULT_OVERLAP_SECONDS = 2.0
DEFAULT_MAX_NEW_TOKENS = 384
LANGUAGE_CANDIDATES = ("pt", "portuguese", "pt-br")
TASK = "transcribe"
HF_TOKEN_ENV_KEYS: Tuple[str, ...] = (
    "HUGGINGFACE_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_TOKEN",
    "HF_API_TOKEN",
    "LITEASR_HF_TOKEN",
)


# ---------------------------------------------------------------------------
# .env loader (sem dependência obrigatória de python-dotenv)
# ---------------------------------------------------------------------------
def _load_local_dotenv(dotenv_path: str = ".env") -> None:
    if not os.path.exists(dotenv_path):
        return

    try:
        with open(dotenv_path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key:
                    continue

                if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                    value = value[1:-1]

                os.environ.setdefault(key, value)
    except Exception:
        pass


_load_local_dotenv()

LOG_LEVEL = os.getenv("LITEASR_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("liteasr-ptbr-service")


@dataclass
class RuntimeInfo:
    backend: str
    device: str
    dtype: str
    gpu_name: Optional[str] = None
    total_vram_gb: Optional[float] = None
    free_vram_gb: Optional[float] = None
    total_ram_gb: Optional[float] = None
    selected_model: str = MODEL_ID
    processor_id: str = PROCESSOR_ID
    selection_reason: str = ""
    model_load_attempts: List[str] = field(default_factory=list)
    hf_token_configured: bool = False


@dataclass
class SegmentResult:
    index: int
    start_s: float
    end_s: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    llm_input: str
    segments: List[Dict[str, Any]]
    runtime: Dict[str, Any]
    elapsed_s: float
    audio_duration_s: float


@dataclass
class BackendProfile:
    backend: str
    device: str
    dtype: torch.dtype
    gpu_name: Optional[str]
    total_vram_gb: Optional[float]
    free_vram_gb: Optional[float]
    total_ram_gb: float
    reason: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bytes_to_gb(value: int) -> float:
    return round(value / (1024 ** 3), 2)


def _system_ram_gb() -> float:
    if psutil is not None:
        try:
            return _bytes_to_gb(int(psutil.virtual_memory().total))
        except Exception:
            pass

    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return _bytes_to_gb(int(pages * page_size))
        except Exception:
            pass

    return 0.0


def _normalize_space(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def _resolve_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    if explicit_token and explicit_token.strip():
        return explicit_token.strip()

    for env_key in HF_TOKEN_ENV_KEYS:
        value = os.getenv(env_key, "").strip()
        if value:
            return value
    return None


def _mps_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def _pick_best_cuda_device() -> Optional[BackendProfile]:
    if not torch.cuda.is_available():
        return None

    best_profile: Optional[BackendProfile] = None
    best_score = -1.0

    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        total_vram_gb = _bytes_to_gb(int(props.total_memory))
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            free_vram_gb = _bytes_to_gb(int(free_bytes))
            total_vram_gb = _bytes_to_gb(int(total_bytes))
        except Exception:
            free_vram_gb = total_vram_gb

        score = (
            float(getattr(props, "multi_processor_count", 0)) * 10.0
            + total_vram_gb * 8.0
            + float(getattr(props, "major", 0)) * 2.0
            + float(getattr(props, "minor", 0)) * 0.1
        )

        if score > best_score:
            best_score = score
            best_profile = BackendProfile(
                backend="cuda",
                device=f"cuda:{idx}",
                dtype=torch.float16,
                gpu_name=torch.cuda.get_device_name(idx),
                total_vram_gb=total_vram_gb,
                free_vram_gb=free_vram_gb,
                total_ram_gb=_system_ram_gb(),
                reason="CUDA disponível; selecionada a GPU com melhor score de memória e multiprocessadores.",
            )

    return best_profile


def detect_best_backend(force_device: Optional[str] = None) -> BackendProfile:
    force_device = (force_device or os.getenv("LITEASR_FORCE_DEVICE") or "").strip().lower()
    ram_gb = _system_ram_gb()

    if force_device:
        if force_device.startswith("cuda") and torch.cuda.is_available():
            idx = force_device.split(":", 1)[1] if ":" in force_device else "0"
            idx_int = int(idx)
            props = torch.cuda.get_device_properties(idx_int)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(idx_int)
                free_vram_gb = _bytes_to_gb(int(free_bytes))
                total_vram_gb = _bytes_to_gb(int(total_bytes))
            except Exception:
                total_vram_gb = _bytes_to_gb(int(props.total_memory))
                free_vram_gb = total_vram_gb
            return BackendProfile(
                backend="cuda",
                device=f"cuda:{idx_int}",
                dtype=torch.float16,
                gpu_name=torch.cuda.get_device_name(idx_int),
                total_vram_gb=total_vram_gb,
                free_vram_gb=free_vram_gb,
                total_ram_gb=ram_gb,
                reason=f"Backend forçado via LITEASR_FORCE_DEVICE={force_device}.",
            )

        if force_device == "mps" and _mps_available():
            return BackendProfile(
                backend="mps",
                device="mps",
                dtype=torch.float16,
                gpu_name="Apple Metal (MPS)",
                total_vram_gb=None,
                free_vram_gb=None,
                total_ram_gb=ram_gb,
                reason=f"Backend forçado via LITEASR_FORCE_DEVICE={force_device}.",
            )

        if force_device == "cpu":
            return BackendProfile(
                backend="cpu",
                device="cpu",
                dtype=torch.float32,
                gpu_name=None,
                total_vram_gb=None,
                free_vram_gb=None,
                total_ram_gb=ram_gb,
                reason=f"Backend forçado via LITEASR_FORCE_DEVICE={force_device}.",
            )

        raise RuntimeError(f"Backend forçado inválido ou indisponível: {force_device}")

    cuda_profile = _pick_best_cuda_device()
    if cuda_profile is not None:
        return cuda_profile

    if _mps_available():
        return BackendProfile(
            backend="mps",
            device="mps",
            dtype=torch.float16,
            gpu_name="Apple Metal (MPS)",
            total_vram_gb=None,
            free_vram_gb=None,
            total_ram_gb=ram_gb,
            reason="CUDA indisponível; MPS disponível.",
        )

    return BackendProfile(
        backend="cpu",
        device="cpu",
        dtype=torch.float32,
        gpu_name=None,
        total_vram_gb=None,
        free_vram_gb=None,
        total_ram_gb=ram_gb,
        reason="Sem CUDA/MPS disponível; usando CPU.",
    )


def _longest_token_overlap(left: str, right: str, max_tokens: int = 40) -> int:
    left_tokens = left.split()
    right_tokens = right.split()
    max_k = min(len(left_tokens), len(right_tokens), max_tokens)
    for k in range(max_k, 0, -1):
        if left_tokens[-k:] == right_tokens[:k]:
            return k
    return 0


def _merge_transcripts(chunks: Sequence[str]) -> str:
    clean_chunks = [_normalize_space(x) for x in chunks if _normalize_space(x)]
    if not clean_chunks:
        return ""

    merged = clean_chunks[0]
    for chunk in clean_chunks[1:]:
        overlap = _longest_token_overlap(merged, chunk)
        if overlap > 0:
            merged = f"{merged} {' '.join(chunk.split()[overlap:])}".strip()
        else:
            merged = f"{merged} {chunk}".strip()
        merged = _normalize_space(merged)
    return merged


# ---------------------------------------------------------------------------
# Audio via ffmpeg
# ---------------------------------------------------------------------------
def load_audio_ffmpeg(path: str, sampling_rate: int = SAMPLE_RATE) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {path}")

    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        path,
        "-ac",
        "1",
        "-ar",
        str(sampling_rate),
        "-f",
        "f32le",
        "-hide_banner",
        "-loglevel",
        "error",
        "pipe:1",
    ]

    try:
        process = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg não foi encontrado. Instale o ffmpeg para carregar mp3, wav, flac e outros formatos."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"Falha ao ler áudio com ffmpeg: {stderr.strip()}") from exc

    audio = np.frombuffer(process.stdout, np.float32)
    if audio.size == 0:
        raise ValueError("O arquivo de áudio foi lido, mas nenhum sample válido foi extraído.")

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------
class LiteASRTranscriptionService:
    def __init__(
        self,
        force_device: Optional[str] = None,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        hf_token: Optional[str] = None,
    ) -> None:
        self.force_device = force_device
        self.chunk_seconds = float(chunk_seconds)
        self.overlap_seconds = float(overlap_seconds)
        self.max_new_tokens = int(max_new_tokens)
        self.hf_token = _resolve_hf_token(hf_token)

        self.backend = detect_best_backend(force_device=self.force_device)
        self.runtime = RuntimeInfo(
            backend=self.backend.backend,
            device=self.backend.device,
            dtype=str(self.backend.dtype).replace("torch.", ""),
            gpu_name=self.backend.gpu_name,
            total_vram_gb=self.backend.total_vram_gb,
            free_vram_gb=self.backend.free_vram_gb,
            total_ram_gb=self.backend.total_ram_gb,
            selection_reason=self.backend.reason,
            hf_token_configured=bool(self.hf_token),
        )

        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._forced_decoder_ids: Optional[List[List[int]]] = None

        self._configure_torch_runtime()

    def _configure_torch_runtime(self) -> None:
        try:
            torch.set_num_threads(max(1, min(os.cpu_count() or 1, 8)))
        except Exception:
            pass

        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        if self.backend.backend == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass

    def _build_forced_decoder_ids(self, processor: Any) -> Optional[List[List[int]]]:
        for lang in LANGUAGE_CANDIDATES:
            try:
                if hasattr(processor, "get_decoder_prompt_ids"):
                    ids = processor.get_decoder_prompt_ids(language=lang, task=TASK)
                    if ids:
                        LOGGER.debug("forced_decoder_ids resolvido com language=%s", lang)
                        return ids
            except Exception:
                continue

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "set_prefix_tokens"):
            for lang in LANGUAGE_CANDIDATES:
                try:
                    tokenizer.set_prefix_tokens(language=lang, task=TASK, predict_timestamps=False)
                    prefix = getattr(tokenizer, "prefix_tokens", None)
                    if prefix:
                        return [[i, int(tok)] for i, tok in enumerate(prefix)]
                except Exception:
                    continue

        LOGGER.warning("Não foi possível montar forced_decoder_ids para PT-BR; prosseguindo sem prompt forçado.")
        return None

    def _release_model(self) -> None:
        self._model = None
        self._processor = None
        self._forced_decoder_ids = None
        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def _load_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        LOGGER.info("Carregando modelo fixo: %s em %s", MODEL_ID, self.backend.device)
        self.runtime.model_load_attempts.append(f"{MODEL_ID} @ {self.backend.device}")

        dtype = self.backend.dtype if self.backend.backend != "cpu" else torch.float32
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        processor_kwargs: Dict[str, Any] = {}

        if self.backend.backend != "cpu":
            model_kwargs["torch_dtype"] = dtype

        if self.hf_token:
            model_kwargs["token"] = self.hf_token
            processor_kwargs["token"] = self.hf_token

        try:
            processor = AutoProcessor.from_pretrained(PROCESSOR_ID, **processor_kwargs)
            model = AutoModel.from_pretrained(MODEL_ID, **model_kwargs)
            model.to(self.backend.device)
            if self.backend.backend != "cpu":
                model.to(dtype=dtype)
            model.eval()
        except Exception:
            self._release_model()
            raise

        self._processor = processor
        self._model = model
        self._forced_decoder_ids = self._build_forced_decoder_ids(processor)
        self.runtime.selection_reason += f" Modelo fixo carregado: {MODEL_ID}."

    def ensure_model_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            self._load_model()
        except Exception as exc:
            backend_hint = (
                f"backend={self.backend.backend}, device={self.backend.device}, "
                f"free_vram_gb={self.backend.free_vram_gb}, total_ram_gb={self.backend.total_ram_gb}"
            )
            raise RuntimeError(
                "Falha ao carregar o modelo fixo LiteASR. "
                f"Modelo: {MODEL_ID}. {backend_hint}. Erro original: {exc}"
            ) from exc

    def warmup(self) -> None:
        self.ensure_model_loaded()
        dummy = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
        try:
            self._transcribe_chunk(dummy)
        except Exception as exc:
            LOGGER.debug("Warmup falhou, mas o serviço segue utilizável: %s", exc)

    def _make_chunks(self, audio: np.ndarray) -> List[Tuple[float, float, np.ndarray]]:
        if self.chunk_seconds <= 0:
            raise ValueError("chunk_seconds deve ser > 0")

        audio = np.asarray(audio, dtype=np.float32)
        total_samples = int(audio.shape[0])
        if total_samples == 0:
            return []

        chunk_size = max(1, int(self.chunk_seconds * SAMPLE_RATE))
        overlap_size = max(0, int(self.overlap_seconds * SAMPLE_RATE))
        step = max(1, chunk_size - overlap_size)

        chunks: List[Tuple[float, float, np.ndarray]] = []
        start = 0
        index = 0
        while start < total_samples:
            end = min(total_samples, start + chunk_size)
            chunk = audio[start:end]
            start_s = start / SAMPLE_RATE
            end_s = end / SAMPLE_RATE
            if np.max(np.abs(chunk)) > 1e-5:
                chunks.append((start_s, end_s, chunk))
            else:
                LOGGER.debug("Chunk %d ignorado por silêncio absoluto.", index)
            if end >= total_samples:
                break
            start += step
            index += 1

        return chunks

    @torch.inference_mode()
    def _transcribe_chunk(self, chunk_audio: np.ndarray) -> str:
        if self._model is None or self._processor is None:
            raise RuntimeError("Modelo não carregado.")

        proc_out = self._processor(
            chunk_audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = proc_out.input_features.to(self.backend.device)
        if self.backend.backend != "cpu":
            input_features = input_features.to(self.backend.dtype)

        decoder_prompt_len = len(self._forced_decoder_ids or [])
        max_target_positions = int(getattr(self._model.config, "max_target_positions", 448))
        safe_max_new_tokens = max_target_positions - decoder_prompt_len

        if safe_max_new_tokens <= 0:
            raise RuntimeError(
                f"Prompt do decoder grande demais: decoder_prompt_len={decoder_prompt_len}, "
                f"max_target_positions={max_target_positions}"
            )

        effective_max_new_tokens = min(self.max_new_tokens, safe_max_new_tokens)

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": effective_max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
        }

        if self._forced_decoder_ids:
            generate_kwargs["forced_decoder_ids"] = self._forced_decoder_ids

        predicted_ids = self._model.generate(input_features, **generate_kwargs)
        text = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return _normalize_space(text)

    def transcribe_audio_array(self, audio: np.ndarray) -> TranscriptionResult:
        started = time.perf_counter()
        self.ensure_model_loaded()

        chunks = self._make_chunks(audio)
        if not chunks:
            raise ValueError("O áudio está vazio ou silencioso demais para transcrição.")

        segments: List[SegmentResult] = []
        chunk_texts: List[str] = []

        for idx, (start_s, end_s, chunk) in enumerate(chunks):
            text = self._transcribe_chunk(chunk)
            if not text:
                continue
            segments.append(
                SegmentResult(
                    index=idx,
                    start_s=round(start_s, 3),
                    end_s=round(end_s, 3),
                    text=text,
                )
            )
            chunk_texts.append(text)

        final_text = _merge_transcripts(chunk_texts)
        elapsed = time.perf_counter() - started
        duration_s = round(len(audio) / SAMPLE_RATE, 3)

        result = TranscriptionResult(
            text=final_text,
            llm_input=final_text,
            segments=[asdict(seg) for seg in segments],
            runtime=asdict(self.runtime),
            elapsed_s=round(elapsed, 3),
            audio_duration_s=duration_s,
        )
        return result

    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        audio = load_audio_ffmpeg(audio_path, sampling_rate=SAMPLE_RATE)
        return self.transcribe_audio_array(audio)

    def transcribe_to_llm_payload(self, audio_path: str, include_segments: bool = True) -> Dict[str, Any]:
        result = self.transcribe_file(audio_path)
        payload: Dict[str, Any] = {
            "type": "asr_transcription",
            "language": "pt-BR",
            "model": MODEL_ID,
            "text": result.text,
            "llm_input": result.llm_input,
            "runtime": result.runtime,
            "audio_duration_s": result.audio_duration_s,
            "elapsed_s": result.elapsed_s,
        }
        if include_segments:
            payload["segments"] = result.segments
        return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LiteASR PT-BR transcription service (modelo fixo: lite-whisper-large-v3-turbo-acc)"
    )
    parser.add_argument("--audio", required=True, help="Caminho do áudio (mp3, wav, flac, etc.)")
    parser.add_argument("--json", action="store_true", help="Imprime o resultado completo em JSON")
    parser.add_argument("--warmup", action="store_true", help="Faz warmup do modelo após carregar")
    parser.add_argument("--force-device", default=None, help="Força backend: cuda:0 | mps | cpu")
    parser.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument("--overlap-seconds", type=float, default=DEFAULT_OVERLAP_SECONDS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Token do Hugging Face. Se omitido, usa HUGGINGFACE_TOKEN/HF_TOKEN do ambiente ou .env.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)

    service = LiteASRTranscriptionService(
        force_device=args.force_device,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        max_new_tokens=args.max_new_tokens,
        hf_token=args.hf_token,
    )

    if args.warmup:
        service.warmup()

    result = service.transcribe_file(args.audio)
    result_dict = asdict(result)

    if args.json:
        print(json.dumps(result_dict, ensure_ascii=False, indent=2))
    else:
        print(result.text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())