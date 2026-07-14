from __future__ import annotations

import datetime as dt
import json
import os
import random
import shutil
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    model_id: str = os.getenv(
        "RAG_VS_CAG_MODEL_ID",
        "unsloth/Meta-Llama-3.1-8B-Instruct",
    )
    tokenizer_id: str = os.getenv(
        "RAG_VS_CAG_TOKENIZER_ID",
        os.getenv(
            "RAG_VS_CAG_MODEL_ID",
            "unsloth/Meta-Llama-3.1-8B-Instruct",
        ),
    )
    embedding_model_id: str = os.getenv(
        "RAG_VS_CAG_EMBEDDING_MODEL_ID",
        "BAAI/bge-m3",
    )
    embedding_max_tokens: int = int(
        os.getenv("RAG_VS_CAG_EMBEDDING_MAX_TOKENS", "2048")
    )
    kaggle_dataset: str = "abdellahhamouda/acqad-dataset"
    hf_split_dataset: str = os.getenv(
        "RAG_VS_CAG_HF_SPLIT_DATASET",
        "abdullah-alamodi/acqad-rag-cag-splits",
    )
    seed: int = 42
    budgets: dict[str, int] = field(
        default_factory=lambda: {"small": 30_000, "medium": 60_000, "large": 90_000}
    )
    rag_top_k: tuple[int, ...] = (2, 4, 6, 10)
    max_new_tokens: int = int(os.getenv("RAG_VS_CAG_MAX_NEW_TOKENS", "300"))
    device_preference: str = os.getenv("RAG_VS_CAG_DEVICE", "auto")
    torch_dtype: str = os.getenv("RAG_VS_CAG_TORCH_DTYPE", "float16")
    bertscore_lang: str = os.getenv("RAG_VS_CAG_BERTSCORE_LANG", "ar")


CONFIG = Config()
DATASET_DIR = PROJECT_ROOT / "dataset"
RESULT_DIR = PROJECT_ROOT / "result"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def configure_local_caches() -> None:
    cache_root = ARTIFACTS_DIR / "external_cache"
    os.environ.setdefault("KAGGLEHUB_CACHE", str(cache_root / "kagglehub"))
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "huggingface" / "transformers"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_root / "sentence_transformers"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))


def acqad_path() -> Path:
    return DATASET_DIR / "acqad_multihop.json"


def split_path(split: str) -> Path:
    return DATASET_DIR / f"acqad_multihop_{split}.json"


def split_filenames() -> list[str]:
    return [
        "acqad_multihop_small.json",
        "acqad_multihop_medium.json",
        "acqad_multihop_large.json",
        "acqad_multihop_metadata.json",
    ]


def download_acqad_multihop(config: Config = CONFIG) -> Path:
    configure_local_caches()
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    import kagglehub

    dataset_dir = Path(kagglehub.dataset_download(config.kaggle_dataset))
    matches = sorted(dataset_dir.rglob("acqad_multihop.json"))
    if not matches:
        raise FileNotFoundError("Kaggle download did not contain acqad_multihop.json")

    target = acqad_path()
    shutil.copy2(matches[0], target)
    return target


def download_acqad_splits(config: Config = CONFIG) -> list[Path]:
    configure_local_caches()
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    paths: list[Path] = []
    for filename in split_filenames():
        path = hf_hub_download(
            repo_id=config.hf_split_dataset,
            repo_type="dataset",
            filename=filename,
            local_dir=DATASET_DIR,
        )
        paths.append(Path(path))
    return paths


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def normalize_record(item: dict[str, Any], fallback_id: int) -> dict[str, Any]:
    context = item.get("context", [])
    if isinstance(context, str):
        context = [context]
    return {
        "id": str(item.get("id", fallback_id)),
        "question": str(item["question"]).strip(),
        "answer": str(item["answer"]).strip(),
        "context": [str(p).strip() for p in context if str(p).strip()],
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    return [normalize_record(item, i) for i, item in enumerate(data)]


def serialize_passage(index: int, passage: str) -> str:
    return f"[Document {index + 1}]\n{passage}\n"


def serialize_knowledge(passages: list[str]) -> str:
    return "\n".join(serialize_passage(i, passage) for i, passage in enumerate(passages))


def unique_passages(records: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    passages: list[str] = []
    for record in records:
        for passage in record["context"]:
            if passage not in seen:
                seen.add(passage)
                passages.append(passage)
    return passages


def build_splits(config: Config = CONFIG) -> dict[str, dict[str, Any]]:
    configure_local_caches()
    from transformers import AutoTokenizer

    records = load_records(acqad_path())
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_id,
        token=os.getenv("HF_TOKEN"),
    )
    shuffled = list(records)
    random.Random(config.seed).shuffle(shuffled)

    metadata = {
        "dataset_source": "acqad_multihop",
        "tokenizer_used": config.tokenizer_id,
        "random_seed": config.seed,
        "creation_time": dt.datetime.now(dt.UTC).isoformat(),
        "split_file_style": "single dataset directory with descriptive JSON filenames",
        "splits": {},
    }
    subsets: dict[str, dict[str, Any]] = {}

    for split, budget in config.budgets.items():
        selected: list[dict[str, Any]] = []
        passages: list[str] = []
        seen: set[str] = set()
        total_tokens = 0

        for record in shuffled:
            context = record["context"]
            new_passages = [p for p in context if p not in seen]
            extra_tokens = 0
            for offset, passage in enumerate(new_passages):
                text = serialize_passage(len(passages) + offset, passage)
                extra_tokens += len(tokenizer.encode(text, add_special_tokens=False))

            if total_tokens + extra_tokens > budget:
                break

            selected.append(record)
            passages.extend(new_passages)
            seen.update(new_passages)
            total_tokens += extra_tokens

        write_json(split_path(split), selected)
        context_occurrences_count = sum(len(record["context"]) for record in selected)
        subsets[split] = {
            "records": selected,
            "passages": passages,
            "token_count": total_tokens,
            "budget": budget,
            "context_occurrences_count": context_occurrences_count,
            "unique_budget_passages_count": len(passages),
        }
        metadata["splits"][split] = {
            "budget": budget,
            "total_tokens_used": total_tokens,
            "questions_count": len(selected),
            "context_occurrences_count": context_occurrences_count,
            "unique_budget_passages_count": len(passages),
            "file_path": str(split_path(split)),
        }

    write_json(DATASET_DIR / "acqad_multihop_metadata.json", metadata)
    return subsets


def load_split(split: str) -> list[dict[str, Any]]:
    return load_records(split_path(split))


def select_device(preference: str = "auto") -> tuple[str, torch.device]:
    import torch

    preference = preference.lower()
    if preference not in {"auto", "cuda", "cpu"}:
        raise ValueError("Device must be one of: auto, cuda, cpu")
    if preference == "cpu":
        return "cpu", torch.device("cpu")
    if torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    if preference == "cuda":
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    return "cpu", torch.device("cpu")


def synchronize(device_kind: str) -> None:
    if device_kind == "cuda":
        import torch

        torch.cuda.synchronize()


def dtype_from_name(name: str) -> torch.dtype | str:
    import torch

    return {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(name.lower(), "auto")


QA_SYSTEM_INSTRUCTION = (
    "أجب بالعربية من السياق فقط، واجمع الأدلة داخليا عند الحاجة. "
    "أخرج الجواب وحده كما ورد كاملا في السياق، مع الوحدة أو اسم الفئة، "
    "واستخدم القيمة الصريحة لا التقريبية. "
    "الصيغة المطلوبة مثل: «12 شهرا» و«اللغة الفرنسية»."
)


def build_qa_messages(passages: list[str], question: str) -> list[dict[str, str]]:
    system_content = (
        f"{QA_SYSTEM_INSTRUCTION}\n\n"
        "السياق:\n"
        f"{serialize_knowledge(passages)}"
    )
    user_content = f"السؤال:\n{question}\n\nالإجابة:"
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def load_generator(config: Config = CONFIG):
    configure_local_caches()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if "llama-3.1-8b-instruct" not in config.model_id.lower():
        raise ValueError(
            "The controlled experiment requires Llama 3.1 8B Instruct; "
            f"received {config.model_id!r}."
        )

    token = os.getenv("HF_TOKEN")
    processor = AutoTokenizer.from_pretrained(config.tokenizer_id, token=token)
    device_kind, device = select_device(config.device_preference)
    requested_dtype = dtype_from_name(config.torch_dtype)
    kwargs = {
        "token": token,
        "dtype": torch.float32 if device_kind == "cpu" else requested_dtype,
    }
    if device_kind == "cuda":
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
    if device_kind != "cuda":
        model.to(device)
    model.eval()
    return processor, model, device_kind, device


def apply_generation_template(processor, messages: list[dict[str, str]]):
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )


def render_generation_prompt(processor, messages: list[dict[str, str]]) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_generated_answer(processor, generated_ids) -> str:
    answer = processor.decode(generated_ids, skip_special_tokens=True)
    return answer.replace("**", "").strip()


def generation_eos_ids(model) -> set[int]:
    configured = model.generation_config.eos_token_id
    if configured is None:
        return set()
    if isinstance(configured, int):
        return {configured}
    return {int(value) for value in configured}


def generate_answer(
    model,
    processor,
    messages: list[dict[str, str]],
    device_kind: str,
    device,
    config: Config = CONFIG,
):
    import torch

    inputs = apply_generation_template(processor, messages).to(device)
    prompt_len = inputs["input_ids"].shape[-1]
    synchronize(device_kind)
    start = perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
        )
    synchronize(device_kind)
    generation_seconds = perf_counter() - start
    generated_ids = output[0, prompt_len:]
    answer = parse_generated_answer(processor, generated_ids)
    return answer, generation_seconds, int(generated_ids.shape[-1])


def print_prompt_log(method: str, prompt_logs: list[dict[str, str]] | None) -> None:
    if not prompt_logs:
        return

    method_label = method.upper()
    print("\n=== PROMPT LOG ===")
    for index, item in enumerate(prompt_logs, start=1):
        print(
            f"\n--- BEGIN {method_label} PROMPT {index} "
            f"| record_id={item['record_id']} ---"
        )
        print(item["prompt"])
        print(f"--- END {method_label} PROMPT {index} ---")


def bertscore_f1(predictions: list[str], references: list[str], config: Config = CONFIG) -> list[float]:
    configure_local_caches()
    patch_bertscore_tokenizer_compat()
    from bert_score import score

    _, device = select_device(config.device_preference)
    _, _, f1 = score(
        predictions,
        references,
        lang=config.bertscore_lang,
        device=str(device),
        verbose=True,
    )
    return [float(x) for x in f1.cpu().numpy().tolist()]


def patch_bertscore_tokenizer_compat() -> None:
    """Restore a tokenizer method still used by bert-score."""
    from transformers import PreTrainedTokenizerBase

    if hasattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
        return

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.cls_token_id is not None and self.sep_token_id is not None:
            if token_ids_1 is None:
                return [self.cls_token_id, *token_ids_0, self.sep_token_id]
            return [
                self.cls_token_id,
                *token_ids_0,
                self.sep_token_id,
                *token_ids_1,
                self.sep_token_id,
            ]

        encoded = self.prepare_for_model(
            token_ids_0,
            pair_ids=token_ids_1,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return encoded["input_ids"]

    PreTrainedTokenizerBase.build_inputs_with_special_tokens = build_inputs_with_special_tokens


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        for key, value in list(row.items()):
            if isinstance(value, np.generic):
                row[key] = value.item()
    return rows


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _result_stem(method: str, split: str, top_k: int | None) -> str:
    if method == "rag":
        if top_k is None:
            raise ValueError("RAG results require a retrieval depth.")
        return f"rag_{split}_k{top_k}"
    return f"cag_{split}"


def save_results(
    method: str,
    split: str,
    rows: list[dict[str, Any]],
    top_k: int | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot save an empty experiment result.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = normalize_rows(rows)
    stem = _result_stem(method, split, top_k)
    created_at = dt.datetime.now(dt.UTC).isoformat()
    result_payload = {
        "method": method,
        "split": split,
        "top_k": top_k,
        "created_at": created_at,
        "questions": len(rows),
        "results": rows,
    }
    write_json(RESULT_DIR / f"{stem}_result.json", result_payload)

    online_values = [float(row["online_latency_seconds"]) for row in rows]
    timing_summary = (
        {"mean_retrieval_seconds": _mean(rows, "retrieval_seconds")}
        if method == "rag"
        else {"mean_cache_reset_seconds": _mean(rows, "cache_reset_seconds")}
    )
    summary = {
        "method": method,
        "split": split,
        "top_k": top_k,
        "created_at": created_at,
        "questions": len(rows),
        "mean_bertscore_f1": _mean(rows, "bertscore_f1"),
        **timing_summary,
        "mean_generation_seconds": _mean(rows, "generation_seconds"),
        "mean_online_latency_seconds": statistics.fmean(online_values),
        "median_online_latency_seconds": statistics.median(online_values),
        "min_online_latency_seconds": min(online_values),
        "max_online_latency_seconds": max(online_values),
        "total_online_latency_seconds": sum(online_values),
        "mean_generated_tokens": _mean(rows, "generated_tokens"),
        "total_generated_tokens": sum(int(row["generated_tokens"]) for row in rows),
        "mean_tokens_per_second": _mean(rows, "tokens_per_second"),
        "prepare_seconds": max(float(row["prepare_seconds"]) for row in rows),
    }
    if method == "cag":
        summary["prefix_tokens"] = max(int(row["prefix_tokens"]) for row in rows)
        summary["knowledge_passages"] = max(int(row["knowledge_passages"]) for row in rows)

    write_json(RESULT_DIR / f"{stem}_summary.json", summary)
    refresh_experiment_overview()
    return summary


def refresh_experiment_overview() -> dict[str, Any]:
    summary_files = sorted(RESULT_DIR.glob("rag_*_summary.json"))
    summary_files.extend(sorted(RESULT_DIR.glob("cag_*_summary.json")))
    summaries = [load_json(path) for path in summary_files]
    depth_comparisons: list[dict[str, Any]] = []
    best_rag_comparisons: list[dict[str, Any]] = []
    for split in CONFIG.budgets:
        cag_rows = [
            row for row in summaries if row["split"] == split and row["method"] == "cag"
        ]
        rag_rows = [
            row for row in summaries if row["split"] == split and row["method"] == "rag"
        ]
        if not cag_rows:
            continue
        cag_row = cag_rows[0]
        for rag_row in rag_rows:
            quality_delta = cag_row["mean_bertscore_f1"] - rag_row["mean_bertscore_f1"]
            latency_delta = (
                cag_row["mean_online_latency_seconds"]
                - rag_row["mean_online_latency_seconds"]
            )
            if quality_delta > 0 and latency_delta <= 0:
                outcome = "cag_preferred"
            elif quality_delta < 0 and latency_delta > 0:
                outcome = "rag_preferred"
            else:
                outcome = "trade_off"
            depth_comparisons.append(
                {
                    "split": split,
                    "rag_top_k": rag_row["top_k"],
                    "cag_minus_rag_bertscore_f1": quality_delta,
                    "cag_minus_rag_latency_seconds": latency_delta,
                    "outcome": outcome,
                }
            )

        if not rag_rows:
            continue
        best_rag = min(
            rag_rows,
            key=lambda row: (
                -row["mean_bertscore_f1"],
                row["mean_online_latency_seconds"],
                row["top_k"],
            ),
        )
        quality_delta = cag_row["mean_bertscore_f1"] - best_rag["mean_bertscore_f1"]
        latency_delta = (
            cag_row["mean_online_latency_seconds"]
            - best_rag["mean_online_latency_seconds"]
        )
        if quality_delta > 0:
            quality_winner = "cag"
        elif quality_delta < 0:
            quality_winner = "rag"
        else:
            quality_winner = "tie"
        if latency_delta < 0:
            latency_winner = "cag"
        elif latency_delta > 0:
            latency_winner = "rag"
        else:
            latency_winner = "tie"
        if quality_winner == "cag" and latency_winner in {"cag", "tie"}:
            overall_outcome = "cag_preferred"
        elif quality_winner == "rag" and latency_winner in {"rag", "tie"}:
            overall_outcome = "rag_preferred"
        elif quality_winner == "tie" and latency_winner == "cag":
            overall_outcome = "cag_preferred"
        elif quality_winner == "tie" and latency_winner == "rag":
            overall_outcome = "rag_preferred"
        else:
            overall_outcome = "trade_off"
        best_rag_comparisons.append(
            {
                "split": split,
                "rag_selection_rule": "highest_mean_bertscore_then_lowest_latency",
                "selected_rag_top_k": best_rag["top_k"],
                "cag_mean_bertscore_f1": cag_row["mean_bertscore_f1"],
                "rag_mean_bertscore_f1": best_rag["mean_bertscore_f1"],
                "cag_minus_rag_bertscore_f1": quality_delta,
                "quality_winner": quality_winner,
                "cag_mean_online_latency_seconds": cag_row[
                    "mean_online_latency_seconds"
                ],
                "rag_mean_online_latency_seconds": best_rag[
                    "mean_online_latency_seconds"
                ],
                "cag_minus_rag_latency_seconds": latency_delta,
                "latency_winner": latency_winner,
                "overall_outcome": overall_outcome,
            }
        )

    overview = {
        "updated_at": dt.datetime.now(dt.UTC).isoformat(),
        "experiment_summaries": summaries,
        "rag_depth_comparisons": depth_comparisons,
        "best_rag_vs_cag": best_rag_comparisons,
    }
    write_json(RESULT_DIR / "rag_cag_exp_overview.json", overview)
    return overview


def summarize_research_questions() -> dict[str, Any]:
    return refresh_experiment_overview()
