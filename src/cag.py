from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from tqdm.auto import tqdm

import torch

from utils import (
    ARTIFACTS_DIR,
    CONFIG,
    Config,
    apply_generation_template,
    bertscore_f1,
    build_qa_messages,
    generation_eos_ids,
    load_generator,
    load_subset,
    normalize_rows,
    parse_generated_answer,
    render_generation_prompt,
    save_results,
    synchronize,
    unique_passages,
)


class CacheAugmentedGenerator:
    _QUESTION_SLOT = "__RAG_VS_CAG_QUESTION_SLOT_7F3A__"
    _SUFFIX_VALIDATION_QUESTION = "ما الإجابة الصحيحة؟"

    def __init__(self, config: Config = CONFIG) -> None:
        self.config = config
        self.processor, self.model, self.device_kind, self.device = load_generator(config)
        self.passages: list[str] = []
        self.prefix_input_ids: torch.Tensor | None = None
        self.prefix_tokens = 0
        self.base_cache_length = 0
        self.prepare_seconds = 0.0
        self.kv_cache = None
        self.suffix_before_question: str | None = None
        self.suffix_after_question: str | None = None

    def _prefix_messages(self) -> list[dict[str, str]]:
        if not self.passages:
            raise RuntimeError("Prepare or load the cache before building prompts.")
        return build_qa_messages(self.passages, "")[:1]

    def _prefix_inputs(self):
        return self.processor.apply_chat_template(
            self._prefix_messages(),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=False,
        ).to(self.device)

    def _prefix_prompt(self) -> str:
        return self.processor.apply_chat_template(
            self._prefix_messages(),
            tokenize=False,
            add_generation_prompt=False,
        )

    def _prepare_suffix_template(self) -> None:
        prefix_prompt = self._prefix_prompt()
        full_prompt = render_generation_prompt(
            self.processor,
            build_qa_messages(self.passages, self._QUESTION_SLOT),
        )
        if not full_prompt.startswith(prefix_prompt):
            raise RuntimeError(
                "The official Llama prompt does not begin with the cached prefix prompt."
            )

        suffix_template = full_prompt[len(prefix_prompt) :]
        if suffix_template.count(self._QUESTION_SLOT) != 1:
            raise RuntimeError("Could not isolate the question slot in the CAG prompt.")
        self.suffix_before_question, self.suffix_after_question = suffix_template.split(
            self._QUESTION_SLOT,
            maxsplit=1,
        )

    def _suffix_input_ids(self, question: str) -> torch.Tensor:
        if self.suffix_before_question is None or self.suffix_after_question is None:
            raise RuntimeError("Prepare the CAG suffix template before answering.")

        suffix_text = (
            self.suffix_before_question + question + self.suffix_after_question
        )
        encoded = self.processor(
            suffix_text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        return encoded["input_ids"].to(self.device)

    def _validate_suffix_encoding(self) -> None:
        if self.prefix_input_ids is None:
            raise RuntimeError("Prepare the CAG prefix before validating its suffix.")

        full_inputs = apply_generation_template(
            self.processor,
            build_qa_messages(self.passages, self._SUFFIX_VALIDATION_QUESTION),
        ).to(self.device)
        combined_ids = torch.cat(
            [
                self.prefix_input_ids,
                self._suffix_input_ids(self._SUFFIX_VALIDATION_QUESTION),
            ],
            dim=-1,
        )
        if not torch.equal(combined_ids, full_inputs["input_ids"]):
            raise RuntimeError(
                "The optimized CAG suffix does not reproduce the official Llama prompt tokens."
            )

    @staticmethod
    def _cache_length(kv_cache) -> int:
        if not hasattr(kv_cache, "get_seq_length"):
            raise RuntimeError(
                f"Cache class {kv_cache.__class__.__name__} does not report its sequence length."
            )
        return int(kv_cache.get_seq_length())

    def _restore_cache(self) -> None:
        if not hasattr(self.kv_cache, "crop"):
            raise RuntimeError(
                f"Cache class {self.kv_cache.__class__.__name__} does not support crop()."
            )
        self.kv_cache.crop(self.base_cache_length)
        restored_length = self._cache_length(self.kv_cache)
        if restored_length != self.base_cache_length:
            raise RuntimeError(
                "Cache reset failed: "
                f"expected {self.base_cache_length}, got {restored_length}."
            )

    def _generate_from_cache(
        self,
        suffix_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        attention_length: int,
    ) -> torch.Tensor:
        eos_ids = generation_eos_ids(self.model)
        generated: list[torch.Tensor] = []

        output = self.model(
            input_ids=suffix_ids,
            attention_mask=attention_mask[:, :attention_length],
            past_key_values=self.kv_cache,
            use_cache=True,
            logits_to_keep=1,
        )
        self.kv_cache = output.past_key_values
        logits = output.logits[:, -1, :]

        for step in range(self.config.max_new_tokens):
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated.append(next_token)

            if (
                int(next_token.item()) in eos_ids
                or step + 1 == self.config.max_new_tokens
            ):
                break

            attention_length += 1

            output = self.model(
                input_ids=next_token,
                attention_mask=attention_mask[:, :attention_length],
                past_key_values=self.kv_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            self.kv_cache = output.past_key_values
            logits = output.logits[:, -1, :]

        if not generated:
            return torch.empty(
                (suffix_ids.shape[0], 0),
                dtype=suffix_ids.dtype,
                device=self.device,
            )
        return torch.cat(generated, dim=-1)

    def prepare_cache(self, passages: list[str]) -> None:
        self.passages = list(passages)
        prefix_inputs = self._prefix_inputs()
        self.prefix_input_ids = prefix_inputs["input_ids"]
        self.prefix_tokens = self.prefix_input_ids.shape[-1]
        self._prepare_suffix_template()
        self._validate_suffix_encoding()

        synchronize(self.device_kind)
        start = perf_counter()
        with torch.inference_mode():
            output = self.model(
                **prefix_inputs,
                use_cache=True,
                logits_to_keep=1,
            )
        synchronize(self.device_kind)

        self.kv_cache = output.past_key_values
        self.base_cache_length = self._cache_length(self.kv_cache)
        self.prepare_seconds = perf_counter() - start

    def save_cache(self, path: Path) -> None:
        if self.kv_cache is None or self.prefix_input_ids is None:
            raise RuntimeError("Prepare the cache before saving it.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kv_cache": self.kv_cache,
                "passages": self.passages,
                "prefix_input_ids": self.prefix_input_ids,
                "prefix_tokens": self.prefix_tokens,
                "base_cache_length": self.base_cache_length,
                "prepare_seconds": self.prepare_seconds,
            },
            path,
        )

    def load_cache(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.kv_cache = payload["kv_cache"]
        self.passages = list(payload["passages"])
        self.prefix_input_ids = payload["prefix_input_ids"].to(self.device)
        self.prefix_tokens = int(payload["prefix_tokens"])
        self.base_cache_length = int(payload["base_cache_length"])
        self.prepare_seconds = float(payload["prepare_seconds"])
        self._prepare_suffix_template()
        self._validate_suffix_encoding()

    def answer(self, question: str) -> tuple[str, float, float, float, int]:
        if self.kv_cache is None or self.prefix_input_ids is None:
            raise RuntimeError("Prepare or load the cache before answering.")

        online_start = perf_counter()
        reset_start = perf_counter()
        self._restore_cache()
        synchronize(self.device_kind)
        cache_reset_seconds = perf_counter() - reset_start

        suffix_ids = self._suffix_input_ids(question)
        full_length = self.prefix_tokens + suffix_ids.shape[-1]
        attention_mask = torch.ones(
            (suffix_ids.shape[0], full_length + self.config.max_new_tokens),
            dtype=torch.long,
            device=self.device,
        )

        synchronize(self.device_kind)
        generation_start = perf_counter()
        try:
            with torch.inference_mode():
                output_tokens = self._generate_from_cache(
                    suffix_ids,
                    attention_mask,
                    full_length,
                )
        finally:
            synchronize(self.device_kind)
            generation_seconds = perf_counter() - generation_start

        answer = parse_generated_answer(self.processor, output_tokens[0])
        online_latency_seconds = perf_counter() - online_start
        generated_tokens = int(output_tokens.shape[-1])
        return (
            answer,
            cache_reset_seconds,
            generation_seconds,
            online_latency_seconds,
            generated_tokens,
        )

    def prompt_for_question(self, question: str) -> str:
        return render_generation_prompt(
            self.processor,
            build_qa_messages(self.passages, question),
        )


def run_cag(
    subset: str,
    limit: int | None = None,
    config: Config = CONFIG,
    save_cache: bool = True,
    prompt_logs: list[dict[str, str]] | None = None,
):
    subset_data = load_subset(subset)
    pool_records = subset_data["pool_records"]
    all_questions = subset_data["pool_evaluation"]
    records = all_questions if limit is None else all_questions[:limit]
    passages = unique_passages(pool_records)

    cag = CacheAugmentedGenerator(config)
    cag.prepare_cache(passages)
    if save_cache:
        cag.save_cache(ARTIFACTS_DIR / "cag_cache" / f"{subset}.pt")

    rows: list[dict[str, Any]] = []
    predictions: list[str] = []
    references: list[str] = []
    for record in tqdm(
        records,
        desc=f"CAG {subset}",
        unit="question",
        dynamic_ncols=True,
    ):
        if prompt_logs is not None:
            prompt_logs.append(
                {
                    "record_id": record["id"],
                    "question": record["question"],
                    "prompt": cag.prompt_for_question(record["question"]),
                }
            )
        (
            answer,
            cache_reset_seconds,
            generation_seconds,
            online_latency_seconds,
            generated_tokens,
        ) = cag.answer(record["question"])
        tokens_per_second = (
            generated_tokens / generation_seconds if generation_seconds > 0 else 0.0
        )
        predictions.append(answer)
        references.append(record["answer"])
        rows.append(
            {
                "method": "cag",
                "subset": subset,
                "top_k": None,
                "record_id": record["id"],
                "question": record["question"],
                "gold_answer": record["answer"],
                "predicted_answer": answer,
                "knowledge_passages": len(passages),
                "model_id": config.model_id,
                "generator_dtype": str(cag.model.dtype).removeprefix("torch."),
                "device": cag.device_kind,
                "max_new_tokens": config.max_new_tokens,
                "cache_reset_seconds": cache_reset_seconds,
                "generation_seconds": generation_seconds,
                "online_latency_seconds": online_latency_seconds,
                "generated_tokens": generated_tokens,
                "tokens_per_second": tokens_per_second,
                "prepare_seconds": cag.prepare_seconds,
                "prefix_tokens": cag.prefix_tokens,
            }
        )

    evaluation_device_kind = cag.device_kind
    del cag
    if evaluation_device_kind == "cuda":
        torch.cuda.empty_cache()

    scores = bertscore_f1(predictions, references, config)
    for row, score in zip(rows, scores, strict=True):
        row["bertscore_f1"] = score

    return normalize_rows(rows), save_results("cag", subset, rows)
