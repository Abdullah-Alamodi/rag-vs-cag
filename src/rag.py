from __future__ import annotations

import os
from time import perf_counter
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm.auto import tqdm

from utils import (
    CONFIG,
    Config,
    bertscore_f1,
    build_qa_messages,
    configure_local_caches,
    generate_answer,
    load_generator,
    load_subset,
    normalize_rows,
    render_generation_prompt,
    save_results,
    select_device,
    unique_passages,
)


class LlamaIndexRAG:
    """Paragraph-level dense RAG using LlamaIndex's in-memory vector store."""

    def __init__(self, embedding_model_id: str, max_tokens: int, device: str) -> None:
        configure_local_caches()
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model_id,
            max_length=max_tokens,
            normalize=True,
            query_instruction="",
            text_instruction="",
            token=os.getenv("HF_TOKEN"),
            device=device,
            model_kwargs={"dtype": "bfloat16" if device == "cuda" else "float32"},
            show_progress_bar=True,
        )
        self.passages: list[str] = []
        self.index: VectorStoreIndex | None = None

    def build_index(self, passages: list[str]) -> float:
        start = perf_counter()
        self.passages = list(passages)
        nodes = [TextNode(id_=str(i), text=passage) for i, passage in enumerate(passages)]
        self.index = VectorStoreIndex(
            nodes,
            embed_model=self.embed_model,
            show_progress=True,
        )
        return perf_counter() - start

    def retrieve(self, question: str, top_k: int) -> list[dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Build the RAG index before retrieval.")
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        retrieved = retriever.retrieve(question)
        return [
            {
                "index": int(item.node.node_id),
                "text": item.node.get_content(metadata_mode=MetadataMode.NONE),
                "score": float(item.score),
            }
            for item in retrieved
        ]


def run_rag(
    subset: str,
    top_k: int,
    limit: int | None = None,
    config: Config = CONFIG,
    prompt_logs: list[dict[str, str]] | None = None,
):
    subset_data = load_subset(subset)
    pool_records = subset_data["pool_records"]
    records = subset_data["pool_evaluation"]
    if limit is not None:
        records = records[:limit]
    passages = unique_passages(pool_records)
    device_kind, _ = select_device(config.device_preference)

    retriever = LlamaIndexRAG(
        config.embedding_model_id,
        max_tokens=config.embedding_max_tokens,
        device=device_kind,
    )
    prepare_seconds = retriever.build_index(passages)
    processor, model, device_kind, device = load_generator(config)

    rows: list[dict[str, Any]] = []
    predictions: list[str] = []
    references: list[str] = []

    for record in tqdm(
        records,
        desc=f"RAG {subset} k={top_k}",
        unit="question",
        dynamic_ncols=True,
    ):
        online_start = perf_counter()
        retrieval_start = perf_counter()
        retrieved = retriever.retrieve(record["question"], top_k)
        retrieval_seconds = perf_counter() - retrieval_start
        messages = build_qa_messages(
            [item["text"] for item in retrieved],
            record["question"],
        )
        if prompt_logs is not None:
            prompt_logs.append(
                {
                    "record_id": record["id"],
                    "question": record["question"],
                    "prompt": render_generation_prompt(processor, messages),
                }
            )
        answer, generation_seconds, generated_tokens = generate_answer(
            model,
            processor,
            messages,
            device_kind,
            device,
            config,
        )
        online_latency_seconds = perf_counter() - online_start
        tokens_per_second = (
            generated_tokens / generation_seconds if generation_seconds > 0 else 0.0
        )
        predictions.append(answer)
        references.append(record["answer"])
        rows.append(
            {
                "method": "rag",
                "subset": subset,
                "top_k": top_k,
                "record_id": record["id"],
                "question": record["question"],
                "gold_answer": record["answer"],
                "predicted_answer": answer,
                "retrieved_contexts": retrieved,
                "knowledge_passages": len(passages),
                "model_id": config.model_id,
                "generator_dtype": str(model.dtype).removeprefix("torch."),
                "device": device_kind,
                "embedding_model_id": config.embedding_model_id,
                "embedding_max_tokens": config.embedding_max_tokens,
                "vector_backend": "llama_index",
                "max_new_tokens": config.max_new_tokens,
                "retrieval_seconds": retrieval_seconds,
                "generation_seconds": generation_seconds,
                "online_latency_seconds": online_latency_seconds,
                "generated_tokens": generated_tokens,
                "tokens_per_second": tokens_per_second,
                "prepare_seconds": prepare_seconds,
            }
        )

    scores = bertscore_f1(predictions, references, config)
    for row, score in zip(rows, scores, strict=True):
        row["bertscore_f1"] = score

    return normalize_rows(rows), save_results("rag", subset, rows, top_k=top_k)
