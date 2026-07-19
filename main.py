from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ACQAD RAG vs CAG comparison.")
    parser.add_argument(
        "command",
        choices=["download", "download-subsets", "subsets", "rag", "cag", "compare"],
    )
    parser.add_argument(
        "--subset",
        default="small",
        choices=["small", "medium", "large"],
    )
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--embedding-model-id", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the exact generation prompt after the result table.",
    )
    args = parser.parse_args()

    def build_config():
        from utils import CONFIG

        updates = {}
        if args.model_id is not None:
            updates["model_id"] = args.model_id
            updates["tokenizer_id"] = args.tokenizer_id or args.model_id
        elif args.tokenizer_id is not None:
            updates["tokenizer_id"] = args.tokenizer_id
        if args.embedding_model_id is not None:
            updates["embedding_model_id"] = args.embedding_model_id
        if args.max_new_tokens is not None:
            updates["max_new_tokens"] = args.max_new_tokens
        if args.device is not None:
            updates["device_preference"] = args.device
        return replace(CONFIG, **updates) if updates else CONFIG

    if args.command == "download":
        from utils import download_acqad_multihop

        print(download_acqad_multihop(build_config()))
    elif args.command == "download-subsets":
        from utils import download_acqad_subsets

        for path in download_acqad_subsets(build_config()):
            print(path)
    elif args.command == "subsets":
        from utils import build_subsets

        subsets = build_subsets(build_config())
        for name, info in subsets.items():
            print(
                name,
                f"tokens={info['token_count']}",
                f"records={info['records_count']}",
                f"questions={info['questions_count']}",
                f"saved_contexts={info['context_occurrences_count']}",
                f"unique_budget_contexts={info['unique_budget_passages_count']}",
            )
    elif args.command == "rag":
        from rag import run_rag
        from utils import print_prompt_log

        prompt_logs = [] if args.print_prompt else None
        _, summary = run_rag(
            args.subset,
            args.top_k,
            args.limit,
            build_config(),
            prompt_logs=prompt_logs,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print_prompt_log("rag", prompt_logs)
    elif args.command == "cag":
        from cag import run_cag
        from utils import print_prompt_log

        prompt_logs = [] if args.print_prompt else None
        _, summary = run_cag(
            args.subset,
            args.limit,
            build_config(),
            prompt_logs=prompt_logs,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print_prompt_log("cag", prompt_logs)
    elif args.command == "compare":
        from cag import run_cag
        from rag import run_rag
        from utils import summarize_research_questions

        config = build_config()
        total_runs = len(config.budgets) * (1 + len(config.rag_top_k))
        current_run = 0
        for subset in config.budgets:
            current_run += 1
            print(
                f"\n[{current_run}/{total_runs}] Starting CAG subset={subset}",
                flush=True,
            )
            _, cag_summary = run_cag(subset, args.limit, config)
            print(json.dumps(cag_summary, ensure_ascii=False, indent=2))
            print(
                f"[{current_run}/{total_runs}] Saved result/cag_{subset}_result.json "
                f"and result/cag_{subset}_summary.json",
                flush=True,
            )
            for top_k in config.rag_top_k:
                current_run += 1
                print(
                    f"\n[{current_run}/{total_runs}] Starting RAG subset={subset} k={top_k}",
                    flush=True,
                )
                _, rag_summary = run_rag(subset, top_k, args.limit, config)
                print(json.dumps(rag_summary, ensure_ascii=False, indent=2))
                print(
                    f"[{current_run}/{total_runs}] Saved "
                    f"result/rag_{subset}_k{top_k}_result.json and "
                    f"result/rag_{subset}_k{top_k}_summary.json",
                    flush=True,
                )
        overview = summarize_research_questions()
        print(json.dumps(overview, ensure_ascii=False, indent=2))
        print("Saved result/rag_cag_exp_overview.json", flush=True)


if __name__ == "__main__":
    main()
