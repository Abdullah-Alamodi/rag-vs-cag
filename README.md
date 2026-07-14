# RAG vs CAG for ACQAD Multi-hop

Simple Python project for comparing retrieval-augmented generation (RAG) and cache-augmented generation (CAG) on Arabic ACQAD multi-hop QA.

This is not a Python package. `uv` is used only to manage the environment and required libraries.

## Files

- `src/utils.py`: shared configuration, dataset handling, split creation, model loading, metrics, and result saving.
- `src/rag.py`: dense RAG with the embedding index stored as Python variables (`passages` and `embeddings`), not a vector database.
- `src/cag.py`: CAG with precomputed KV cache.
- `main.py`: command runner for downloading data, building splits, and running the comparison.
- `rag_vs_cag_experiment.ipynb`: notebook entry point for showing results.
- `context/`: methodology notes and the original split-construction reference.

## Dataset Layout

Use one `dataset/` directory with descriptive JSON files:

- `dataset/acqad_multihop.json`
- `dataset/acqad_multihop_small.json`
- `dataset/acqad_multihop_medium.json`
- `dataset/acqad_multihop_large.json`
- `dataset/acqad_multihop_metadata.json`

This is more Pythonic here than `dataset/small/small.json` because there are only three split files and no split-specific assets. A flat `dataset/` directory is easier to inspect, pass into scripts, and version-control selectively.

Experiment outputs are written to `result/`.

## Result Layout

Each condition writes JSON only:

- `result/cag_<split>_result.json`: complete per-question CAG results.
- `result/cag_<split>_summary.json`: aggregate CAG metrics.
- `result/rag_<split>_k<depth>_result.json`: complete per-question RAG results,
  including retrieved passage text, index, and score.
- `result/rag_<split>_k<depth>_summary.json`: aggregate RAG metrics.
- `result/rag_cag_exp_overview.json`: all available condition summaries and pairwise
  RAG-CAG quality/latency comparisons.

The result writers do not create CSV or JSONL files.

## Split Construction Logic

The split builder follows the methodology scenario directly:

1. Load the full ACQAD multi-hop file.
2. Normalize each record to `id`, `question`, `answer`, and `context`, excluding decomposition.
3. Shuffle records with seed `42`.
4. Visit each shuffled record as an atomic unit.
5. Use a temporary set to count only unseen passages toward the token budget.
6. If adding the current record's unseen passages would exceed the split token budget, stop and exclude that current record.

The saved subset keeps each included record's full original `context` list. No context passages are removed from the saved split JSON. The temporary set is only for budget accounting and for reporting the unique passage count.

## Setup

```powershell
$env:UV_CACHE_DIR = "$(Get-Location)\.uv-cache"
$env:UV_PYTHON_INSTALL_DIR = "$(Get-Location)\.uv-python"
uv sync
```

Set credentials if needed:

```powershell
Copy-Item .env.example .env
```

## Workflow

Download ACQAD multi-hop:

```powershell
uv run python main.py download
```

Build the token-budget splits:

The Meta checkpoint is gated. Accept its Hugging Face license and set `HF_TOKEN` before
building splits or running experiments.

```powershell
uv run python main.py splits
```

Run a smoke test:

```powershell
uv run python main.py rag --split small --top-k 2 --limit 1
uv run python main.py cag --split small --limit 1
```

Both methods load `unsloth/Meta-Llama-3.1-8B-Instruct` through `AutoTokenizer`
and `AutoModelForCausalLM`, using the same Llama 3.1 8B Instruct weights as the
generator in the original CAG study.
Prompts are serialized by Llama's official chat template. RAG and CAG use the same system
instruction, context placement, user question structure, and greedy decoding policy.

Run the comparison:

```powershell
uv run python main.py compare
```

The execution device defaults to `auto`, which uses CUDA when available and CPU
otherwise. Use `--device cuda` or `--device cpu` to select one explicitly:

```bash
uv run python main.py rag --split small --top-k 2 --limit 1 --device cuda
uv run python main.py cag --split small --limit 1 --device cpu
```

Open the notebook:

```powershell
uv run jupyter lab rag_vs_cag_experiment.ipynb
```

## Methodology Defaults

- Dataset file: `acqad_multihop.json`.
- Splits: `small = 30k`, `medium = 60k`, `large = 90k` token budgets.
- Tokenizer and generator: `unsloth/Meta-Llama-3.1-8B-Instruct`.
- Generator precision: FP16, matching the original CAG implementation.
- RAG embeddings: `BAAI/bge-m3` in dense-only mode through LlamaIndex's in-memory
  vector store, with one node per full paragraph, unprefixed passages and queries,
  normalized embeddings, and a 2,048-token encoding limit.
- RAG retrieval depths: `k = 2, 4, 6, 10`.
- CAG: precompute the KV cache once per subset and reuse it for questions.
- Generation cap: `max_new_tokens = 300`, matching the original CAG implementation.
- Metric: BERTScore F1.
- Timing: RAG reports retrieval time, CAG reports KV-cache reset time, and both report
  generation time, complete online latency, generated tokens, and generation throughput.
  Offline index/cache preparation remains separate.
