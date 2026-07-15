$ErrorActionPreference = "Stop"

# Optional but useful if your shell does not already have these set.
$env:UV_CACHE_DIR = "$(Get-Location)\.uv-cache"
$env:UV_PYTHON_INSTALL_DIR = "$(Get-Location)\.uv-python"


$subsets = @("small", "medium", "large")
$topKs = @(1, 3, 5, 10)

foreach ($subset in $subsets) {
    Write-Host "=== Running CAG | subset=$subset ==="
    uv run .\main.py cag --subset $subset

    foreach ($topK in $topKs) {
        Write-Host "=== Running RAG | subset=$subset | top-k=$topK ==="
        uv run .\main.py rag --subset $subset --top-k $topK
    }
}

Write-Host "=== Building research-question summaries ==="
uv run .\main.py compare --limit 0
