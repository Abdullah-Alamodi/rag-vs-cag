$ErrorActionPreference = "Stop"

# Optional but useful if your shell does not already have these set.
$env:UV_CACHE_DIR = "$(Get-Location)\.uv-cache"
$env:UV_PYTHON_INSTALL_DIR = "$(Get-Location)\.uv-python"


$splits = @("small", "medium", "large")
$topKs = @(1, 3, 5, 10)

foreach ($split in $splits) {
    Write-Host "=== Running CAG | split=$split ==="
    uv run .\main.py cag --split $split

    foreach ($topK in $topKs) {
        Write-Host "=== Running RAG | split=$split | top-k=$topK ==="
        uv run .\main.py rag --split $split --top-k $topK
    }
}

Write-Host "=== Building research-question summaries ==="
uv run .\main.py compare --limit 0