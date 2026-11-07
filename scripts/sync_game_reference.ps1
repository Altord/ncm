param(
    [string]$Destination = "data/reference"
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$dest = Join-Path $root $Destination
New-Item -ItemType Directory -Force -Path $dest | Out-Null

function Sync-Repo([string]$Name, [string]$Url, [string]$Ref, [string[]]$Paths = @()) {
    $path = Join-Path $dest $Name
    if (-not (Test-Path (Join-Path $path ".git"))) {
        if (Test-Path $path) { throw "$path exists but is not a git checkout; move it before syncing." }
        if ($Paths.Count -gt 0) {
            git clone --depth 1 --filter=blob:none --sparse --branch $Ref $Url $path
        } else {
            git clone --depth 1 --branch $Ref $Url $path
        }
        if ($LASTEXITCODE -ne 0) { throw "Clone failed: $Name" }
        if ($Paths.Count -gt 0) {
            git -C $path sparse-checkout set --cone @Paths
            if ($LASTEXITCODE -ne 0) { throw "Sparse checkout failed: $Name" }
        }
    } else {
        git -C $path fetch --depth 1 origin $Ref
        if ($LASTEXITCODE -ne 0) { throw "Fetch failed: $Name" }
        git -C $path checkout -f FETCH_HEAD
        if ($LASTEXITCODE -ne 0) { throw "Checkout update failed: $Name" }
        if ($Paths.Count -gt 0) {
            git -C $path sparse-checkout set --cone @Paths
        } else {
            git -C $path sparse-checkout disable
        }
    }
    $sha = git -C $path rev-parse HEAD
    "{0}`t{1}`t{2}" -f $Name, $Ref, $sha | Add-Content -Encoding utf8 (Join-Path $dest "SOURCES.tsv")
    Write-Host "$Name @ $sha"
}

Set-Content -Encoding utf8 (Join-Path $dest "SOURCES.tsv") "name`tref`tcommit"
Sync-Repo "godot-docs" "https://github.com/godotengine/godot-docs.git" "4.7" @("classes", "getting_started", "tutorials", "engine_details", "about", "_styleguides", "LICENSE.txt", "index.rst")
Sync-Repo "dotnet-docs" "https://github.com/dotnet/docs.git" "main" @("docs/csharp", "docs/fundamentals", "docs/architecture", "docs/core", "docs/standard", "LICENSE", "LICENSE-CODE")
Sync-Repo "godot-demo-projects" "https://github.com/godotengine/godot-demo-projects.git" "4.3" @("mono", "LICENSE.md", "README.md")
Sync-Repo "godot-csharp-essentials" "https://github.com/microsoft/godot-csharp-essentials.git" "main"
Sync-Repo "google-eng-practices" "https://github.com/google/eng-practices.git" "master"

python (Join-Path $root "scripts/fetch_game_design_concepts.py") (Join-Path $dest "game-design-concepts")
if ($LASTEXITCODE -ne 0) { throw "Game Design Concepts fetch failed" }

Write-Host "Reference sources synced to $dest"
