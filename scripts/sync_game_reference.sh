#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${1:-$repo_root/data/reference}"
mkdir -p "$destination"
manifest="$destination/SOURCES.tsv"
printf 'name\tref\tcommit\n' > "$manifest"

sync_repo() {
  local name="$1" url="$2" ref="$3"
  shift 3
  local path="$destination/$name"
  if [[ ! -d "$path/.git" ]]; then
    if [[ -e "$path" ]]; then
      echo "Refusing to overwrite non-git path: $path" >&2
      return 1
    fi
    if (( $# )); then
      git clone --depth 1 --filter=blob:none --sparse --branch "$ref" "$url" "$path"
    else
      git clone --depth 1 --branch "$ref" "$url" "$path"
    fi
  else
    git -C "$path" fetch --depth 1 origin "$ref"
    git -C "$path" checkout -f FETCH_HEAD
  fi
  if (( $# )); then
    git -C "$path" sparse-checkout set --cone "$@"
  else
    git -C "$path" sparse-checkout disable
  fi
  printf '%s\t%s\t%s\n' "$name" "$ref" "$(git -C "$path" rev-parse HEAD)" >> "$manifest"
}

sync_repo godot-docs https://github.com/godotengine/godot-docs.git 4.7 \
  classes getting_started tutorials engine_details about _styleguides LICENSE.txt index.rst
sync_repo dotnet-docs https://github.com/dotnet/docs.git main \
  docs/csharp docs/fundamentals docs/architecture docs/core docs/standard LICENSE LICENSE-CODE
sync_repo godot-demo-projects https://github.com/godotengine/godot-demo-projects.git 4.3 \
  mono LICENSE.md README.md
sync_repo godot-csharp-essentials https://github.com/microsoft/godot-csharp-essentials.git main
sync_repo google-eng-practices https://github.com/google/eng-practices.git master

python3 "$repo_root/scripts/fetch_game_design_concepts.py" "$destination/game-design-concepts"

echo "Reference sources synced to $destination"
