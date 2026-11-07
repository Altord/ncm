# Godot and C# game-development reference set

This reference set supports the Godot + C# game-generation experiment. Source checkouts live under ignored `data/reference/`; they are intentionally not committed into this repository. On Linux run `bash scripts/sync_game_reference.sh`; on Windows PowerShell run `pwsh -File scripts/sync_game_reference.ps1`. `SOURCES.tsv` records exact Git commits used.

For the project-design, architecture, accessibility, and player-experience process, also read [GAME_DESIGN_AND_ARCHITECTURE.md](GAME_DESIGN_AND_ARCHITECTURE.md). It is an original checklist with links to relevant references; it is not mixed into the tokenizer corpus as if prose alone taught the model to make compelling games.

## Included sources

| Source | Scope | License/provenance |
|---|---|---|
| [Godot 4.7 documentation](https://github.com/godotengine/godot-docs/tree/4.7) | Selected API, tutorials, C# integration, workflow, best-practice pages (`classes/`, `getting_started/`, `tutorials/`, `engine_details/`, `about/`, `_styleguides/`) | CC BY 3.0; preserve the included license and attribution |
| [Microsoft .NET documentation](https://github.com/dotnet/docs) | C#, .NET fundamentals, architecture guides, analyzers and conventions | Documentation is CC BY 4.0; code samples have separate terms. Keep repository license files and source notices |
| [Official Godot demo projects](https://github.com/godotengine/godot-demo-projects/tree/4.3/mono) | C# demos: Pong, Dodge the Creeps, multiplayer Pong and related examples | MIT; preserve `LICENSE.md`; branch 4.3 means API examples need checking against Godot 4.7 |
| [Microsoft Godot C# Essentials](https://github.com/microsoft/godot-csharp-essentials) | End-to-end C# lessons and a complete sample game | MIT; keep LICENSE with the checkout |
| [Google Engineering Practices](https://github.com/google/eng-practices) | Code review, change preparation, reviewable scope, and engineering feedback | CC BY 3.0; archived upstream, so useful as a stable snapshot rather than current policy |
| [Ian Schreiber, Game Design Concepts](https://gamedesignconcepts.wordpress.com/) | Foundational game analysis, prototyping, mechanics, feedback, balance, and player experience | CC BY 3.0 US; sync script downloads posts with title, date, URL, and attribution |

The source repositories are authoritative references, not a claim that their recommendations are universally optimal. The Godot docs branch is pinned to 4.7; .NET docs, the Microsoft course, and Google practices track their upstream default branch, and exact revisions are recorded by the sync script. The Game Design Concepts WordPress source is retrieved through its public API and records the fetched post IDs in `SOURCE.json`.

## Curated learning path

This list combines official references, established books/courses, and recurring community recommendations. Popularity is not proof of quality; use each for its intended job and check engine/version fit.

1. **Learn Godot workflow:** official [step-by-step introduction](https://docs.godotengine.org/en/stable/getting_started/step_by_step/) and [best practices](https://docs.godotengine.org/en/stable/tutorials/best_practices/index.html). For hands-on practice, GDQuest's [Learn GDScript From Zero](https://gdquest.github.io/learn-gdscript/) is free/open source, but it teaches GDScript rather than C#; use it for engine concepts and transfer syntax carefully.
2. **Learn the target language:** [C# fundamentals](https://learn.microsoft.com/en-us/dotnet/csharp/) and [coding conventions](https://learn.microsoft.com/en-us/dotnet/csharp/fundamentals/coding-style/coding-conventions), then the Godot docs' C# pages. The engine binding has version-specific constraints, so general .NET examples are not automatically Godot-ready.
3. **Learn game design:** Ian Schreiber's locally synced [Game Design Concepts](https://gamedesignconcepts.wordpress.com/) is the first structured course. Pair it with the mechanics/dynamics/aesthetics lens in [MDA](https://aaai.org/papers/ws04-04-001-mda-a-formal-approach-to-game-design-and-game-research/) and iterative player testing. MDA is a vocabulary for analysis, not a recipe for fun.
4. **Learn maintainable implementation:** Google Engineering Practices (locally synced) gives concise review/change-quality guidance. Use [C4](https://c4model.com/) when a system diagram makes boundaries clearer, and [Game Programming Patterns](https://gameprogrammingpatterns.com/) as a pattern catalog; apply a pattern only when it solves a concrete problem.
5. **Learn interaction quality:** [Nielsen Norman Group's usability heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/), [Game Accessibility Guidelines](https://gameaccessibilityguidelines.com/), and [Xbox Accessibility Guidelines](https://learn.microsoft.com/en-us/gaming/accessibility/guidelines) provide review prompts. Validate with actual playtests and relevant players.

GDQuest, MDA, C4, usability/accessibility sites, and Game Programming Patterns remain links in the documentation; their full content is not copied into the reference corpus because redistribution terms are absent, unclear, or source-specific. The sync script only bundles sources whose reuse terms are verified.

## Build the first tokenizer corpus on Linux

After syncing the references, build fixed-length, file-level split data and prepare the trainer caches:

```bash
source .venv/bin/activate

python -m training.build_reference_corpus \
  --root data/reference \
  --output data/mycode/godot-csharp-reference \
  --sequence-length 64

wc -l data/mycode/godot-csharp-reference/{train,validation}.jsonl

python -m training.prepare_data \
  --config config/train/godot-csharp-tokenizer.yaml \
  --stage tokenizer

python -m training.prepare_data \
  --config config/train/godot-csharp-tokenizer.yaml \
  --stage tokenizer \
  --input data/mycode/godot-csharp-reference/validation.jsonl \
  --cache-dir data/cache/godot-csharp/tokenizer-validation
```

The builder uses short C# / Godot project-file sequences plus a capped amount of relevant prose. It records each row's source path and license label, deduplicates identical files, and keeps chunks from the same file in one split. Inspect `data/mycode/godot-csharp-reference/manifest.json` and `data/reference/SOURCES.tsv` before training. This is suitable for a **VQ-VAE tokenizer reconstruction experiment** only.

The GPU's earlier short tokenizer run measured about 211 examples/second at batch size 16. At that observed speed, 8 hours would consume roughly 6.1 million example presentations (about 380,000 optimizer steps). The reference corpus is much smaller than that in unique examples, so forcing 380,000 steps could repeat it many times. Use a cap of 40 epochs and a wall-clock-oriented ceiling of 330,000 steps, whichever is lower:

```bash
N=$(wc -l < data/mycode/godot-csharp-reference/train.jsonl)
EPOCH_STEPS=$(((N + 15) / 16))
MAX_STEPS=$((EPOCH_STEPS * 40))
if (( MAX_STEPS > 330000 )); then MAX_STEPS=330000; fi
echo "rows=$N steps_per_epoch=$EPOCH_STEPS max_steps=$MAX_STEPS"
```

First run a 100-step smoke pass and inspect its validation loss. If it is finite and checkpoints are written, resume that checkpoint for the longer tokenizer run:

```bash
python -m training.train_tokenizer \
  --config config/train/godot-csharp-tokenizer.yaml --max-steps 100

nohup systemd-inhibit --what=sleep --why="NCM tokenizer experiment" \
  python -m training.train_tokenizer \
  --config config/train/godot-csharp-tokenizer.yaml \
  --resume runs/godot-csharp/tokenizer/checkpoints/latest.pt \
  --max-steps "$MAX_STEPS" \
  > runs/godot-csharp/tokenizer/overnight.log 2>&1 < /dev/null &
echo "PID=$!"
```

The training config checkpoints every 5,000 steps and evaluates every 10,000 to limit full-validation overhead. Keep the machine powered on. On wake, inspect `runs/godot-csharp/tokenizer/logs/{train,validation}.jsonl`, checkpoint timestamps, and the final artifact under `artifacts/godot-csharp/`. Treat falling training loss with flat or worsening validation loss as overfitting; do not interpret lower reconstruction loss as game-generation skill.

## How the sources should be used

1. **For an agent at generation time:** use these documents as searchable retrieval context (RAG / file search), returning small relevant passages with source paths. Current API references belong in retrieval because they change and need citations.
2. **For training:** do not dump every page into the tiny starter training run. Plain documentation teaches continuation of prose and code, not the complete task behavior “build, run, inspect errors, fix, deliver.” Build task examples with prompt, project files, expected behavior, build/test results, and license provenance. Split train/validation/test by project or tutorial, never by adjacent chunks from the same page.
3. **For evaluation:** require the model to create a Godot project from a clean directory, run `dotnet build`, run headless Godot import/parse checks, exercise deterministic tests, and report the produced files. Score compile success, test pass rate, requirement coverage, and repair iterations—not only token loss.
4. **For architecture/style/player experience:** use the engine's scene/node/resource patterns, .NET conventions, and the project playbook as starting points. Add project-specific architecture decisions, acceptance tests, and play feedback; do not try to make one generic “best practices” text corpus resolve conflicting tradeoffs or certify that a game is fun.

## Overnight experiment decision

Do **not** leave the present 8-hour training command running on the starter corpus. It contains only four training and two validation examples, so an 8-hour run mostly measures repeated memorization and wastes compute. The recent 100-step attention A/B showed nearly identical losses, with throughput changing by run/config; it is an implementation benchmark, not evidence of coding competence.

Use the overnight window to fetch the references and build a real, versioned dataset/evaluation suite. Then run a bounded pilot (for example 500–2,000 steps, validation every 100–200 steps, checkpoint every 200–500) only after the dataset contains many distinct, licensed, runnable tasks and a held-out project split. Stop when validation loss plateaus, code-task scores regress, or VRAM/thermal limits are hit. The exact step count must be selected from measured seconds/step on this GPU; eight hours is a wall-clock budget, not a target number of updates.

## First useful test matrix

| Test | Purpose | Pass signal |
|---|---|---|
| Baseline repeatability: same seed/config, 3 short runs | Separate real changes from GPU/runtime noise | Similar loss curves and throughput spread reported |
| Sequence/context and batch sweep | Find safe throughput without OOM | Peak VRAM, examples/s, positions/s and validation loss recorded |
| Dataset scaling: tiny, 100s, then 1k+ examples | Detect memorization and verify data pipeline | Training loss falls while held-out task metrics improve |
| Clean-room code generation tasks | Test actual user goal | Project builds/imports, tests pass, requirements met |
| Retrieval ablation | Compare model-only vs docs retrieved at inference | Higher API correctness/task success with citations |
| Data ablation | Compare code/tasks alone vs docs-derived examples | Held-out game task score improves without contamination |

The overnight tokenizer experiment above studies code-aware quantization and codebook use. It does not train the prompt-following generator. The current VQ-VAE context is 64 GPT-2 tokens, and the generator prefix is 16 tokens; target examples are truncated to that 64-token sequence. This cannot represent or emit a complete game, regardless of how many documentation pages are added. The next architecture milestone is a VRAM-tested longer context; the next generator-data milestone is licensed prompt-to-project examples with build/runtime verification.
