# Local Game-Building Agent Plan

**Project:** Build a local-first, multi-specialist system that can turn broad or detailed instructions into a complete, playable game, including generated or procedural assets.

**Research focus:** Next-Concept Modeling (NCM) as a possible specialist for code generation. NCM is an experiment within the system, not an assumption that every task should use NCM.

## Goals and boundaries

- The user can ask for a game in broad terms or specify mechanics, style, and constraints.
- The system produces a project with source code, assets, build/run instructions, and a playable result.
- The system checks its own output by building and running it, inspecting results, and fixing concrete failures.
- We measure results on repeatable tasks and improve the weakest stage.
- Existing Hermes interface, tools, model profiles, and routing are already working. Treat them as fixed and do not change them as part of this project.

## Existing Hermes model setup (fixed constraint)

**Online/API profiles:**

- `deepseek-v4-flash` — manager/routing (`h-manager-deepseek`)
- `deepseek-v4-pro` — stronger manager (`h-manager-deepseek-pro`)
- `gpt-5.4` / Codex — premium coding, architecture, review, judge
- `kimi-k2.6` — heavy coding (`h-coder-heavy`)
- `MiniMax-M2.7` — documentation, blueprints, bulk writing
- Gemini key configured; stable profile mapping is uncertain

**Local Ollama-style models:**

- `qwen3:30b`, `qwen2.5-coder`, `qwen3-coder-next`, `devstral`, local `deepseek-r1`, Llama, Phi, Gemma, and GPT-OSS variants

The user reports that the Hermes workflow already uses existing tools and modest coding models, but local runs take a long time. This plan builds on the existing setup; it does not propose replacing profiles, changing routes, or adding paid subscriptions.

## Target machine and hardware assumptions

- Deployment/training PC: Linux-only, 64 GB system RAM, Radeon RX 9060 XT with 16 GB VRAM.
- An RX 6950 XT may be added later; verify exact ROCm/Linux support and multi-GPU behavior on the actual machine before depending on it.
- The user reports all their GPUs have 16 GB. Confirm exact models and installed Linux distribution when setting up the machine.
- Native Linux ROCm support and supported OS/version combinations must be checked against AMD's current compatibility matrix before installing software.
- Two GPU cards do not automatically pool their VRAM. Plan initially for separate workloads on separate cards; treat model sharding as a later experiment.
- System RAM helps with caching, datasets, and CPU offload, but is not a same-speed substitute for VRAM.

## Specialist roles

An agent is a role, instructions, tools, and output contract; it does not require a unique model. Start with few roles, and reuse existing Hermes profiles where appropriate.

1. **Studio Director** — Hermes orchestration. Converts the request into work items, routes tasks, tracks artifacts, and decides when the acceptance checks pass.
2. **Game Designer** — initially a Director subtask/role. Produces a concise game brief: mechanics, controls, target engine, scenes/levels, art/sound needs, and acceptance checks.
3. **NCM Code Builder** — the research target. Produces and edits code/project files for well-scoped tasks, follows project conventions, and reports assumptions. Start as a prototype; do not assume NCM is already a capable coding model.
4. **Asset Workshop** — image-generation model and/or procedural tools for sprites, backgrounds, UI, and textures. Use an asset manifest with paths, dimensions, formats, and licensing/provenance notes.
5. **Build & Playtest Harness** — deterministic scripts and engine/compiler tools to build, launch, run smoke tests, validate asset references, and capture logs/screenshots.
6. **Visual Reviewer** — vision model or image-analysis tools to inspect screenshots against the brief and report specific visual defects.
7. **Sound Workshop (later)** — optional audio-generation or procedural-audio tools for music and effects.

## End-to-end workflow

1. **Intake:** Director asks for only essential missing constraints, otherwise records sensible defaults.
2. **Brief:** Designer outputs a structured spec and acceptance checklist.
3. **Scaffold:** Create a clean project from a known engine/template and record how to build/run it.
4. **Implementation:** Code Builder implements mechanics in small, reviewable milestones.
5. **Assets:** Asset Workshop creates/assembles assets from the manifest; code references exact generated paths.
6. **Build/test:** Harness builds and launches the game, checks logs and basic interactions, and captures screenshots.
7. **Review:** Director and Visual Reviewer compare the result to the spec and turn failures into bounded repair tasks.
8. **Repair loop:** Make targeted code/asset changes; rerun relevant checks after each change.
9. **Delivery:** Package source, assets, instructions, known limitations, and a reproducible playable build where practical.

## Development phases and learning milestones

### Phase 0 — Freeze the baseline

- Record the target Linux distribution, kernel, GPU models/VRAM, ROCm, framework versions, engine, and current Hermes/tool interface.
- Preserve current API/local Hermes profiles and routes.
- Select one small game genre and one engine as the initial benchmark.

**Exit:** A reproducible environment record and a baseline task set exist.

### Phase 1 — Make a reliable game-production loop

- Use existing Hermes models/tools for planning and code while keeping the current profiles unchanged.
- Add structured task/asset manifests and deterministic build, launch, smoke-test, and screenshot steps.
- Start with one small game (e.g. a simple 2D platformer or top-down arcade game) and require a runnable project.

**Exit:** The system can produce and repair a small game through the existing orchestration, with failures visible in logs/screenshots.

### Phase 2 — Establish a benchmark

- Create a fixed set of prompts covering broad instructions and precise specifications.
- Score completion, build success, required mechanics, asset correctness, runtime defects, repair count, elapsed time, and user intervention.
- Save prompts, outputs, logs, and screenshots so later changes can be compared fairly.

**Exit:** We know which stages consume time and which failures recur.

### Phase 3 — NCM code-model experiment

- Inspect the NCM repository and its actual training/inference interfaces before choosing an implementation path.
- Define a narrow coding task and data format; begin with a small, measurable prototype rather than a costly full training run.
- Compare NCM outputs to the existing baseline on the same code tasks, automated checks, and hardware.
- Only expand training or dataset size when metrics show useful gains.

**Exit:** Evidence that NCM improves at least one defined coding task at an acceptable speed/quality tradeoff, or a clear diagnosis of the bottleneck.

### Phase 4 — Improve specialist handoffs

- Add the Asset Workshop and Visual Reviewer with strict artifact contracts.
- Keep model calls sequential or concurrent according to actual VRAM and speed measurements.
- Add Sound Workshop only when the core game pipeline is dependable.

**Exit:** Assets and visual review improve delivered games without making failures harder to trace.

### Phase 5 — Scale scope carefully

- Add more game genres, larger projects, and longer-running playtests only after small games pass reliably.
- Re-evaluate the hardware bottleneck before buying GPUs or starting long training runs.

**Exit:** Quality, latency, and resource use remain measured as scope grows.

## Initial success criteria

For the first benchmark game, require:

- The project builds from the documented instructions.
- The game launches and supports the required core interactions.
- Required assets exist and load; no missing-reference errors appear.
- Automated checks pass and a screenshot/playtest review finds no blocking defect.
- A failed check produces an actionable repair task and the system can rerun it.
- We record total elapsed time and human interventions, not just generated-code quality.

Opus-level parity is a long-term comparison target, not an assumed result. The first milestone is a dependable, measurable game-making workflow; NCM's role is judged against that workflow's coding baseline.

## GPU memory: current options and decision rule

There is no inexpensive accessory that turns the installed 16 GB cards into one fast 32 GB GPU. External GPU enclosures provide a connection for a separate GPU; they do not add VRAM to an existing card. Multiple cards can run separate models/workloads, while model splitting requires software support and has communication/performance costs.

Potential 32 GB-or-more cards to investigate only if measurements justify an upgrade:

- **Used Radeon PRO W6800 (32 GB):** likely the most practical used AMD route. It is listed in AMD ROCm Linux hardware documentation, but confirm the exact software matrix for the chosen release. A current price tracker shows used U.S. listings from roughly US$915, with other trackers showing higher prices; inspect condition, cooling, power, seller, warranty/returns, and landed Canadian cost.
- **Used AMD Instinct MI50 (32 GB):** sometimes advertised around US$500–$700, but it is an older passive-cooled accelerator with legacy software compatibility concerns. Treat it as a tinkering option only; do not buy it for the main workstation unless we first prove that the intended framework and workload run on it.
- **Radeon PRO W7800 (32 GB):** newer/faster workstation option, but pricing is generally far from “cheap” (Canadian listing found around CAD $2,241; verify stock and final delivered price).
- **GeForce RTX 5090 (32 GB):** consumer card with 32 GB, but current pricing/availability makes it a premium option, and it changes the software path to NVIDIA/CUDA rather than the current AMD-first plan.
- **Radeon RX 7900 XTX (24 GB):** faster/high-end consumer AMD card, but it is 24 GB, not 32 GB. It may be a practical capacity step-up if 24 GB is sufficient and local pricing is favorable.

**Decision rule:** Do not buy hardware just to reach a nominal VRAM number. First profile the actual workload: peak VRAM, model load/inference time, tokens or samples per second, offload, and whether the bottleneck is memory capacity or compute. The extra capacity can enable larger models/context or reduce offload, but it does not by itself make the model more capable or faster.

## Sources for hardware validation

- AMD RX 9060 XT specifications (8 GB and 16 GB variants): https://www.amd.com/en/products/graphics/desktops/radeon.html
- ROCm Radeon/Linux compatibility matrices: https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html
- AMD Radeon PRO W7900 48 GB specifications: https://www.amd.com/en/products/graphics/workstations/radeon-pro/w7900.html
- AMD RX 7900 XTX 24 GB specifications: https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xtx.html
- AMD PRO W6800 used-market price snapshot (variable listings): https://perunitprice.com/gpu/radeon-pro-w6800/
- MI50 current low-end used listing snapshot (check compatibility carefully): https://rackbites.com/gpu/instinct-mi50-32gb
- ROCm Linux GPU support and unsupported-library caveat: https://rocm.docs.amd.com/_/downloads/install-on-linux/en/latest/pdf/
- Canadian W7800 listing (verify current stock/price): https://www.walmart.ca/en/ip/Radeon-Pro-W7800-Professional-Graphics-Card-Workstation-AI-3D-Rendering-32GB-GDDR6/5DBLP04DDCLE

- **Used AMD Instinct MI50 (32 GB):** sometimes advertised at roughly US$500–$700, but it is an older passive-cooled accelerator with legacy software compatibility concerns. Treat it as a tinkering option only; do not buy it for the main workstation unless we first prove that the intended framework and workload run on it.



