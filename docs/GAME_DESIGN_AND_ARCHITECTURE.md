# Game design, architecture, and player-quality playbook

This document is an original, practical checklist for the game-building workflow. It is meant to guide planning, implementation, and review; it is not a universal formula for making a game fun. The agent should treat design principles as hypotheses to test with a playable build and player feedback.

## Use the references for different jobs

- **Godot documentation:** engine-specific choices such as scene boundaries, resources, signals, input, UI, performance, and project organization. The local Godot reference checkout includes the `tutorials/best_practices/` and engine documentation sections.
- **.NET documentation:** C# language conventions, architecture concepts, testing, diagnostics, and maintainability. Prefer the current official docs, then apply the subset that fits the game.
- **OpenGDD:** a structured format for expressing a game design so goals, systems, content, and implementation are reviewable. Use it as a document structure, not a source of universal design rules. [Specification](https://opengdd.org/)
- **Xbox Accessibility Guidelines:** useful prompts for accessible text, contrast, audio/visual cues, input remapping, difficulty, object clarity, UI navigation, motion, and photosensitivity. Treat the guidelines as design and test ideas, not a certification claim. [Guidelines](https://learn.microsoft.com/en-us/gaming/accessibility/guidelines)
- **Ian Schreiber's Game Design Concepts:** a free, openly licensed course on analyzing games, prototyping, mechanics, feedback, balance, and player experience. It is broad foundational material; adapt its exercises to digital games and validate ideas with playtests. [Course](https://gamedesignconcepts.wordpress.com/)
- **MDA framework:** mechanics produce dynamics that are experienced as aesthetics; use this vocabulary to connect implementation choices to player experience. It is an analysis framework, not a formula for fun. [Paper](https://aaai.org/papers/ws04-04-001-mda-a-formal-approach-to-game-design-and-game-research/)
- **Google Engineering Practices:** checklists for making changes understandable and reviewable. The repository is archived upstream; its documents are CC BY 3.0 and are synced locally with attribution. [Guides](https://github.com/google/eng-practices)
- **Usability and accessibility:** Nielsen Norman Group's ten usability heuristics, Game Accessibility Guidelines, and Xbox Accessibility Guidelines are useful inspection prompts. [NN/g heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/) · [Game Accessibility Guidelines](https://gameaccessibilityguidelines.com/) · [Xbox guidelines](https://learn.microsoft.com/en-us/gaming/accessibility/guidelines)

Godot documentation, .NET documentation, Google Engineering Practices, and the openly licensed Game Design Concepts course are synced by `scripts/sync_game_reference.sh` (or its PowerShell equivalent). Other resources remain linked rather than copied because their reuse terms are unclear or not verified. The local sources are reference material; this playbook is the portable checklist agents should apply to each project.

## Before implementation: make the game legible

For a broad request, first draft a short design brief and ask the user to approve or revise it when a key choice would materially change the game. For a specific request, preserve the user's choices and fill in only the missing details.

Record:

1. **Player and promise:** who the game is for, what experience it promises, and the one-sentence hook.
2. **Design pillars:** three to five qualities that should be visible in play (for example, readable tactics, short runs, expressive movement). Use them to resolve scope conflicts.
3. **Core loop:** what the player repeatedly does, what feedback follows, and why the next repetition is worthwhile.
4. **Controls and feedback:** input, resulting action, animation/audio/UI feedback, and how the player learns whether it worked.
5. **Goal, stakes, and progression:** what counts as success or failure, what changes over time, and how the game avoids unexplained dead ends.
6. **Content and style:** art direction, sound direction, tone, level/content plan, and explicit asset sources or generation approach.
7. **Scope and platform:** target devices, resolution/input assumptions, session length, performance budget, and what is deliberately out of scope.
8. **Acceptance checks:** observable conditions that can be tested. Replace “fun,” “polished,” or “intuitive” with concrete checks wherever possible.

For an uncertain idea, build one small playable vertical slice containing the main loop, one complete success/failure path, representative feedback, and a restart. Test that slice before producing lots of levels or assets.

## Player experience checks

Review the playable build for:

- The player can identify the immediate goal, available actions, and important state without guessing.
- Controls respond consistently; important actions have clear visual, audio, or haptic feedback where available.
- New mechanics are introduced when the player can use them, with a safe opportunity to understand them.
- Difficulty grows from the intended skill or decision, rather than surprise, unreadable effects, or arbitrary punishment.
- Menus, pause, restart, save/load, and failure recovery behave predictably.
- Text remains readable; color is not the only way to convey critical meaning; important audio cues have visual alternatives where practical.
- Offer remapping, adjustable difficulty, subtitles/captions, motion controls, and timing options when relevant to the design. Explain tradeoffs if an option changes the intended challenge.
- Effects, animation, camera motion, and sound reinforce the action and remain configurable when they can create discomfort or block information.
- The game has a coherent visual hierarchy and audio mix: important elements are easy to spot and hear, and decorative effects do not obscure play.

These are review prompts, not pass/fail laws. Accessibility requirements vary by game and player; validate important choices with people who use the relevant features when possible.

## Architecture and implementation checks

- Start with the smallest structure that supports the design. Avoid adding patterns or layers without a concrete change they make easier.
- Keep scene responsibilities understandable. Prefer composed scenes and explicit dependencies over one scene or global singleton owning the whole game.
- Put tunable content in resources/data when designers need to iterate on it; keep behavior in code where it is easier to validate.
- Keep game rules testable independently where practical; keep engine-facing nodes responsible for adapting input, physics, visuals, and audio to those rules.
- Use signals/events for meaningful boundaries and avoid hidden coupling or broad global state.
- Make important state transitions explicit: title → gameplay → pause → win/loss → restart or next level.
- Make save data versionable and handle missing/invalid data with a recoverable result.
- Keep assets and code provenance in a manifest. Do not silently include art, music, fonts, or sound with unknown rights.
- Optimize after measuring a real bottleneck. Track target frame rate, load time, memory, and worst-case scene behavior for the selected hardware.
- Build and run the project after each coherent milestone. Add deterministic tests for rules and state transitions; use headless Godot import/parse checks plus a human play pass for experience issues.

## Agent workflow for a complete game request

1. **Designer/planner:** turn the request into a brief, identify unknowns, establish pillars and acceptance checks, and define a small vertical slice.
2. **Architect:** map scenes, scripts, data resources, inputs, state transitions, and file ownership to those requirements.
3. **Implementer:** produce the project in small buildable increments and keep a manifest of files/assets.
4. **Verifier:** restore/build, import/run headlessly, run rule tests, inspect logs, and report failures with reproduction steps.
5. **Play reviewer:** launch the game and assess whether players understand goals, controls, feedback, pacing, readability, and recovery. Record observations and concrete changes.
6. **Repair loop:** fix the highest-impact verified issue, rerun checks, and retain a working build.

The design and play-review roles can be Hermes agents, an LLM, or a human. The NCM generator should only be assigned a role its context and evaluation support. Asset creation is a separate capability and requires its own model/tool and rights tracking.

## Evaluate evidence, not just prose or loss

Report separately:

- **Build integrity:** project files valid, C# build passes, Godot imports without errors.
- **Requirement coverage:** each acceptance check passes, with evidence or an explicit gap.
- **Runtime behavior:** start, controls, core loop, win/loss, pause, restart, and save/load work as specified.
- **Player experience:** brief play-test observations about clarity, responsiveness, meaningful choices, pacing, accessibility, and visual/audio hierarchy.
- **Engineering quality:** architecture remains understandable, tests cover important rules, and no high-impact errors remain in logs.

Do not collapse these into one score until there is a calibrated benchmark. A project can compile and still be confusing or dull; a subjective “looks fun” judgment is not a substitute for trying it with players.
