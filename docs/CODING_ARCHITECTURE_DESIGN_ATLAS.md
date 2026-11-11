# Coding, architecture, and game design atlas

This is a broad, practical glossary of software design principles, architecture styles, patterns, and game-development ideas. It is meant to be readable as a reference, not memorized as a checklist. There is no finite, authoritative list of *every* principle: names overlap, advice evolves, and many recommendations are context-dependent. Entries below focus on durable, widely used concepts and point to primary or official references.

Each entry gives a plain-language meaning and a compact example. Examples are sketches, not complete production code. “Use” means “consider when the problem fits,” not “apply everywhere.” Architecture is a set of trade-offs; Microsoft's architecture guidance explicitly says no one style fits every situation ([source](https://learn.microsoft.com/en-us/dotnet/architecture/microservices/multi-container-microservice-net-applications/)).

## A quick way to use this atlas

1. State the behavior or quality you need (for example, “a hit should update health and play feedback”).
2. Identify the constraint (frame time, testability, reuse, save compatibility, team size).
3. Pick the smallest idea that addresses it.
4. Check the result with tests, profiling, and playtests.
5. Remove the abstraction if it makes the code harder to understand without a measurable benefit.

## 1. Everyday design principles

| Principle | Plain meaning | Tiny example |
|---|---|---|
| Separation of concerns | Keep different jobs in different places. | `DamageRules` calculates damage; `HealthBar` displays it. |
| Encapsulation / information hiding | Protect internal state behind a small public API. | `Health.ApplyDamage()` clamps health instead of exposing a writable field. |
| Abstraction | Name the important behavior while hiding irrelevant detail. | `ISaveStore.Save()` hides whether storage is a file or cloud service. |
| Modularity | Split a system into parts with clear boundaries. | Input, combat, UI, and persistence are separate modules. |
| High cohesion | A unit's contents belong together. | `Inventory` owns item slots and inventory rules. |
| Low coupling | A change in one unit should require few unrelated edits. | A HUD listens for `HealthChanged`, instead of reaching into player internals. |
| Single responsibility (SRP) | A unit should have one coherent reason to change. | Split file parsing from gameplay validation when they evolve independently. |
| Open/closed (OCP) | Add a new variation through an extension point where that helps. | Add a new `IWeapon` implementation without editing the firing loop. |
| Liskov substitution (LSP) | A subtype must honor the promises of its base type. | Every `IDamageable` implementation accepts the documented damage range. |
| Interface segregation (ISP) | Give callers small interfaces for the jobs they need. | `ISaveReader` is separate from `ISaveWriter` for read-only screens. |
| Dependency inversion (DIP) | Core rules depend on abstractions, not infrastructure details. | `QuestService` uses `IQuestRepository`; a file repository implements it. |
| Dependency injection (DI) | Supply a unit's collaborators from outside. | A constructor receives `ISaveStore` instead of creating a disk store itself. |
| Explicit dependencies | Make required collaborators visible. | `new CombatSystem(clock, random, catalog)` reveals its needs. |
| Composition over inheritance | Combine small behaviors instead of building deep class trees. | A character has `Health`, `Mover`, and `Weapon` components. |
| Program to an interface | Call a stable contract rather than a concrete implementation. | Systems use `IDamageable`, not `Goblin`. |
| DRY (Don't Repeat Yourself) | Keep one authoritative version of knowledge that changes together. | One `ApplyArmor()` rule is shared by player and enemy damage paths. |
| KISS (Keep It Simple) | Prefer the simplest design that meets the actual need. | Use a list for 12 quest steps instead of a generic workflow engine. |
| YAGNI (You Aren't Gonna Need It) | Do not build speculative features before a need exists. | Don't add multiplayer abstractions to a confirmed single-player prototype. |
| Law of Demeter / least knowledge | Ask a close collaborator to do work; avoid traversing its internals. | `player.Inventory.Add(item)` rather than `player.GetBag().Slots.Add(item)`. |
| Least astonishment | Make behavior match what a caller would reasonably expect. | `Remove(item)` returns whether removal happened. |
| Single source of truth | Store each fact once and derive views from it. | Health is stored in the health component; UI reads it rather than maintaining a second copy. |
| Make invalid states unrepresentable | Design types and constructors to rule out impossible states. | `PositiveDamage` can only be constructed with a value above zero. |
| Fail fast | Detect invalid input near its source. | Reject an invalid level ID when loading, not during a later scene transition. |
| Robustness / graceful degradation | Handle expected failures predictably and preserve useful behavior. | Missing optional voice audio logs a warning but doesn't prevent gameplay. |
| Immutability | Prefer values that cannot change after creation when practical. | A `Vector2` value or immutable `WeaponStats` snapshot is passed to a calculation. |
| Locality of reasoning | Keep related data and behavior close enough to understand together. | A cooldown's duration and update rule live in the same component. |
| Explicit over implicit | Make consequential behavior visible in code or data. | Serialize an enum name rather than depending on a fragile numeric position. |
| Optimize for change | Make likely changes local and safe. | Weapon balance values live in resources, not scattered constants. |
| Measure before optimizing | Profile the actual bottleneck. | Pool projectiles only after allocation or GC spikes are observed. |

**SOLID** is the five principles SRP, OCP, LSP, ISP, and DIP. Treat them as design questions, not score targets. A one-method interface for every class or an inheritance hierarchy for every variation can add friction instead of reducing it. Microsoft's .NET guidance describes SOLID as internal design/dependency guidance and DI as one way to apply DIP ([source](https://learn.microsoft.com/en-us/dotnet/architecture/microservices/microservice-ddd-cqrs-patterns/microservice-application-layer-web-api-design)).

## 2. Programming paradigms and ways to structure behavior

| Idea | Meaning | Tiny example |
|---|---|---|
| Object-oriented programming | Model interacting objects that combine state and behavior. | `Door.Open()` updates a door's state and plays its animation. |
| Functional programming | Prefer pure transformations and explicit data flow. | `newHealth = ApplyDamage(oldHealth, damage)` has no hidden side effects. |
| Imperative programming | Describe the steps that change state. | Read input, update velocity, move the character. |
| Declarative programming | Describe the desired result; a system handles mechanics. | A Godot scene/resource declares nodes and properties in editor data. |
| Data-oriented design | Organize data and work around access patterns and throughput. | Process arrays of projectile positions together when profiling shows scale requires it. |
| Event-driven design | React to events without direct caller-to-callee control flow. | `EnemyDefeated` updates score and quest progress through subscribers. |
| Reactive programming | Propagate changing values through a dependency graph or stream. | A UI label updates when an observable health value changes. |
| Actor model | Isolated actors communicate with messages and manage their own state. | Each network session processes its own message queue. |
| Pipeline | Pass data through ordered transformations. | Load save bytes → validate version → migrate → build game state. |
| Functional core, imperative shell | Keep rules as pure functions; isolate I/O and engine effects at the edges. | Pure `CanPurchase()` rule, called by a node that plays sound and saves. |

These paradigms can be mixed. Godot itself uses nodes/scenes, signals, resources, and engine callbacks; choose the paradigm that suits each feature rather than forcing the entire game into one style.

## 3. Architecture styles

| Style | What it organizes | Good fit / caution | Tiny example |
|---|---|---|---|
| Layered / N-tier | UI, application, domain, and infrastructure responsibilities. | Familiar for business apps; avoid pass-through layers with no value. | UI calls `PurchaseService`, which calls a repository. |
| Modular monolith | One deployable application with strong internal module boundaries. | Often a good default for a small game/team; modules can evolve independently. | `Combat`, `World`, and `Save` modules in one Godot project. |
| Microservices | Independently deployed services communicating over a network. | Useful for independently scaling/operating large systems; adds network and operations cost. | Separate account and matchmaking services for a live game. |
| Hexagonal / ports and adapters | Core behavior defines ports; outside systems provide adapters. | Helps replace/test external dependencies. | `ISaveStore` is a port; JSON file storage is an adapter. |
| Clean / Onion architecture | Domain/application rules sit inside; infrastructure depends inward. | Useful when business rules must outlive storage/UI choices; can be excessive for tiny games. | Combat rules don't reference Godot nodes or disk code. |
| MVC | Model, view, controller responsibilities. | Common for screen-driven UI; boundaries vary by framework. | Model holds settings; view draws menu; controller handles button actions. |
| MVP | Presenter coordinates view and model. | Useful when UI behavior needs isolated tests. | `SettingsPresenter` updates an `ISettingsView`. |
| MVVM | View binds to view-model state/commands. | Strong with binding-centric UI frameworks; Godot C# may need explicit signals/binding glue. | A settings view-model exposes `MusicVolume` and `ApplyCommand`. |
| ECS (entity-component-system) | Entities are IDs, components hold data, systems process them. | Useful for many uniform entities; not automatically faster or simpler in every game. | `MovementSystem` processes all entities with `Position` and `Velocity`. |
| Plugin / microkernel | A small core loads optional modules through a stable contract. | Good for editor extensions/mod support; creates versioning and lifecycle obligations. | Game discovers gameplay modifiers implementing `IGamePlugin`. |
| Client-server | Server owns authoritative state; clients request actions and render. | Useful for networked games; requires trust boundaries and latency handling. | Server validates a hit; client predicts movement. |
| Peer-to-peer | Peers exchange state directly. | Can reduce server cost, but complicates trust, NAT, and synchronization. | Deterministic small-game session shares player inputs. |
| Event-driven architecture | Components publish and consume events. | Decouples reactions; global event buses can hide dependencies and ordering. | `LevelCompleted` triggers results UI and unlock save. |
| CQRS | Separate write commands from read/query models. | Useful when read and write needs diverge; usually unnecessary for a small offline game. | `CompleteQuest` command updates state; quest-log query builds display data. |
| Event sourcing | Store state changes as events and rebuild state by replay. | Useful for audit/history/time travel; persistence and migrations become harder. | Save `ItemCollected` events and rebuild inventory. |
| Pipe and filter | Compose independent processing stages. | Good for import/build/content pipelines. | Source files → parser → AST chunks → token encoder → cache. |
| Blackboard | Independent specialists read/write shared working memory. | Useful in some AI/planning systems; shared state needs ownership discipline. | NPC behavior modules add candidate actions to a board. |
| Service-oriented architecture | Expose coarse-grained services over explicit contracts. | Broader service integration style; avoid turning every class into a remote service. | Inventory service API for multiple game clients. |

## 4. Domain modeling and boundaries

| Concept | Meaning | Tiny example |
|---|---|---|
| Domain-Driven Design (DDD) | Shape software around a complex problem domain and its language. | Designers and code both use the term “stamina” with the same rule. |
| Ubiquitous language | Shared terms used by designers, programmers, and tests. | `QuestAccepted` means the same thing in dialogue, code, and test cases. |
| Bounded context | A boundary within which a model and its terms have one meaning. | “Character” in combat differs from “character” in account/profile data. |
| Entity | An object identified by continuity, not just its current values. | A quest remains the same quest as its progress changes. |
| Value object | Immutable value defined by its contents. | `Damage(physical: 10, fire: 3)`. |
| Aggregate / invariant | A consistency boundary that protects rules when state changes. | `Inventory.TryAdd()` enforces capacity and item-stack limits. |
| Repository | Domain-facing interface for retrieving/storing aggregates. | `IQuestRepository.Load(questId)`. |
| Domain service | Domain rule that doesn't naturally belong to one entity/value. | `LineOfSightRules.CanSee(attacker, target, world)`. |
| Domain event | A fact that has already happened in the domain. | `BossDefeated` can unlock a door. |
| Anti-corruption layer | Translate an external model into your own model at a boundary. | Convert a mod's item schema into the game's `ItemDefinition`. |
| Bounded context map | Show relationships and translation between model boundaries. | `SaveFormat v1` is translated into current `InventoryState`. |
| Anemic domain model | Data-only objects while rules live elsewhere. | An `Inventory` with public mutable slots and no capacity enforcement. Sometimes fine for simple DTOs; risky when rules are important. |

DDD is most valuable when domain rules are genuinely complex. Microsoft notes that a simple CRUD application may not need it ([source](https://learn.microsoft.com/en-us/th-th/dotnet/architecture/modern-web-apps-azure/develop-asp-net-core-mvc-apps)).

## 5. Common design patterns

The classic Gang of Four patterns are a vocabulary for recurring object-design problems. The groups below summarize those patterns plus frequently used application and game patterns. A pattern is not a requirement.

### Creational patterns

| Pattern | Meaning | Tiny example |
|---|---|---|
| Factory Method | Let a method choose the concrete product. | `CreateEnemy(kind)` returns the matching enemy scene. |
| Abstract Factory | Create families of compatible objects. | `FantasyUiFactory` creates matching fantasy buttons and panels. |
| Builder | Assemble a complex object through named steps. | `LevelBuilder.AddRoom(...).Connect(...).Build()`. |
| Prototype | Create a new object by copying a configured prototype. | Duplicate a configured projectile resource/scene. |
| Singleton | Provide one globally reachable instance. | A carefully scoped audio settings service. Use sparingly: globals hide dependencies and complicate tests. |
| Object Pool | Reuse objects to avoid repeated creation/destruction costs. | Reuse bullets after profiling shows allocation spikes. |

### Structural patterns

| Pattern | Meaning | Tiny example |
|---|---|---|
| Adapter | Translate one interface into another. | Wrap a platform save API behind `ISaveStore`. |
| Bridge | Separate two dimensions of variation. | `Weapon` delegates firing to a `IFireMode`. |
| Composite | Treat one item and a group uniformly. | A UI container and its child controls both implement `IUiElement`. |
| Decorator | Add behavior by wrapping an object. | `PoisonedWeapon` wraps a weapon and adds damage-over-time. |
| Facade | Offer a simple front to a complicated subsystem. | `Audio.PlayUiConfirm()` hides bus and asset lookup. |
| Flyweight | Share repeated immutable data to reduce memory. | Many enemies reference one shared `EnemyDefinition` resource. |
| Proxy | Stand in for another object and control access. | Lazy-load a large portrait texture when first displayed. |

### Behavioral patterns

| Pattern | Meaning | Tiny example |
|---|---|---|
| Chain of Responsibility | Pass a request through possible handlers. | UI input travels modal → menu → game controls until handled. |
| Command | Represent an action as an object/data value. | `MoveCommand(direction)` supports replay or undo. |
| Interpreter | Evaluate a small language/grammar. | Parse simple dialogue conditions like `has_key && level > 2`. |
| Iterator | Traverse a collection without exposing its internals. | Enumerate active quest objectives. |
| Mediator | Centralize coordination among collaborating peers. | A battle coordinator starts phases and signals actors. |
| Memento | Capture state for later restoration. | Store a checkpoint snapshot for retry. |
| Observer / publish-subscribe | Notify listeners when an event/value changes. | Godot `HealthChanged` signal updates a health bar. |
| State | Change behavior according to an explicit state. | `Idle`, `Chasing`, and `Attacking` enemy states. |
| Strategy | Swap an algorithm behind a stable contract. | Choose pathfinding or targeting strategy by enemy type. |
| Template Method | A base workflow calls customizable steps. | Base importer validates, then subtype parses a format. |
| Visitor | Add operations over a stable object structure. | AST visitor collects identifiers from syntax nodes. |
| Null Object | Provide a harmless implementation instead of null checks. | `SilentAudio` implements `IAudio` for headless tests. |
| Specification | Package a business predicate for reuse/composition. | `CanEquip(item).And(MeetsLevelRequirement)`. |
| Unit of Work | Group related persistence changes as one commit. | Save quest and reward changes together. |
| Service Locator | Resolve services from a global registry. | `Services.Get<ISaveStore>()`; use cautiously because dependencies are hidden. |
| Result / Either | Return success or a typed error explicitly. | `Result<SaveData, SaveError>` distinguishes missing/corrupt/version mismatch. |

### Distributed-system patterns (mostly not for a local single-player prototype)

| Pattern | Meaning | Tiny example |
|---|---|---|
| Retry with backoff | Retry temporary failures after increasing delays. | Retry a matchmaking request after a brief network outage. |
| Timeout | Stop waiting after a bound. | Cancel a cloud save request after 8 seconds. |
| Circuit breaker | Temporarily stop calls to a failing dependency. | Pause repeated requests to an unavailable leaderboard. |
| Bulkhead | Isolate resource pools so one failure doesn't consume everything. | Separate telemetry queue from save writes. |
| Idempotency key | Make repeated requests have one logical effect. | Retried purchase request cannot grant currency twice. |
| Saga | Coordinate a multi-step transaction with compensating actions. | Reserve a hosted session, then allocate a server; release reservation if allocation fails. |
| Transactional outbox | Save state and its outgoing event together before delivery. | Persist score update and leaderboard event in one DB transaction. |
| Cache-aside | Read cache first, load source on miss, then cache. | Load remote player profile once and reuse briefly. |
| Strangler migration | Replace an old system gradually behind a boundary. | Route new save versions to a new save service while old saves remain supported. |

## 6. Reliability, security, and operations

| Principle/pattern | Meaning | Tiny example |
|---|---|---|
| Quality attributes / nonfunctional requirements | State how well the system must behave, not only what it does. | “Load a level within 2 seconds on target hardware.” |
| Quality attribute scenario | Specify stimulus, context, response, and measurable target. | “When a save file is corrupt at startup, show recovery choices within 1 second.” |
| Modifiability tactic | Make a likely change local through encapsulation, interfaces, or limited dependencies. | Swap the audio backend without changing combat code. |
| Reliability | Correct behavior consistently when needed, including failure recovery. | Autosave to a temporary file, validate it, then replace the previous save. |
| Observability | Use logs, metrics, and traces to understand behavior. | Log load duration and save failures with a save-slot ID. |
| Actionable telemetry | Measure a user/system outcome and connect it to a decision. | Track failed level loads, not just raw frame counts. |
| Defense in depth | Use multiple independent protections. | Validate save path, size, version, and parsed values. |
| Least privilege | Give each component only the authority it needs. | A mod can read its own data but cannot overwrite game executables. |
| Secure defaults | Start in the safer configuration. | Network listener is off until explicitly configured. |
| Input validation | Check data at trust boundaries before use. | Reject negative inventory count in a save file. |
| Threat modeling | Identify assets, adversaries, entry points, and mitigations before implementation. | For mods: consider malicious scripts and untrusted archives. |
| Fail closed | On security-sensitive uncertainty, deny the sensitive operation. | Unknown permission means no online account action. |
| Rate limiting | Bound repeated requests. | Limit login attempts or online chat sends. |
| Idempotency | Repeating an operation produces no duplicate effect. | Replayed reward claim does not grant a second reward. |
| Graceful shutdown / recovery | Save or release state predictably on interruption. | Stop accepting new matches, finish or cancel active save writes. |
| Configuration externalization | Separate environment-specific settings from code. | Use a dev endpoint in development and production endpoint in release. |
| Twelve-Factor concepts | A set of practices for deployable services: config, logs, processes, backing services, and more. | Store service URL in configuration, not hard-coded source. Often less relevant to offline game code. |
| Well-Architected pillars | Review operations, security, reliability, performance, cost, and sustainability trade-offs. | Profile load time and memory before adding a cache. |
| Small reversible changes | Keep changes reviewable and recoverable. | Add one persistence migration with a rollback-compatible backup. |

For security guidance, use current primary standards and threat models rather than relying on a glossary. OWASP's secure coding project emphasizes hands-on, language-specific practice ([OWASP Secure Coding](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)); NIST SSDF organizes secure development practices across the lifecycle ([NIST SP 800-218](https://csrc.nist.gov/pubs/sp/800/218/final)). Cloud-specific well-architected frameworks are useful checklists, but their service operations advice does not automatically transfer to a local game ([AWS pillars](https://docs.aws.amazon.com/wellarchitected/latest/framework/definitions.html), [Google Cloud framework](https://docs.cloud.google.com/architecture/framework)).

## 7. Testing and development practices

| Practice | Meaning | Tiny example |
|---|---|---|
| Unit test | Test one small behavior in isolation. | `ClampHealth(-4, 100)` returns zero. |
| Integration test | Test collaborating modules or real adapters together. | Save a quest, reload it, and compare state. |
| End-to-end test | Exercise a user-facing flow through the running application. | Start game → load level → finish objective → see results screen. |
| Contract test | Check that two components honor their agreed interface. | Save adapter returns the versioned format expected by loader. |
| Regression test | Keep a test for a previously fixed defect. | A negative index never crashes inventory lookup. |
| Property-based test | Check general properties across generated inputs. | `Wrap(i,n)` is always in `[0,n)` when `n > 0`. |
| Fuzz test | Feed many malformed/random inputs to find crashes or unsafe behavior. | Fuzz save-file parser with random bytes. |
| Golden / snapshot test | Compare output to a reviewed expected artifact. | A level-generation seed produces a stable map snapshot. |
| Mutation testing | Deliberately alter code to see whether tests detect the change. | Change `<` to `<=`; range tests should fail. |
| TDD | Red test → minimal implementation → refactor. | Write failing damage-boundary test, implement clamp, clean up. |
| Test pyramid | Many small tests, fewer integration tests, a small number of end-to-end tests. | Hundreds of rule tests, some save tests, few full game flows. |
| Static analysis | Detect likely defects/style problems without running the game. | C# analyzer catches an unobserved async task or nullable issue. |
| Code review | Use another person/checklist to catch defects and improve clarity. | Review one focused combat change with tests and rationale. |
| Continuous integration | Build and verify changes automatically on commit/PR. | CI compiles the Godot C# project and runs unit tests. |
| Continuous delivery | Keep the software in a deployable state through automated integration/release steps. | Generate a test build after a green main branch. |
| Refactoring | Change structure without changing externally visible behavior. | Extract duplicated validation while all behavior tests stay green. |
| Trunk-based / small-batch development | Integrate small changes frequently. | Land one tested save migration at a time. |
| Semantic versioning | Communicate API compatibility through version numbers. | Breaking mod API change increments major version. |
| Feature flag | Enable/disable an incomplete behavior without maintaining long-lived branches. | Hide a new inventory UI behind a development toggle. |

Testing principles particularly relevant to this project: evaluate syntax with the compiler, behavior with executable unit/property tests, and player experience with playtests. Compilation alone cannot show that reconstructed code preserves behavior or that a game mechanic is fun.

## 8. Game architecture and player-facing design

| Idea | Meaning | Tiny example |
|---|---|---|
| Scene composition | Build reusable gameplay objects from nested scenes/nodes. | A door scene contains sprite, collision, animation, and interaction script. |
| Self-contained reusable scene | A scene should work without brittle assumptions about its parent. | Door exports a target/event contract instead of reaching for `../../Player`. |
| Signals as observer | Publish events to interested listeners. | Door emits `opened`; HUD or sound reacts. |
| Resources as data | Store reusable authored configuration separately from runtime behavior. | `WeaponDefinition.tres` holds damage, range, and icon. |
| Data-driven design | Represent variations as data instead of duplicated code. | Ten enemy variants share a script and use different stat resources. |
| State machine | Make mutually exclusive behavior states and transitions explicit. | `Patrol → Alert → Chase → Attack`. |
| Fixed-step simulation | Update simulation at a controlled timestep when determinism/physics requires it. | Physics movement runs in `_PhysicsProcess`, rendering in `_Process`. |
| Frame-rate independence | Scale time-based motion by delta or a fixed simulation step. | `position += velocity * delta`. |
| Determinism | Same initial state and inputs produce same simulation result. | Seed random loot generation for reproducible replays. |
| Input abstraction | Translate device input to game actions. | `move_left` action maps keyboard and controller. |
| Feedback loop | Show the player that an action caused a result. | Hit flash, sound, damage number, and enemy reaction. |
| Affordance | Make possible actions understandable from appearance/context. | A highlighted lever looks interactable and has a prompt. |
| Discoverability | Help players find controls, goals, and available actions. | First-use prompt appears near a new mechanic. |
| Onboarding | Teach mechanics in context, in small steps. | Safe room lets player try jump before combining jump with hazards. |
| Game feel | Tune input response, animation, sound, camera, and feedback for satisfying control. | A jump has responsive buffering and a clear landing cue. |
| MDA (Mechanics, Dynamics, Aesthetics) | Connect rules to emergent play and intended experience. | Stamina drain (mechanic) creates cautious pacing (dynamic) for tension (aesthetic). |
| Iterative prototyping | Test a small playable version before investing in content/polish. | Gray-box one combat room to test dodge timing. |
| Player-centered evaluation | Observe target players doing real tasks; revise based on evidence. | Watch whether players understand an objective without verbal hints. |
| Accessibility by design | Offer multiple ways to perceive, understand, and control the game. | Remappable controls, captions, readable contrast, difficulty options. |
| Inclusive defaults | Make common barriers avoidable without special setup where feasible. | Subtitles on by default when dialogue contains required information. |
| Pacing and cognitive load | Manage how much new information and challenge arrive at once. | Introduce one enemy behavior before combining it with a timer. |
| Meaningful choice | Give player decisions understandable consequences. | Spend a limited upgrade resource on either mobility or damage. |
| Consistent rules | Teach a mechanic once and apply it predictably. | The same color consistently signals a harmful surface. |
| Reward clarity | Make progress and reward conditions legible. | Quest UI says what remains and what completion grants. |
| Save compatibility | Treat saves as durable user data with versions and migration plans. | Version 2 loader converts old item IDs before loading inventory. |

Godot's official docs emphasize nodes, scenes, the scene tree, and signals as central concepts ([key concepts](https://docs.godotengine.org/en/latest/getting_started/introduction/key_concepts_overview.html)). Its scene-organization guidance recommends focused, loosely coupled scenes and warns against brittle environmental assumptions ([scene organization](https://docs.godotengine.org/en/stable/tutorials/best_practices/scene_organization.html)). For design analysis, MDA is a useful shared vocabulary, not a formula for fun ([original paper](https://aaai.org/papers/ws04-04-001-mda-a-formal-approach-to-game-design-and-game-research/)).

## 9. Which ideas should enter NCM training?

### First: keep codec and generator goals separate

The active experiment in this repo is a **VQ-VAE codec reconstruction** run. Its input and target are the same code text. Its job is to encode and reconstruct code faithfully. A paragraph explaining SOLID or game design does not teach the codec to produce better architecture; mixing prose into this corpus changes the reconstruction distribution and makes the current experiment harder to interpret.

The later **generator** is where architecture/design instruction belongs. It needs examples with a task, relevant context, a correct implementation, tests, and—when useful—a short rationale. Principles should be taught by concrete decisions and counterexamples, not as a raw vocabulary dump.

### High priority for Godot C# generator examples

| Priority | Teach through examples | What the training example should show |
|---|---|---|
| Essential | C# correctness, Godot 4 C# API/version fit, null/edge handling, clear naming | A user request, compiling C# solution, and checks for boundary cases. |
| Essential | Cohesion, encapsulation, composition, explicit ownership | A focused node/component with clear exported dependencies and no brittle parent-path assumptions. |
| Essential | Signals/events and direct calls used appropriately | A local direct call for a required collaborator; a signal for independent reactions. |
| Essential | State machines and frame-rate-aware game-loop behavior | State transitions, `_PhysicsProcess` vs `_Process`, delta handling, and tests. |
| Essential | Data-driven variation and Godot Resources/scenes | New enemy/weapon variation implemented by data where appropriate, not copy-pasted scripts. |
| Essential | Tests at multiple levels | Unit/property tests for rules, integration tests for save/scene interaction, compiler/build check. |
| Essential | Player feedback, onboarding, accessibility | Requested feature includes response cues, understandable controls, and accessible alternatives. |
| High | DRY, KISS, YAGNI, SOLID, DI | Show both a good fit and an over-engineered counterexample; explain the trade-off. |
| High | Save versioning, errors, observability, secure boundaries | Handle corrupt/old saves and untrusted input predictably, with diagnostic detail. |
| Conditional | DDD, bounded contexts, clean/hexagonal architecture | Use only in tasks with meaningful domain complexity or replaceable infrastructure. |
| Conditional | ECS/data-oriented design, pooling, caching | Tie to entity scale or measured performance constraints; include profiling evidence. |
| Conditional | Networking, client/server authority, deterministic simulation | Include only when the prompt asks for multiplayer, replay, or synchronization. |
| Low for ordinary small games | Microservices, event sourcing, CQRS, sagas, service locator | Teach as recognition/trade-off material; don't default to them for a local game. |

### A useful generator-data record

Each training/evaluation case should carry structured fields such as:

```json
{
  "task": "Add a reusable health component with damage and death feedback.",
  "constraints": ["Godot 4 C#", "scene can be reused", "negative damage is ignored"],
  "context": "Existing node exposes a HealthChanged signal.",
  "solution": "...complete C# implementation...",
  "tests": ["negative damage", "lethal damage", "health never exceeds maximum"],
  "principles": ["encapsulation", "composition", "signals", "boundary testing"],
  "rationale": "The component owns health rules; listeners react without reaching into its state."
}
```

Include paired examples where the same principle is helpful in one context and harmful in another. Example: a signal is appropriate for optional UI/audio reactions; a direct call may be clearer for a required, immediate dependency. This prevents “always use pattern X” behavior.

### What the current codec results imply

Prior stress examples show small token mistakes can break identifiers, operators, or expressions. The current follow-up plan—codebook-capacity ablation, then occurrence-aware weighting and AST-aligned segmentation—is aimed at reconstruction. These are sensible codec experiments, but architecture principles will not fix a token that the codec changes. The 4,096-entry result must be measured before claiming capacity solved the problem.

The pointer/copy idea is most appropriate for the later generator when a prompt or retrieved code provides names/literals to copy. Adding access to original source tokens inside the codec decoder would bypass the representation under test. The separate codec plan documents this distinction.

### Will this make NCM faster or cheaper than an LLM?

Architecture/design data alone does not determine speed or token cost. Generator training changes quality and behavior; tokenizer/model architecture, context length, inference implementation, hardware, batching, and output length determine runtime and token throughput. A compact model can be faster and cheaper for a narrow task, but the project must benchmark end-to-end latency, memory, tokenization, and compile/behavioral success against an LLM baseline. Better compilation accuracy is not guaranteed by a codec change or by adding principles to data.

## 10. Suggested training sequence for these principles

1. Finish and evaluate the active 4,096-codebook codec experiment without changing its inputs mid-run.
2. Stabilize a codec using held-out reconstruction, compiler checks, and behavioral tests; keep source-code reconstruction data code-only.
3. Build a separate instruction-target generator dataset with correct Godot C# tasks, runnable tests, and selected principle tags/rationales.
4. Evaluate general implementation correctness, Godot API/version correctness, compiler success, behavior, architecture quality, and player-facing design separately.
5. Add principle-specific challenge sets: duplicated logic, brittle node paths, excessive global state, bad frame-time handling, inaccessible controls, and over-engineering traps.
6. Use retrieval for long reference material and version-specific API documentation; don't expect model weights to retain every API detail perfectly.

## References and further reading

- [Microsoft: Architectural principles for .NET applications](https://learn.microsoft.com/en-us/dotnet/architecture/modern-web-apps-azure/architectural-principles)
- [Microsoft: Common web application architectures](https://learn.microsoft.com/en-us/dotnet/architecture/modern-web-apps-azure/common-web-application-architectures)
- [Microsoft: .NET microservice architecture and pattern trade-offs](https://learn.microsoft.com/en-us/dotnet/architecture/microservices/multi-container-microservice-net-applications/)
- [Microsoft: Dependency injection in .NET](https://learn.microsoft.com/en-us/dotnet/core/extensions/dependency-injection)
- [Martin Fowler: Patterns of Enterprise Application Architecture catalog](https://martinfowler.com/eaaCatalog/)
- [Martin Fowler: Event Sourcing](https://martinfowler.com/eaaDev/EventSourcing.html)
- [Martin Fowler: Inversion of Control Containers and Dependency Injection](https://martinfowler.com/articles/injection.html)
- [SEI: Modifiability tactics](https://www.sei.cmu.edu/library/modifiability-tactics/)
- [SEI: Software architecture concepts](https://www.sei.cmu.edu/our-work/software-architecture/)
- [AWS Well-Architected Framework](https://docs.aws.amazon.com/wellarchitected/latest/framework/definitions.html)
- [Google Cloud Well-Architected Framework](https://docs.cloud.google.com/architecture/framework)
- [NIST Secure Software Development Framework (SSDF)](https://csrc.nist.gov/pubs/sp/800/218/final)
- [OWASP Secure Coding Practices Quick Reference Guide](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)
- [Godot: Nodes, scenes, scene tree, and signals](https://docs.godotengine.org/en/latest/getting_started/introduction/key_concepts_overview.html)
- [Godot: Scene organization](https://docs.godotengine.org/en/stable/tutorials/best_practices/scene_organization.html)
- [Godot: Best practices index](https://docs.godotengine.org/en/stable/tutorials/best_practices/index.html)
- [Robert Nystrom: Game Programming Patterns](https://gameprogrammingpatterns.com/)
- [Hunicke, LeBlanc, Zubek: MDA framework paper](https://aaai.org/papers/ws04-04-001-mda-a-formal-approach-to-game-design-and-game-research/)
- [Nielsen Norman Group: 10 usability heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/)
- [Game Accessibility Guidelines](https://gameaccessibilityguidelines.com/)
- [Xbox Accessibility Guidelines](https://learn.microsoft.com/en-us/gaming/accessibility/guidelines)

## Existing local companions

- [Godot/C# reference set](GAME_CODING_REFERENCE.md) — pinned documentation and courses intended to be available locally.
- [Game design and architecture checklist](GAME_DESIGN_AND_ARCHITECTURE.md) — project questions for code structure, player experience, accessibility, and review.
- [DeepSeek codec plan](DEEPSEEK_NCM_CODEC_PLAN.md) — current codec evidence, experiment order, and pointer/copy analysis.
