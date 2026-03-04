# Bridge AI Research Plan (Monolithic Self-Play)

Status: `implementation complete; research execution is now initialized`
Owner: Kevin (primary), Codex (execution)  
Last updated: 2026-03-04

## Rule
- After each completed step or any planning revision, update this document before moving to the next major task.
- Keep this document as the implementation control plane; any module-level decision should be traceable here.
- For experimental methodology, run hypotheses, and result tracking, use `/Users/kevin/projects/bridge_ai/RESEARCH_PLAN.md`.
- After running experiments and logging outcomes or conclusions, update `RESEARCH_PLAN.md` immediately.
- Keep Bazel binaries invocation-compatible with the documented flags:
  - `bazel run //:selfplay -- --config-path=...`
  - `bazel run //:train -- --config-path=...`
  - `bazel run //:eval -- --config-path=...`
  - `bazel run //:pipeline -- --config-path=...`
  - `bazel run //:smoke -- --config-path=... --manifest-path=...`
  - `bazel run //:manifest_check -- --manifest-path=...`

## Project Objective

Build a bridge AI research stack with:
- one monolithic transformer policy/value model,
- pure self-play and search as the primary strength driver,
- uncertainty-aware play using determinization (information-set style),
- an opinionated but low-maintenance infrastructure path on Modal,
- and an initial UI to inspect sample games and model decisions.

## Milestone 1 — Foundation and project skeleton

### Planned modules (implemented now)
- `pyproject.toml`
  - packaging metadata and dependency groups.
- `requirements.txt`
  - practical dependency pin baseline.
- `tests/test_bridge_env.py`
  - deterministic bridge rule fixtures for legality and scoring sanity checks.
- `.bazelrc`
  - Bazel defaults for local runs and failure diagnostics.
- `WORKSPACE`
  - Bazel workspace declaration for non-pyproject execution.
- `BUILD.bazel`
  - Bazel targets for self-play, training, evaluation, and UI entrypoints.
- `AGENTS.md`
  - process constraints and mandatory plan-updating rule.
- `PLAN.md`
  - this document.
- `src/bridge_ai/common/types.py`
  - canonical enums and domain objects (`Card`, `Suit`, `Rank`, `Seat`, `Phase`, etc.).
- `src/bridge_ai/common/cards.py`
  - deck construction and shuffle/deal helpers.
- `src/bridge_ai/common/actions.py`
  - action encoding for a unified action space.
- `src/bridge_ai/common/state.py`
  - immutable state dataclasses used by env, model, search, and UI.
- `src/bridge_ai/env/bridge_env.py`
  - rule-engine shell for bridge game progression (auction → play), legal action masks, and scoring hooks.
- `src/bridge_ai/models/monolithic_transformer.py`
  - single neural trunk + phase-aware action/value heads.
- `src/bridge_ai/search/ismcts.py`
  - imperfect-information MCTS facade with determinization hooks.
- `src/bridge_ai/data/buffer.py`
  - replay buffer interface and serialization format.
- `src/bridge_ai/data/lin_parser.py`
  - LIN decoding + strict replay validation helpers for real-record ingestion.
- `src/bridge_ai/selfplay/runner.py`
  - self-play worker and trajectory generation entrypoints.
- `src/bridge_ai/training/train_loop.py`
  - training loop and checkpoint management.
- `src/bridge_ai/eval/evaluator.py`
  - fixed-deal evaluator and ladder-match harness stubs.
- `src/bridge_ai/infra/modal_app.py`
  - Modal job/function wiring abstraction.
- `src/bridge_ai/ui/streamlit_app.py`
  - minimal viewer for sample game logs and model action traces.
- `src/bridge_ai/infra/experiment_runner.py`
  - one-command smoke orchestration (`selfplay`, `train`, `eval`) runner.
- `src/bridge_ai/infra/manifest_checker.py`
  - manifest consistency entrypoint for reproducibility checks.
- `src/bridge_ai/infra/pipeline.py`
  - iterative pipeline orchestrator that chains self-play, training, and evaluation.
- `.gitignore`
  - ignore artifacts and checkpoints.

### Current Implementation Status
- [x] Create repository-level plan/agent instructions.
- [x] Scaffold package modules and interfaces.
- [x] Implement baseline project structure, model/env/search/self-play/eval/trainer placeholders.
- [x] Add Bazel build/runtime scaffolding for repository execution.
- [x] Fill bridge rule legality and scoring (auction + play mechanics, trick winner, passout handling, contract scoring baseline).
- [x] Implement full tokenization flow for replay ingestion through `BridgeInputEncoder.encode_dict`.
- [x] Implement transformer internals updates including padded-sequence masking in the encoder forward path.
- [x] Implement determinization + IS-MCTS rollout integration.
- [x] Wire baseline infrastructure entrypoints (Modal functions for self-play/train/eval and checkpoint output path updates).
- [x] Add richer replay viewer with step diagnostics and per-move action-confidence display.
- [x] Add experiment/version manifest and artifact catalog path (`artifacts/manifest.json`) with run writes from selfplay/train/eval.
- [x] Add manifest reproducibility metadata and frozen config snapshot writes.
- [x] Add deterministic bridge-env regression test suite and Bazel test target.
- [x] Add real-bridge LIN replay test fixture coverage in `tests/test_bridge_env.py` for end-to-end legal-action replay validation.
- [x] Add manifest reproducibility validation helper (`validate_manifest`) for run auditability.
- [x] Add deterministic checkpoint baseline comparison path in evaluator (`baseline_checkpoint`, delta metrics).
- [x] Add dedicated research planning artifact (`RESEARCH_PLAN.md`) for experiment design, execution log, and conclusions.
- [x] Add Bazel smoke + manifest check entrypoints:
  - `//:smoke`
  - `//:manifest_check`
- [x] Add explicit baseline checkpoint field to default eval config (`configs/default.yaml`).
- [x] Add Bazel-binary CLI overrides (`--config-path`, `--manifest-path`) for `selfplay`, `train`, `eval`, `smoke`, and `manifest_check`.
- [x] Fix training forward-call signature mismatch in `train_loop.py` (`legal_action_mask`).
- [x] Update evaluator search step to use model-guided action rollouts through `ISMCTS`.
- [x] Fix `//:ui` runtime launch path to start Streamlit server mode under Bazel and recursively discover replay files in subdirectories.
- [x] Add dedicated UI regression tests and Bazel test target (`//:test_ui`) for replay discovery and launcher guard behavior.

### Execution Log
- 2026-03-01: Initial research scaffolding committed.
- 2026-03-01: `AGENTS.md` updated with requirement: `PLAN.md` must be updated after each completed/revised step.
- 2026-03-01: Added module-by-module scaffold for monolithic self-play bridge AI, including:
  - common abstractions (`Card`, `Action`, phase/state types),
  - `BridgeEnv` with phase-aware transitions,
  - monolithic transformer model skeleton,
  - IS-MCTS abstraction,
  - replay buffer,
  - self-play, training, evaluation, Modal, and UI starters,
  - baseline YAML configs and packaging metadata.
- 2026-03-01: Added Bazel execution scaffolding:
  - `WORKSPACE`,
  - `BUILD.bazel` with `//:selfplay`, `//:train`, `//:eval`, `//:ui` targets,
  - updated AGENTS runtime policy to require Bazel entrypoints.
- 2026-03-01: Added `.bazelrc` for local Bazel defaults and developer ergonomics.
- 2026-03-01: Reworked `src/bridge_ai/env/bridge_env.py` with fuller bridge auction/play legality,
  contract finalization, trick winner resolution, and baseline scoring logic for made/failed contracts.
- 2026-03-01: Completed replay-to-encoder integration by wiring `BridgeInputEncoder.encode_dict` and training tokenization.
- 2026-03-01: Added padded-sequence masking in `BridgeMonolithTransformer.forward`, and expanded serialized state payloads
  (e.g., declarer/dummy and doubled level) for training/eval replay conversion.
- 2026-03-01: Added determinization-aware root rollout logic in `src/bridge_ai/search/ismcts.py` with model-guided policy rollouts and sampled hidden-card completion.
- 2026-03-01: Expanded `src/bridge_ai/infra/modal_app.py` with separate Modal worker functions for self-play/training/eval orchestration.
- 2026-03-01: Enhanced `src/bridge_ai/ui/streamlit_app.py` with richer replay diagnostics (phase, action decoding, action probability/value columns, raw row toggle).
- 2026-03-01: Added lightweight experiment manifest logging (`src/bridge_ai/data/manifest.py`) and wired self-play/train/eval to append entries to `artifacts/manifest.json`.

- 2026-03-01: Added run reproducibility metadata and per-run frozen config snapshots (`run_signature`, `config_snapshot`).
- 2026-03-01: Added deterministic bridge-env regression tests (`tests/test_bridge_env.py`) for passout/follow-suit/trick winner/final score checks.
- 2026-03-03: Added real-game LIN replay validation test harness in `tests/test_bridge_env.py` for deterministic legality checks over sampled BBO-formatted records.
- 2026-03-04: Centralized LIN ingest/replay validation in `src/bridge_ai/data/lin_parser.py` and reused it in `tests/test_bridge_env.py` for strict full/partial real-record checks.
- 2026-03-04: Added a curated 20-entry real-world LIN fixture under `tests/fixtures/real_lin_records.txt`, normalized fixture formatting, and fixed `src/bridge_ai/data/lin_parser.py` to support explicit 4-hand `md` records.
- 2026-03-01: Added Bazel test target `//:test_env_rules` in `BUILD.bazel`.
- 2026-03-01: Added manifest validation helper (`validate_manifest`, `validate_manifest_entry`) and coverage for the behavior in `tests/test_bridge_env.py`.
- 2026-03-01: Added deterministic fixed-seed evaluator path with optional baseline checkpoint head-to-head metrics.
- 2026-03-01: Added `//:smoke` and `//:manifest_check` Bazel entrypoints for experiment execution and reproducibility checks.
- 2026-03-01: Added `baseline_checkpoint` to `configs/default.yaml`.
- 2026-03-01: Added argparse-compatible overrides to all Bazel runtime entrypoints:
  - `selfplay`, `train`, `eval` (`--config-path`),
  - `smoke` (`--config-path`, `--manifest-path`),
  - `manifest_check` (`--manifest-path`).
- 2026-03-01: Fixed training loop forward call keyword mismatch in `train_loop.py` (`legal_action_mask` vs `legal_mask`).
- 2026-03-01: Updated evaluator to use model-guided ISMCTS during play (`search.select_action(..., model=model)`).
- 2026-03-01: Updated `AGENTS.md` with `//:smoke` / `//:manifest_check` execution guidance.
- 2026-03-01: Added `RESEARCH_PLAN.md` for hypothesis tracking, run protocols, experiment ledgers, and decision criteria.
- 2026-03-01: Updated AGENTS governance to require immediate `RESEARCH_PLAN.md` updates after experiments/conclusions.
- 2026-03-01: Synchronized `PLAN.md` with completed pipeline-first implementation:
  - added `//:pipeline` orchestration module to implementation inventory;
  - marked Milestone 2/3/4 task execution states based on current code; 
  - added iterative `//:pipeline` in next-action checklist and made Bazel invocation docs reflect `--config-path` conventions.
- 2026-03-01: Expanded Streamlit UI controls for replay filtering and comparisons:
  - added seed/variant/determinization selectors for replay discovery,
  - added optional baseline replay comparison view (contract/result/seed),
  - surfaced determinization metadata in per-game and per-move diagnostics.
- 2026-03-01: Completed optional model auxiliary-path plumbing:
  - added `model.use_auxiliary_heads` option to config and model config dataclass,
  - added optional `return_aux` output mode with `trick_share` and `contract_level_logits`.
- 2026-03-01: Added training online-refresh hook:
  - added `training.online_refresh` and `training.online_refresh_every`,
  - `train()` now optionally runs self-play before each configured iteration before reloading replay.
- 2026-03-01: Added determinization count into replay transition metadata to support replay-level experiment slicing.
- 2026-03-01: Updated `RESEARCH_PLAN.md` to include concrete research run templates, reproducibility gates, and completed tiny smoke ledger.
- 2026-03-01: Added `configs/smoke.yaml` for immediate one-episode, low-latency smoke execution.
- 2026-03-01: Updated `.gitignore` to ignore Bazel/output artifacts (`bazel-*`, `artifacts/`) by default.
- 2026-03-02: Added GitHub Actions CI workflow (`.github/workflows/ci.yml`) to run `bazel test //:test_env_rules`, `bazel run //:smoke` with smoke manifest, and `bazel run //:manifest_check` on push/PR.
- 2026-03-04: Fixed Streamlit UI launch behavior in `src/bridge_ai/ui/streamlit_app.py` so `bazel run //:ui` starts Streamlit server mode (in-process bootstrap) instead of remaining in bare mode, and updated replay loading to recursively index nested `replays/**.json` artifacts by relative path.
- 2026-03-04: Added `tests/test_ui.py` with launcher/replay-discovery regressions and registered Bazel target `//:test_ui` in `BUILD.bazel`.
- 2026-03-04: Fixed Linux CI Bazel smoke-analysis failure (`@@rules_python++pip+pypi//watchdog` missing) by adding explicit `watchdog` dependency in `requirements.txt` and pinning it in `requirements_lock.txt`.

- 2026-03-01: Fixed training checkpoint resume semantics in `src/bridge_ai/training/train_loop.py` to persist and restore `iteration` from checkpoints, enabling true continuation across process restarts (weight-only warm starts now resume from the saved iteration).
- 2026-03-01: Added configurable `training.checkpoint_every` in `src/bridge_ai/training/train_loop.py` and periodic checkpoint persistence at runtime every N iterations.
- 2026-03-01: Removed legacy local test configs and started a fresh namespace with `configs/local_real_scale2000.yaml` for clean 2000-iteration experiments.
- 2026-03-01: Completed a fresh `bazel run //:pipeline -- --config-path=configs/local_real_scale2000.yaml` stretch to full `iterations=2000` with `training.checkpoint_every: 100`.
- 2026-03-01: Confirmed `local_real_scale2000` end-to-end manifest integrity:
  - `selfplay` + `train` + `eval` all `status=ok`
  - train final `loss=0.03137740562669933` at `iteration: 1999`
  - `eval mean_score=-187.5`, `win_rate_vs_zero=0.0`, `score_std=188.33148966649205`
  - `bazel run //:manifest_check -- --manifest-path=artifacts/local_real_scale2000/manifest.json` returned `manifest_issues=[]`.

### Milestone 1b — Bazel-first execution

- [x] Add `WORKSPACE` and `BUILD.bazel`.
- [x] Add Bazel binaries:
  - `//:selfplay`
  - `//:train`
  - `//:eval`
  - `//:ui`
- [x] Add Bazel regression target:
  - `//:test_env_rules`
- [x] Add explicit AGENTS constraint to avoid direct `python` execution.

## Milestone 2 — Model and algorithm implementation

### Core modeling choices
- Single monolithic transformer with a shared encoding trunk.
- Phase-conditioned behavior via tokenized phase input (`auction`, `lead`, `play`, `defense`).
- Shared policy/value heads for all phases with strict phase/legal masking.
- Replay-based bootstrap from self-play only after warm stability checks.

### Planned implementation tasks
- [x] Build full tokenization schema:
  - ownership visibility tokens
  - auction history tokens
  - dummy/public cards
  - trick history and current-trick cards
  - vulnerability and scoring context fields
- [x] Implement policy head with legal-action masking.
- [x] Implement value head that predicts a normalized final outcome.
- [x] Add trajectory storage of `(state, policy_target, value_target, metadata)`.
- [x] Add optional auxiliary heads (toggle via config) and optional forward return path for auxiliary outputs.
- [x] Implement train loop:
  - sample self-play trajectories
  - store `(state, policy_target, value_target, metadata)`
  - optimize policy/value losses
  - checkpoint management and resume

## Milestone 3 — Self-play stack and ISMCTS

### Key algorithmic requirements
- [x] Determinization:
  - sample hidden states from information set
  - run search per sample
  - aggregate action statistics at infoset root
- [x] Search:
  - PUCT-like value backup adapted for sampled game branches
  - partner-as-shared-utility updates at pair level where available in environment contract
  - configurable node caps for latency control (`search_config` and model-guided rollouts)
- [x] Output target policy:
  - visit distribution from search, with configurable fallback to model policy
- [x] Actor loop:
  - supports seed schedules and multiple determinization samples per move
- [x] Learner loop:
  - added `training.online_refresh` and `training.online_refresh_every` controls for optional self-play-to-trainer refresh per train iteration.
- [x] Evaluator loop:
  - fixed-seed evaluation vs baseline checkpoints

## Milestone 4 — Modal-first infrastructure

### Infrastructure target
- Minimize local hardware management by running:
  - actor fleets,
  - trainer jobs,
  - evaluator jobs
  as Modal functions in GPU/CPU-appropriate containers.

### Implementation tasks
- [x] Define separate job entrypoints for:
  - actor generation,
  - checkpoint training,
  - evaluation.
- [x] Add manifest-driven run config:
  - `configs/default.yaml`,
  - `configs/local.yaml`/`configs/modal.yaml`.
- [x] Standard artifact layout:
  - `checkpoints/`,
  - `replays/`,
  - `evaluation/`,
  - `ui/` snapshots.
- [x] Add iterative pipeline entrypoint to run full actor/trainer/eval cycles.
- [x] Add `configs/local.yaml` / `configs/modal.yaml` specialization files if/when Modal deployment is enabled.

## Milestone 5 — UI and analysis

### UI requirements (MVP)
- [x] Show one complete sample game:
  - deal, auction, play-by-play, contract/result.
- [x] Show model outputs per decision:
  - top-k legal actions,
  - action probabilities,
  - predicted value.
- [x] Provide comparison controls:
  - current checkpoint vs baseline checkpoint,
  - determinization count display,
  - random seed and variant selectors.
- [x] Add baseline diff view using paired replay selection and result/contract summary.

## Implementation checkpoints and next immediate actions

1. Run Bazel test smoke:
   - `bazel test //:test_env_rules`
   - `bazel test //:test_ui`
2. Run baseline reproducibility checks:
   - `bazel run //:manifest_check`
   - use explicit per-run manifest path, e.g. `artifacts/baseline/manifest.json`
   - confirm stable `run_signature` for fixed config,
   - confirm `config_snapshot` file exists and matches run config.
3. Run one pipeline iteration end-to-end:
   - `bazel run //:pipeline`
4. Run one full experiment batch:
   - `bazel run //:selfplay -- --config-path=configs/smoke.yaml`
   - `bazel run //:train -- --config-path=configs/smoke.yaml`
   - `bazel run //:eval -- --config-path=configs/smoke.yaml`
   - `bazel run //:ui`
5. Inspect manifest + evaluator outputs for trend signals and regressions.
6. Add experiment and ablation entries to `RESEARCH_PLAN.md` after each run.
7. Shift pipeline execution to Modal-backed runs:
   - validate `configs/modal.yaml` against local baseline behavior
   - run `bazel run //:selfplay`, `bazel run //:train`, `bazel run //:eval` in Modal-compatible mode from the same config namespace
   - keep `artifacts`, `replays`, and `checkpoints` namespaces separated per sweep for reproducibility.
8. Track modal scaling outcomes in `RESEARCH_PLAN.md`:
   - evaluate stability, cost, and throughput deltas versus local baseline.
   - only expand workload size after manifest-validated baseline parity.
9. Keep this section as runtime gating; local warm-up experiments are complete, next milestone is Modal transfer and ops hardening.

## Non-functional requirements

- Reproducibility:
  - deterministic seeding by config,
  - manifest and checkpoint hashes.
- Scalability:
  - batch-friendly tensor outputs,
  - actor/trainer decoupling,
  - stateless model serving hooks.
- Transparency:
  - keep each module thin and inspectable,
  - avoid hidden configuration magic.
