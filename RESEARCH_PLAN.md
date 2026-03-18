# Bridge AI Research Plan

Status: `in progress`  
Last updated: 2026-03-04

## Scope

This document tracks the *research loop* (hypotheses, experiments, and conclusions).

`PLAN.md` tracks implementation and infrastructure milestones.  
This file tracks scientific progress and experiment execution.

## Real-game corpus sources for bridge replay validation

- `https://www.tistis.nl/pbn/` — public event-level PBN archives with downloadable tournament hand files (many links enumerate tournament `.pbn` bundles).
- `https://github.com/ureshvahalia/bridge_deals_db` — community-aggregated bridge deal bundles (PBN/LIN/RBN release artifacts) useful for quick bootstrap of offline replay corpora.
- `https://www.bridgebase.com/tools/hvdoc.html` + `https://www.bridgebase.com/tools/handviewer.html`
  - BBO handviewer documentation and examples for LIN payloads (`lin`, `myhand`) useful for extracting real deal records (`md`, `mb`, `pc`) for end-to-end env replay checks.

## Ready-to-run research launch (no design work needed)

1. Confirm clean workspace state:
   - `bazel test //:test_env_rules`
   - remove any stale outputs (`rm -rf artifacts replays checkpoints`), or keep them and use a new manifest path.
2. Run smoke and reproducibility checks:
   - `bazel run //:selfplay -- --config-path=configs/smoke.yaml`
   - `bazel run //:train -- --config-path=configs/smoke.yaml`
   - `bazel run //:eval -- --config-path=configs/smoke.yaml`
   - `bazel run //:manifest_check -- --manifest-path=artifacts/smoke/manifest.json`
3. Run one-orbit pipeline:
   - `bazel run //:pipeline -- --config-path=configs/smoke.yaml`
4. Start first comparative variant:
   - copy `configs/smoke.yaml` to `configs/smoke_search.yaml`
   - set search-related fields:
     - `selfplay.use_search: true`
     - `evaluation.use_search: true`
     - `selfplay.search_simulations: 4`
     - `evaluation.search_simulations: 4`
     - `selfplay.determinization_count: 1` (or `4` for deeper study)
   - run `bazel run //:pipeline -- --config-path=configs/smoke_search.yaml`

## Primary research objective

Demonstrate whether a monolithic, transformer-based bridge policy/value model trained mostly by self-play and determinized search can match or exceed strong bridge baselines under reproducible experimental conditions.

## Core hypotheses

1. **Monolithic policy/value representation is sufficient**
   - A single transformer with unified action space can match separated bidding/play baselines with stable phase conditioning.
2. **Self-play plus IS-Determinization improves over warm-start behavior**
   - Repeated on-policy self-play with search-generated policy targets should lift performance over random/naive or static behavior cloning baselines.
3. **Determinization depth/sample count controls quality-latency tradeoff**
   - More determinization samples improve decision stability with diminishing returns.
4. **Fixed-deal reproducibility is required**
   - Reproducible manifests, signed run signatures, and config snapshots are necessary for trustworthy comparisons.

## Baseline model variants under test

- V1: no search, no determinization (argmax policy only).
- V2: model + deterministic rollout.
- V3: model + IS-MCTS root rollout using determinization.
- V4: model + larger determinization + wider search depth.

## Experimental design and metrics

For each candidate run:

- Use fixed seeds/deal sets when comparing checkpoints.
- Log:
  - `manifest.json` entries (reproducibility metadata),
  - run-level config snapshots (`artifacts/*/run_configs/<run_id>.yaml`),
  - fixed seed sequence and result summary.
- Report:
  - mean final score
  - score variance / standard deviation
  - win rate vs zero
  - optional `delta_vs_baseline`

### Minimum validity rules

- Same `run_signature` discipline for comparable settings.
- Same seed list or explicit seed schedule in experiment config.
- For engineering sanity checks: `rounds=1`, `num_episodes=1`.
- For scientific claims: `>=100` fixed deals in evaluation.
- All new manifest entries must pass `manifest_check`.

## Experiment execution framework

- Run via Bazel targets only (no direct `python` module execution):
  - `//:selfplay`
  - `//:train`
  - `//:eval`
  - `//:pipeline`
  - `//:smoke`
  - `//:manifest_check`
- Set manifest namespace per sweep:
  - baseline: `artifacts/baseline/manifest.json`
  - search study: `artifacts/search_v1/manifest.json`
  - deterministic variants: `artifacts/determinization_v1/manifest.json`
- Keep all output directories explicit in each config (`storage.manifest_path`, `storage.replay_dir`, `storage.checkpoint_dir`).

  - Keep per-run model continuity by setting `init_checkpoint` or loading latest file where needed.

## Run templates

- **Smoke baseline (no search)**
  - config: `configs/smoke.yaml`
  - command: `bazel run //:pipeline -- --config-path=configs/smoke.yaml`
  - expectations:
    - one successful `selfplay` entry
    - one successful `train` entry
    - one successful `eval` entry
    - manifest passes validation
- **Monolithic + search**
  - config: `configs/smoke_search.yaml` (create from smoke)
  - same invocation
  - expectations:
    - no search crashes
    - score trend consistent across repeated seeds when scale increases
- **Determinization ablation**
  - vary:
    - `selfplay.determinization_count`
    - `evaluation.num_determinizations`
    - `selfplay.search_simulations`
    - `evaluation.search_simulations`
  - run as pipeline per cell and compare against baseline manifest.

## Planned experiments (next phase)

1. **Seeded smoke baseline**
   - confirm full loop completion and manifest integrity.
2. **Checkpoint continuity validation**
   - verify that self-play, learner resume, and snapshot promotion all use the active checkpoint lineage rather than fresh starts.
3. **Duplicate head-to-head checkpoint trend**
   - checkpoint A vs checkpoint B on identical boards with seat swaps.
4. **Search-depth and determinization ablations**
   - compare V3/V4 candidates on fixed duplicate board sets.
5. **Action-space coupling study**
   - long-term, compare monolithic policy against phase-separated baselines.

## Experiment ledger

### 2026-03-10
- Completed self-play strength infrastructure verification:
  - `bazel test //:test_env_rules` passes with real execution of env + infrastructure regressions.
  - `bazel test //:test_ui` passes.
  - `bazel run //:smoke -- --config-path=configs/smoke.yaml --manifest-path=artifacts/smoke/manifest.json` passes with `manifest_issues=[]`.
  - `bazel run //:pipeline -- --config-path=configs/smoke_rating.yaml` passes for two actor/learner/eval iterations in an isolated namespace.
- `configs/smoke_rating.yaml` verification details:
  - iteration 0:
    - self-play uses no prior checkpoint,
    - training emits `/Users/kevin/.codex/worktrees/f2ee/bridge_ai/checkpoints/smoke_rating/iter_000001.pt`,
    - evaluation falls back to single-model fixed-suite scoring (`mean_score=-750.0` on seed `5000`).
  - iteration 1:
    - self-play auto-loads the promoted latest checkpoint,
    - training resumes from the prior snapshot and emits `/Users/kevin/.codex/worktrees/f2ee/bridge_ai/checkpoints/smoke_rating/iter_000002.pt`,
    - evaluation switches to duplicate checkpoint-vs-checkpoint mode against `iter_000001.pt`,
    - duplicate result: `pair_diff_total=650.0` on the fixed board suite,
    - Elo-style ratings update to:
      - `iter_000002.pt`: `1512.0`
      - `iter_000001.pt`: `1488.0`
    - rating artifacts written to:
      - `/Users/kevin/.codex/worktrees/f2ee/bridge_ai/artifacts/smoke_rating/ratings/current.json`
      - `/Users/kevin/.codex/worktrees/f2ee/bridge_ai/artifacts/smoke_rating/ratings/history.jsonl`
    - duplicate match artifacts written under:
      - `/Users/kevin/.codex/worktrees/f2ee/bridge_ai/artifacts/smoke_rating/evaluation/`
- Reproducibility checks:
  - `bazel run //:manifest_check -- --manifest-path=artifacts/smoke_rating/manifest.json` returns `manifest_issues=[]`.
  - runtime path resolution now writes relative `artifacts/`, `replays/`, and `checkpoints/` into the repository workspace under Bazel rather than Bazel execroot.
- Conclusion:
  - checkpoint promotion, replay windows, duplicate evaluation, and Elo history are now functional end to end;
  - the next experiments should increase the number of paired boards and then layer search/determinization ablations on the same harness.
- Added real Modal-continuation tooling:
  - `bazel build //:modal_continue //:league_eval` passes,
  - new regression coverage for legacy checkpoint import, initial baseline creation, and pairwise league reporting is covered by `bazel test //:test_env_rules`.
- Began non-smoke continuation from the preserved 2000-step checkpoint using `configs/modal_real_scale3000.yaml`:
  - source checkpoint: `/Users/kevin/projects/bridge_ai/checkpoints/local_real_scale2000/latest.pt`
  - Modal launcher imports it as immutable `iter_002000.pt` and seeds `latest.pt` in Modal-backed storage,
  - detached Modal app id: `ap-GcJBky40zdgqY8jAzM5xR7`
  - confirmed remote progress via Modal volume manifest:
    - first resumed train block reached `iter_002200.pt`
      - `train_1773145070936817`
      - learner wall-clock: about `2254.7` seconds for 200 learner steps
    - second resumed train block reached `iter_002400.pt`
      - `train_1773147335727418`
      - learner wall-clock: about `2497.7` seconds for 200 learner steps
    - quick duplicate evals on Modal are running after each block and rating artifacts are being updated remotely.
- Modal GPU status:
  - a GPU launch with `nonpreemptible=true` failed immediately because Modal forbids that combination,
  - a GPU launch without `nonpreemptible` also stopped before user-code execution in this workspace,
  - an independent minimal GPU probe reproduced the same `app is stopped or disabled` behavior, so GPU capacity/access in this workspace is the blocking external constraint.
- Interpretation:
  - the Modal CPU path is now operational and reproducible,
  - the final `iter_003000.pt` checkpoint is still in flight and will take multiple additional hours at the observed CPU throughput,
  - any local league output produced before `iter_003000.pt` exists should be treated as provisional only, since `scale3000` still resolves to the current `latest.pt` identity.

### 2026-03-01
- Executed end-to-end tiny-chain smoke on local machine using temporary tiny configs:
  - `selfplay_1772362922134331` (`/tmp/bridge_ai_tiny_abs.yaml`)
    - `episodes=1`, `steps=60`, `status=ok`
  - `train_1772362982182016` (`/tmp/bridge_ai_tiny_abs.yaml`)
    - `loss=2050.2269743124643`, `items=60`, `status=ok`
  - `eval_1772363031320711` (`/tmp/bridge_ai_tiny_eval_nosearch.yaml`)
    - `mean_score=650.0`, `win_rate_vs_zero=1.0`, `rounds=1`, `status=ok`
  - `selfplay_1772363076633062` + `train_1772363080301671` + `eval_1772363084495160` (`/tmp/bridge_ai_tiny_pipeline.yaml`)
    - pipeline completed, `eval.mean_score=0.0`
  - `selfplay_1772363098694376` + `train_1772363108351559` + `eval_1772363111782472` (`/tmp/bridge_ai_tiny_pipeline_search.yaml`)
    - search-enabled pipeline completed, `eval.mean_score=0.0`
- Conclusion: pipeline and manifest workflow are functional; results are not scientifically significant at sample size 1.
- Ran full baseline verification pass for smoke and manifest checks (current date):
  - `bazel test //:test_env_rules` now passes after BUILD test target fix.
  - `bazel run //:selfplay -- --config-path=configs/smoke.yaml` exits successfully.
  - `bazel run //:train -- --config-path=configs/smoke.yaml` exits with `FileNotFoundError: replays/smoke/latest.json` when run standalone (depends on prior replay output in same invocation scope).
  - `bazel run //:eval -- --config-path=configs/smoke.yaml` exits successfully, `mean_score=-1700.0`, `win_rate_vs_zero=0.0`.
  - `bazel run //:pipeline -- --config-path=configs/smoke.yaml` exits successfully, `loss=5.605102479457855`, `eval.mean_score=-350.0`.
  - `bazel run //:smoke -- --config-path=configs/smoke.yaml --manifest-path=artifacts/smoke/manifest.json` exits successfully.
  - `bazel run //:manifest_check -- --manifest-path=artifacts/manifest.json` exits clean, `manifest_issues=[]`.
  - `bazel run //:manifest_check -- --manifest-path=artifacts/smoke/manifest.json` exits clean, `manifest_issues=[]`.
- Immediate note: standalone `train` requires compatible replay artifact locality; full-scope smoke/pipeline runs are the reliable loop until we explicitly add cross-invocation output persistence.
- Ran first local non-smoke pipeline with explicit per-sweep namespaces:
  - `configs/local_real.yaml` (created in-repo for this run):
    - `storage.replay_dir: replays/local_real`
    - `storage.checkpoint_dir: checkpoints/local_real`
    - `storage.manifest_path: artifacts/local_real/manifest.json`
    - model: `hidden_dim=128`, `num_layers=4`, `num_heads=4`
    - episodes: 4, search disabled
    - evaluation rounds: 4
  - Command result:
    - `bazel run //:pipeline -- --config-path=configs/local_real.yaml`
    - `selfplay_ok=True`, `train_ok=True`
    - `eval.mean_score=0.0`, `win_rate_vs_zero=0.0`, `score_std=0.0`
    - `loss=6.355249804835165`
  - Manifest integrity:
    - `bazel run //:manifest_check -- --manifest-path=artifacts/local_real/manifest.json`
    - `manifest_issues=[]`
- Ran first scaled local training sweep:
  - `configs/local_real_iter1.yaml`:
    - selfplay: `num_episodes=16`, `max_steps=120`, `search_simulations=2`, `determinization_count=2`
    - training: `iterations=3`, `epochs=2`, `batch_size=16`
    - evaluation: `rounds=16`, fixed seed sequence `[7..22]`
    - pipeline: `iterations=3`
  - `bazel run //:pipeline -- --config-path=configs/local_real_iter1.yaml`
    - `selfplay_ok=True`, `train_ok=True`, `eval_ok=True` for all 3 iterations
    - eval per iteration: `mean_score=0.0`, `win_rate_vs_zero=0.0`, `score_std=0.0`
    - first-iteration loss trace:
      - `5.9763`, `4.6164`
    - second-iteration loss trace:
      - `3.3288`, `2.8845`
    - third-iteration loss trace:
      - `2.7344`, `2.6565`
  - `bazel run //:manifest_check -- --manifest-path=artifacts/local_real_iter1/manifest.json`
  - `manifest_issues=[]`

- Ran local large-iteration scaling attempt:
  - config: `configs/local_real_iter4.yaml`
  - run command: `bazel run //:pipeline -- --config-path=configs/local_real_iter4.yaml`
  - settings: 8 self-play episodes, 80 max steps, `training.iterations=3000`, batch size 64, no search
  - status: interrupted at iteration 905 with `KeyboardInterrupt` (intentional checkpoint/CPU budget pause)
  - observed loss trend:
    - iter 0: `6.4052`
    - iter 100: `0.9596`
    - iter 200: `0.2805`
    - iter 500: `0.10399`
    - iter 800: `0.06699`
    - iter 900: `0.06681`
    - iter 905: `0.06376`
  - interpretation: monotonic improvement from ~6.4 to ~0.06 by 900 iterations with no divergence; run was interrupted before checkpoint emit could be observed in manifest (checkpoint/iteration persistence needed stronger guarantees on resume).
  - manifest check:
    - `bazel run //:manifest_check -- --manifest-path=artifacts/local_real_iter4/manifest.json`
    - `manifest_issues=[]`
- Follow-up: this confirms the local loop is runnable at a larger scale and produces stable per-iteration persistence behavior.
- 2026-03-18:
  - Queried Modal workspace billing directly through `modal.billing.workspace_billing_report(...)` for March 2026.
  - observed spend:
    - workspace month-to-date: `$5.07589633`
    - detached continuation app `ap-GcJBky40zdgqY8jAzM5xR7`: `$2.07796300`
  - observed detached continuation progress from the Modal volume:
    - remote snapshots currently present: `iter_002000.pt`, `iter_002200.pt`, `iter_002400.pt`
    - last recorded event is `selfplay_1773149842931819`; no `iter_002600.pt` exists yet.
  - estimate:
    - completed `2000 -> 2400` cost implies roughly `$1.03898` per 200-step learner block
    - remaining `2400 -> 3000` continuation is about `$3.11694` on the current 32-core CPU path
  - interpretation:
    - this does not look like hard credit exhaustion on a Starter workspace with `$30 / month` included compute, though a separate workspace budget cap cannot be ruled out from the CLI alone
    - the previous retry path would have recomputed the first 400 learner steps; `configs/modal_real_scale3000.yaml` plus `modal_continue.py` now use `modal.target_step: 3000` and `force_bootstrap: false` so reruns only schedule the remaining blocks from the saved volume.

- Ran clean 2000-iteration local stretch from new namespace:
  - config: `configs/local_real_scale2000.yaml`
  - run command: `bazel run //:pipeline -- --config-path=configs/local_real_scale2000.yaml`
  - settings:
    - `selfplay.num_episodes: 8`, `selfplay.max_steps: 80`, search disabled
    - `training.iterations: 2000`, `training.batch_size: 64`, `training.checkpoint_every: 100`
    - `evaluation.rounds: 4`
  - result:
    - training reached `iteration: 1999` successfully (full 2000 iterations)
    - loss improved from ~`6.50` at iter 0 to `0.0313774056` at iter 1999
    - `eval mean_score=-187.5`, `win_rate_vs_zero=0.0`, `score_std=188.33148966649205`
    - `manifest` entries for `selfplay`, `train`, and `eval` all `status=ok`
    - `bazel run //:manifest_check -- --manifest-path=artifacts/local_real_scale2000/manifest.json` passed (`manifest_issues=[]`)
  - checkpoint behavior:
    - `training.checkpoint_every=100` writes `checkpoints/local_real_scale2000/latest.pt` at the configured cadence and on completion.
- 2026-03-04:
  - Added centralized LIN parser/replay utilities in `src/bridge_ai/data/lin_parser.py`:
    - decode/parse real LIN records (`md`, `mb`, `pc`, vulnerability, claims),
    - reconstruct initial states with dealer/claims, and
    - replay complete games with legal-action assertions.
- Updated `tests/test_bridge_env.py` to consume that parser for `test_replay_of_real_lin_games_no_illegal_action` and enforce strict terminal/non-terminal invariants per fixture.
- Next immediate action: add a corpus manifest + fixture loader so multiple downloaded files can be fed into the same strict replay path.
- Added a curated 20-record real LIN fixture at `tests/fixtures/real_lin_records.txt` and fixed explicit 4-hand LIN parsing support (`md` hand sections with 4 hands now parse correctly).

### 2026-03-03
- Added a real-game regression path for full end-to-end replay validation:
  - parser support for key LIN tokens (`md`, `mb`, `pc`, `sv`, `mc`) is present in `tests/test_bridge_env.py`.
  - added fixed real-world fixtures (BBO-format LIN strings) under `_REAL_LIN_GAME_RECORDS`.
  - strict terminal/non-terminal expectations are now enforced per fixture.
  - this test layer is intended as the first guardrail before scaling data-driven sweeps.

## Immediate next experiments

1. Let the detached Modal CPU continuation finish through `iter_003000.pt`:
   - current confirmed remote snapshots: `iter_002200.pt`, `iter_002400.pt`,
   - sync local artifacts once the detached app completes.
2. Run the final non-smoke checkpoint league after `iter_003000.pt` exists locally:
   - `initial` vs `scale2000` vs `scale3000`,
   - fixed `gating` suite first, `ladder` only if the ranking signal is directionally consistent.
3. If Modal GPU capacity becomes available later, rerun the same 2000 -> 3000 continuation on GPU and compare wall-clock plus resulting checkpoint strength against the CPU path.
4. Only after the real 3000-step checkpoint comparison is complete, return to search and determinization ablations on the duplicate harness.

## Failure modes to monitor

- Illegal-action leakage from token/state representation to policy head.
- Contract/bidding drift from self-play-only optimization.
- Determinization consistency bugs in hidden-card completion.
- Scoring regressions from rule edge cases.
- Reproducibility drift (snapshot missing or mismatched seeds).

## Decision criteria

- Continue once:
  - smoke/pipeline runs are green
  - manifest validation is clean
  - duplicate checkpoint-vs-checkpoint evaluation is producing stable paired results
  - first rating trend shows newer checkpoints beating an anchored baseline in a repeatable direction.
- Pause or rework if:
  - repeated illegal-action fallbacks dominate for multiple seeds
  - legal-move generation repeatedly fails under search mode
  - duplicate board variance remains too high to separate checkpoints
  - newer checkpoints fail to beat the anchored baseline across repeated paired suites.
