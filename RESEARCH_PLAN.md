# Bridge AI Research Plan

Status: `in progress`  
Last updated: 2026-03-01

## Scope

This document tracks the *research loop* (hypotheses, experiments, and conclusions).

`PLAN.md` tracks implementation and infrastructure milestones.  
This file tracks scientific progress and experiment execution.

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
2. **Search-depth and determinization ablations**
   - compare V3/V4 candidates on fixed seed set.
3. **Head-to-head checkpoint trend**
   - `baseline_checkpoint` enabled in `evaluation` config.
4. **Action-space coupling study**
   - long-term, compare monolithic policy against phase-separated baselines.

## Experiment ledger

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

## Immediate next experiments

1. Add `configs/smoke_search.yaml` using `configs/smoke.yaml` with search flags enabled.
2. Run:
   - `bazel run //:pipeline -- --config-path=configs/smoke_search.yaml`
3. Log results in this ledger with:
   - config path,
   - run IDs,
   - metric deltas (`mean_score`, `score_std`, `win_rate_vs_zero`),
   - runtime notes.

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
  - first ablation move changes mean score in a repeatable direction.
- Pause or rework if:
  - repeated illegal-action fallbacks dominate for multiple seeds
  - legal-move generation repeatedly fails under search mode
  - score variance explodes under unchanged seed schedule after two iterations.
