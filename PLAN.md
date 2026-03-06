# Bridge AI Execution Plan (Bidding + Belief -> Sampled Deals -> DDS)

Status: `baseline stack implemented; architecture pivot approved; next implementation phase is bidding-plus-belief`
Owner: Kevin (primary), Codex (execution)
Last updated: 2026-03-06

## Rule

- `PLAN.md` is the implementation control plane.
- Update this file after each major implementation step or direction change.
- Update `RESEARCH_PLAN.md` after experiments, results, or hypothesis changes.
- Keep Bazel as the main local execution entrypoint:
  - `bazel run //:selfplay -- --config-path=...`
  - `bazel run //:train -- --config-path=...`
  - `bazel run //:eval -- --config-path=...`
  - `bazel run //:pipeline -- --config-path=...`
  - `bazel run //:smoke -- --config-path=... --manifest-path=...`
  - `bazel run //:manifest_check -- --manifest-path=...`

## Project objective

Build a bridge AI research stack where:
- a transformer handles bidding prediction and hidden-hand inference,
- the model conditions on private information plus full public history,
- public history explicitly includes the full auction history and later public play history,
- the model produces a posterior over hidden cards / hands,
- complete legal deals are sampled from that posterior,
- sampled deals feed downstream play-time search and later a DDS-backed play oracle,
- and Modal remains the scaling path for training, evaluation, and experiments.

## Strategic position

The current repository already contains a working monolithic policy/value baseline with self-play, determinization-aware search, evaluation, manifests, UI, and Modal execution.

That system is no longer the intended long-term bridge architecture.

It is now the control baseline. The target architecture is:

1. transformer bidding model
2. transformer hidden-hand belief model
3. constrained whole-deal sampler
4. search / DDS-backed play logic

This is a deliberate pivot away from asking one network to emit all bids and card plays directly.

## Bootstrap policy

Use supervised bootstrap, but keep it intentionally bounded.

- Bootstrap from at most `1,000` full bridge games initially.
- Prefer real tournament or high-quality archived games when available.
- Treat supervised data as initialization, not the main long-term source of improvement.
- After the first offline bidding/belief model is stable, shift the research effort toward self-play refinement and sampled-deal downstream evaluation.

## Self-play improvement model

Self-play remains part of the plan, but its role changes.

Under the new architecture, self-play should improve:
- bidding policy quality
- hidden-hand belief quality
- coordination quality between partners using the same policy

Assumption:
- both partners on a side use the same checkpoint and the same bidding policy
- in the simplest setup, all four seats can use the same checkpoint during self-play

Improvement loop:

1. bootstrap the transformer on up to `1,000` full games
2. run self-play auctions and full deals with shared-policy partnerships
3. record:
   - the public trajectory
   - the chosen bids
   - the true hidden-card assignments from the simulator
   - downstream scores from sampled-deal search / DDS evaluation
4. use stronger lookahead-backed targets to refine bidding
5. use the known full deal from self-play as exact supervision for hidden-card belief

Expected feedback cycle:
- better bidding creates more informative auctions
- more informative auctions improve posterior inference
- better posterior samples improve downstream action evaluation
- better downstream evaluation creates better bidding targets for the next generation

## Keep vs replace

### Keep and reuse

- `src/bridge_ai/common/*`
  - card/state/action abstractions remain useful.
- `src/bridge_ai/env/bridge_env.py`
  - legality, scoring, and replay validation remain core infrastructure.
- `src/bridge_ai/data/lin_parser.py`
  - real-deal ingestion remains relevant for supervised bidding/belief data.
- `src/bridge_ai/infra/*`
  - manifests, pipeline orchestration, Modal, and config loading stay in place.
- `src/bridge_ai/ui/streamlit_app.py`
  - extend rather than replace; it should inspect beliefs and sampled deals.
- existing monolithic configs/checkpoints
  - preserve as control baselines for later comparison.

### De-emphasize

- monolithic full-move policy learning as the main research path
- scaling no-search self-play as if loss alone will yield strong bridge play
- deeper investment in the current play-policy head before the belief-model path is tested

### Add

- a bidding-and-belief transformer model
- belief-specific training targets and dataset builders
- a constrained posterior sampler that outputs complete legal deals
- DDS integration layer for downstream play evaluation / action scoring
- belief calibration and sample-validity evaluation tooling

## Target system design

### Inputs to the model

The model should condition on:
- own hand
- seat / dealer / vulnerability / scoring context
- full auction history
- public play history once cardplay is modeled
- any known public cards such as dummy after opening lead

### Outputs from the model

The first target version should produce:
- `bidding_policy`
  - distribution over legal calls at the current auction state
- `hidden_card_belief`
  - posterior logits or probabilities for ownership of each unseen card
- optional auxiliary heads
  - HCP ranges
  - suit length buckets
  - partnership shape summaries

### Downstream inference path

At inference time:

1. run the model on private hand + full public history
2. obtain beliefs over unseen card ownership
3. sample one or more complete legal deals from the posterior
4. run downstream play logic on those deals
5. aggregate action values across sampled deals

The initial downstream play logic can be simple search over sampled deals.
The intended stronger downstream play logic is DDS-backed cardplay evaluation.

## Milestone 1 — Preserve the current baseline

### Goal

Keep the existing monolithic system runnable as a control baseline while the new architecture is implemented.

### Status

- [x] Bridge environment, legality, scoring, and replay validation exist.
- [x] Monolithic transformer policy/value model exists.
- [x] Determinization-aware search exists.
- [x] Self-play, train, eval, pipeline, UI, and manifest tooling exist.
- [x] Modal runner exists and has completed smoke and scaled benchmark runs.
- [x] Search-guided smoke run is validated end to end.

### Guardrail

Do not remove or silently repurpose the monolithic stack. Keep it runnable for control experiments until the belief-model path is proven.

## Milestone 2 — Supervised bidding and belief data

### Goal

Turn complete deal records into supervised examples for bidding and hidden-hand inference.

### Bootstrap constraint

- [ ] Cap the first supervised bootstrap corpus at `1,000` full games total.
- [ ] Prefer full tournament-style records rather than synthetic partial examples.
- [ ] Keep provenance metadata so the exact bootstrap subset is reproducible.

### Planned modules

- `src/bridge_ai/data/belief_dataset.py`
  - builds supervised examples from full deals and auction records
- `src/bridge_ai/data/belief_targets.py`
  - defines target schemas for bidding and hidden-card ownership

### Planned tasks

- [ ] Define a training example schema containing:
  - private seat hand
  - public state
  - full auction history
  - optional public cardplay history
  - target next bid
  - target ownership of every unseen card
- [ ] Build dataset conversion from LIN/PBN-style full records where possible.
- [ ] Build an explicit `bootstrap_1000` dataset manifest:
  - source URL or archive name
  - game identifiers
  - split assignment
- [ ] Add support for seat-wise supervision:
  - each auction position becomes its own example
  - targets reflect that seat's hidden information state.
- [ ] Add optional auxiliary labels:
  - opponent / partner HCP
  - suit lengths
  - shape buckets
- [ ] Add held-out dataset split logic for offline belief evaluation.

## Milestone 3 — Bidding + belief transformer

### Goal

Implement a transformer that predicts both bidding and hidden-card beliefs from private hand plus full public history.

### Planned modules

- `src/bridge_ai/models/bidding_belief_transformer.py`
- `src/bridge_ai/training/belief_train_loop.py`

### Planned tasks

- [ ] Reuse or adapt the existing encoder/tokenization path where sensible.
- [ ] Add a bidding head for legal-call prediction.
- [ ] Add a belief head that predicts owner logits for each unseen card.
- [ ] Keep the architecture transformer-based with a shared trunk and separate heads.
- [ ] Add losses for:
  - bidding cross-entropy / KL
  - card-owner cross-entropy
  - optional auxiliary losses
- [ ] Add checkpoint save/load path independent of the monolithic baseline checkpoints.
- [ ] Keep the first training recipe offline and supervised only.
- [ ] Do not expand the bootstrap corpus beyond `1,000` games before measuring whether self-play refinement is already adding value.

## Milestone 4 — Posterior sampling over complete deals

### Goal

Turn belief outputs into complete legal hidden-hand assignments.

### Planned modules

- `src/bridge_ai/inference/posterior_sampler.py`
- `src/bridge_ai/inference/constraints.py`

### Planned tasks

- [ ] Implement a constrained sequential sampler:
  - assign each unseen card to a seat,
  - mask impossible seats,
  - maintain exact remaining hand counts,
  - respect cards already known or played.
- [ ] Start with factorized per-card logits plus constraint-aware sampling.
- [ ] Evaluate whether an autoregressive sampler is needed for better joint modeling.
- [ ] Track sample validity metrics and posterior concentration diagnostics.
- [ ] Add deterministic seeded sampling for reproducible comparisons.

## Milestone 4b — Self-play refinement for bidding and belief

### Goal

Use self-play to improve the bootstrap model after the first offline bidding/belief checkpoint exists.

### Planned tasks

- [ ] Add a self-play data path for the new architecture:
  - full simulated deal
  - seat-private observation
  - full public auction history
  - chosen bid
  - exact hidden-card ownership targets
  - downstream rollout / DDS-backed value signal where available
- [ ] Assume shared-policy partners by default during self-play.
- [ ] Compare self-play refinement against the offline-only bootstrap model on fixed held-out boards.
- [ ] Track whether self-play improves:
  - bidding quality
  - posterior calibration
  - downstream score
- [ ] Keep the self-play-generated targets clearly separated from the initial `bootstrap_1000` corpus.

## Milestone 5 — DDS-backed play integration

### Goal

Use sampled deals for downstream play-time decision support rather than asking the network to output cardplay directly.

### Planned modules

- `src/bridge_ai/play/dds_adapter.py`
- `src/bridge_ai/play/sample_play_policy.py`

### Planned tasks

- [ ] Integrate an external open-source DDS implementation rather than writing one from scratch.
- [ ] Add a deal-to-DDS conversion layer.
- [ ] Add batched or repeated evaluation over sampled deals.
- [ ] Aggregate DDS-backed values across samples into play action scores.
- [ ] Keep a non-DDS fallback path for environments where the solver is unavailable.

### Integration target

Prefer one of:
- the upstream `dds-bridge/dds` C++ library directly
- or a stable Python-facing wrapper such as `endplay.dds` if packaging/runtime constraints are acceptable

## Milestone 6 — Evaluation and research controls

### Goal

Measure whether the new architecture is actually better, not just lower loss.

### Planned tasks

- [ ] Add offline bidding metrics:
  - next-call accuracy
  - log-loss
- [ ] Add offline belief metrics:
  - per-card owner accuracy
  - per-card owner log-loss
  - calibration
  - legal whole-deal sample rate
- [ ] Add downstream play metrics:
  - score against fixed board sets
  - delta versus monolithic baseline
  - latency per decision
- [ ] Keep fixed-seed evaluation and manifest discipline for all comparisons.

## Milestone 7 — Modal scaling

### Goal

Run the new data, training, and evaluation flow remotely once the architecture is locally valid.

### Planned tasks

- [ ] Extend Modal config/runtime support for belief-model training jobs.
- [ ] Keep artifacts namespaced by experiment family:
  - monolithic control
  - bidding-belief baseline
  - bidding-belief + sampler
  - bidding-belief + DDS
- [ ] Benchmark:
  - offline supervised training throughput
  - sampled-deal inference throughput
  - downstream DDS evaluation throughput

## Execution log

- 2026-03-01 to 2026-03-05:
  - built and validated the original monolithic bridge stack:
    - env, model, search, self-play, train, eval, manifests, UI, and Bazel targets
  - added real LIN replay validation
  - added Modal volume-backed execution
  - completed remote smoke and non-smoke Modal runs
  - launched a larger `scale5x` Modal run as the current monolithic control experiment
  - fixed search-evaluator correctness issues
  - validated one search-guided smoke pipeline end to end
- 2026-03-06:
  - architecture direction formally changed
  - full project plan rewritten around:
    - bidding prediction
    - hidden-hand posterior modeling
    - constrained whole-deal sampling
    - downstream DDS-backed play evaluation
  - monolithic self-play/search stack retained as control baseline only
  - added an explicit bootstrap constraint:
    - initial supervised data cap is `1,000` full games
  - added the new self-play refinement loop:
    - shared-policy partnerships
    - self-play improves both bidding targets and hidden-hand supervision
    - supervised bootstrap is now treated as initialization only
  - implemented the first runnable bidding-plus-belief stack:
    - replaced the old `selfplay/train/eval` execution path with:
      - bootstrap dataset build from full auction records,
      - transformer bidding + hidden-card belief training,
      - holdout evaluation with posterior-sampling diagnostics
    - added:
      - `src/bridge_ai/data/bootstrap_records.py`
      - `src/bridge_ai/data/belief_dataset.py`
      - `src/bridge_ai/models/bidding_belief_transformer.py`
      - `src/bridge_ai/inference/posterior_sampler.py`
      - root `NOTES.md` benchmark note for Jack/GIB/WBridge5 black-box comparisons
    - updated:
      - `configs/default.yaml`
      - `configs/smoke.yaml`
      - `configs/modal.yaml`
      - `configs/modal_smoke.yaml`
      so the primary runtime path now targets the new architecture.
  - validated the first local smoke pipeline on the new stack:
    - `bazel run //:pipeline -- --config-path=configs/smoke.yaml`
    - result:
      - `selfplay_ok=True`
      - `train_ok=True`
      - `eval_ok=True`
      - `examples=23`
      - `bid_accuracy=0.5652`
      - `bid_loss=2.0753`
      - `belief_accuracy=0.2910`
      - `belief_loss=1.3865`
      - `avg_true_owner_prob=0.2540`
      - `sampler_validity_rate=1.0`
    - interpretation:
      - the code path is now end-to-end runnable,
      - output distributions are still weak, but they are not collapsed or obviously broken,
      - the legal constrained sampler is producing valid full deals consistently on smoke scale.
  - validated the new stack on Modal:
    - smoke run:
      - `modal run -m bridge_ai.infra.modal_app --config-path configs/modal_smoke.yaml --job pipeline_worker`
      - result:
        - `selfplay_ok=True`
        - `train_ok=True`
        - `eval_ok=True`
        - `examples=16`
        - `bid_accuracy=0.5000`
        - `belief_accuracy=0.1859`
        - `sampler_validity_rate=1.0`
    - small GPU run:
      - `modal run -m bridge_ai.infra.modal_app --config-path configs/modal.yaml --job pipeline_worker`
      - result:
        - `selfplay_ok=True`
        - `train_ok=True`
        - `eval_ok=True`
        - `examples=17`
        - `bid_accuracy=0.5882`
        - `belief_accuracy=0.2760`
        - `sampler_validity_rate=1.0`
      - remote artifacts confirmed on the shared volume:
        - `/vol/bridge-ai/replays/belief/latest.json`
        - `/vol/bridge-ai/checkpoints/belief/latest.pt`
        - `/vol/bridge-ai/artifacts/belief/manifest.json`
        - `/vol/bridge-ai/artifacts/belief/belief_eval_preview.json`
    - interpretation:
      - the new bidding-plus-belief path now works both locally and on Modal,
      - small neural nets train successfully on remote hardware,
      - and the output distributions are at least plausible enough to inspect rather than obviously degenerate.
  - implemented the reproducible `bootstrap_1000` tournament-data path:
    - added `src/bridge_ai/data/tournament_bootstrap.py`
    - added configs:
      - `configs/bootstrap1000.yaml`
      - `configs/modal_bootstrap1000.yaml`
    - source archive:
      - `bridge_deals_db` release `bridge_deals.tar.gz`
    - selected event files:
      - `BermudaBowl2023.json`
      - `VeniceCup2023.json`
      - `dOrsiTrophy2023.json`
      - `WuhanCup2023.json`
    - record selection:
      - `250` room records per event
      - `1000` total room records
  - added per-epoch holdout logging and plot artifacts:
    - training now writes:
      - `training_history.json`
      - `accuracy_curves.svg`
    - plot tracks holdout bid accuracy and holdout belief accuracy over training.
  - validated local `bootstrap_1000` dataset build:
    - `bazel run //:selfplay -- --config-path=configs/bootstrap1000.yaml`
    - result:
      - `num_records=1000`
      - `train_examples=9289`
      - `holdout_examples=2378`
  - completed the first Modal `bootstrap_1000` training run:
    - `modal run -m bridge_ai.infra.modal_app --config-path configs/modal_bootstrap1000.yaml --job pipeline_worker`
    - result:
      - `examples=1024`
      - `bid_accuracy=0.7217`
      - `bid_loss=0.9524`
      - `belief_accuracy=0.3767`
      - `belief_loss=1.2708`
      - `avg_true_owner_prob=0.2924`
      - `sampler_validity_rate=1.0`
    - pulled artifacts locally:
      - `training_history_bootstrap1000.json`
      - `accuracy_curves_bootstrap1000.svg`
      - `bootstrap1000_manifest_modal.json`
    - interpretation:
      - the first real tournament bootstrap already lifts both bidding and belief metrics substantially over the smoke-scale runs,
      - and the holdout curves show monotonic improvement over the 10-epoch Modal training window.
  - upgraded belief measurement to cover play-phase inference, not just auction-state inference:
    - dataset examples now include:
      - full public auction history,
      - visible dummy cards when exposed,
      - public played-card history,
      - current trick state,
    - evaluator now reports:
      - `auction_belief_accuracy`
      - `play_belief_accuracy`
      - `play_belief_accuracy_by_played_count`
    - training history now records:
      - `holdout_auction_belief_accuracy`
      - `holdout_play_belief_accuracy`
      and the SVG curve includes the play-belief trajectory.
  - reran `bootstrap_1000` on Modal with the upgraded measurement path:
    - final held-out metrics:
      - `bid_accuracy=0.7373`
      - `belief_accuracy=0.3748`
      - `auction_belief_accuracy=0.3772`
      - `play_belief_accuracy=0.3728`
    - training trajectory:
      - epoch `0`:
        - `bid_accuracy=0.1994`
        - `auction_belief_accuracy=0.2637`
        - `play_belief_accuracy=0.2460`
      - epoch `10`:
        - `bid_accuracy=0.7373`
        - `auction_belief_accuracy=0.3772`
        - `play_belief_accuracy=0.3728`
    - interpretation:
      - the model now has a measurable belief-quality trace through play,
      - and both auction-phase and play-phase belief improved materially during the tournament bootstrap run.
  - repaired the GitHub Actions CI dependency graph on Linux:
    - root cause:
      - `torch==2.2.2` on Linux x86_64 pulls platform-scoped CUDA-side `nvidia_*` wheels plus `triton`,
      - but those dependencies were missing from `requirements.txt` and `requirements_lock.txt`,
      - so Bazel `rules_python` failed analysis in the smoke workflow on `main`.
    - fix:
      - added the exact Linux-only transitive requirements to both requirement files,
      - keeping the existing pinned `torch` version instead of re-resolving unrelated packages.
    - local verification:
      - `bazel test //:test_env_rules`
      - `bazel run //:smoke -- --config-path=configs/smoke.yaml --manifest-path=artifacts/smoke/manifest.json`
      - `bazel run //:manifest_check -- --manifest-path=artifacts/smoke/manifest.json`
      - confirmed the lockfile now covers the full Linux x86_64 dependency set declared by `torch==2.2.2`.

## Immediate next actions

1. Finish preserving the active Modal run outputs as monolithic control artifacts.
2. Improve offline evaluation artifacts:
   - add calibration reporting and confidence histograms.
3. Add self-play refinement once the first offline model is stable.
4. Integrate DDS only after sampled deals are valid and reproducible.

## Non-functional requirements

- Reproducibility:
  - deterministic seeds
  - manifest snapshots
  - explicit config snapshots
- Modularity:
  - clean separation between model, sampler, and DDS adapter
- Auditability:
  - control baseline remains runnable
  - belief-model experiments remain comparable
- Pragmatism:
  - prefer an external DDS implementation over writing one
  - avoid scaling the wrong architecture just because infrastructure is already working
