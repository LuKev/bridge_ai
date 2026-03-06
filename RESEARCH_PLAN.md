# Bridge AI Research Plan (Bidding + Belief -> Sampled Deals -> DDS)

Status: `pivot approved; first belief-model smoke runs completed locally and on Modal`
Last updated: 2026-03-06

## Scope

This document tracks the research loop:
- hypotheses
- experiment designs
- execution log
- conclusions

`PLAN.md` is the implementation control plane.
This file is the scientific control plane.

## Research objective

Determine whether a transformer-centered bridge system can achieve strong practical play by focusing machine learning on the parts it is best suited for:
- bidding
- posterior inference over hidden hands

and then using:
- constrained legal deal sampling
- and downstream search / DDS-backed cardplay

for actual play decisions.

The explicit modeling target is:

- `p(next_bid | own_hand, full public history)`
- `p(hidden_cards | own_hand, full public history)`

where full public history includes:
- seat, dealer, vulnerability, scoring context
- the full auction history
- and later any public play information

## Why this direction

The prior monolithic policy/value approach remains useful as a control baseline, but it is no longer the primary research thesis.

The new thesis is:
- bridge ML should specialize in interpreting public sequences and inferring hidden information
- the model should provide calibrated uncertainty over unseen cards
- play strength should come from reasoning over sampled deals, not just from direct cardplay logits

## Bootstrap stance

Use real-game supervision as a bootstrap only, not as the main scaling strategy.

Initial constraint:
- cap the first supervised bootstrap corpus at `1,000` full games total

Reasoning:
- enough to teach the model basic bidding syntax and initial hidden-hand correlations
- small enough that the research does not over-index on offline imitation
- forces the project to answer the real question: whether self-play plus sampled-deal reasoning can improve beyond a modest supervised warm start

## Active hypotheses

1. **Bidding plus hidden-hand belief is a better transformer target than full end-to-end bridge play**
   - Sequence modeling over auction history and partial information should be more learnable than direct full-game action prediction.
2. **Full auction history is essential context for hidden-hand inference**
   - Posterior quality should degrade materially if the model does not condition on the complete auction sequence.
3. **Posterior calibration matters more than raw training loss**
   - A lower loss checkpoint is not useful unless the resulting hidden-hand beliefs are well calibrated and produce valid whole-deal samples.
4. **Whole-deal sampling is the bridge from ML to practical play**
   - Per-card marginals alone are insufficient; the system needs legal complete deal samples.
5. **DDS is valuable only after inference is credible**
   - A perfect-information solver improves play only if the sampled deals approximate the true posterior over hidden hands.
6. **Shared-policy self-play should improve both bidding and belief**
   - If partners use the same model and bidding system, self-play should let conventions co-adapt while also producing exact hidden-hand supervision from simulated full deals.

## Control baseline

The existing monolithic stack remains the control system:
- monolithic transformer policy/value model
- self-play
- determinization-aware search
- fixed-seed evaluator
- Modal execution

Use it for:
- infrastructure validation
- comparison of downstream playing strength
- guarding against regressions while the new architecture is immature

Do not treat its training loss as the primary research metric.

## Self-play refinement thesis

Under the new architecture, self-play is still important, but in a different role than before.

It should improve:
- the bidding policy
- the hidden-hand posterior
- partner coordination under a shared bidding system

Assumption for the main line:
- partners on a side share the same checkpoint and the same bidding policy
- in the simplest training setup, all four seats may use the same checkpoint

Improvement mechanism:
1. start from the `bootstrap_1000` supervised checkpoint
2. generate self-play deals and auctions
3. use the simulator's known full deal as exact supervision for hidden-card targets
4. use sampled-deal downstream evaluation to create stronger bidding targets than plain imitation
5. retrain the bidding and belief heads jointly

Expected virtuous cycle:
- better bids create more informative auctions
- more informative auctions sharpen hidden-hand inference
- sharper inference improves sampled deals
- better sampled deals improve downstream bid evaluation
- improved evaluation yields stronger next-round bidding targets

## Research model variants

- `M0`
  - existing monolithic policy/value baseline
- `B1`
  - transformer bidding model only
- `B2`
  - shared transformer trunk with bidding head and per-card owner belief head
- `B3`
  - `B2` plus auxiliary hand-summary targets
- `B4`
  - `B3` plus constrained whole-deal sampling
- `B5`
  - `B4` plus downstream search over sampled deals
- `B6`
  - `B4` plus DDS-backed play evaluation over sampled deals

## Training targets

Each supervised example should be seat-specific and include:
- private hand for the acting seat
- full public history
  - especially the full auction history
- target next bid at that decision point
- target ownership labels for all unseen cards

Optional auxiliary targets:
- partner/opponent HCP buckets
- suit-length buckets
- shape classes

Important distinction:
- bidding targets are direct supervised labels
- hidden-card targets come from the true full deal
- downstream play is not the first training target

## Sampling research question

The central systems question is:

How should a transformer posterior be converted into complete legal deals?

Candidate methods:

1. **Constrained sequential sampling from factorized per-card logits**
   - easiest first implementation
   - assign cards one by one while enforcing legal hand counts
2. **Autoregressive card-owner generation**
   - stronger joint modeling
   - more complex implementation
3. **Project-and-repair from independent marginals**
   - useful as a baseline
   - likely weaker and less principled

The first implementation should be method 1.

## Metrics

### Bidding metrics

- next-call accuracy
- next-call log-loss
- legal-call masked accuracy

### Belief metrics

- per-card owner accuracy
- per-card owner log-loss
- calibration / reliability
- top-k owner recall for unseen cards

### Sampling metrics

- whole-deal legal validity rate
- rejection / repair rate
- posterior sample diversity
- posterior concentration under fixed auction contexts

### Downstream play metrics

- duplicate score on fixed board sets
- delta versus monolithic baseline
- latency per move
- performance versus simple search without DDS

## Minimum validity rules

- Use held-out data for bidding/belief claims.
- Keep fixed seed schedules for any sampled-deal comparisons.
- Do not claim downstream play gains unless:
  - sampled deals are legal at high rate
  - belief calibration is measured
  - DDS/search comparisons use the same board sets
- Keep manifests and config snapshots for all reported experiments.

## Data sources

Primary near-term sources:
- existing LIN parsing path in the repository
- public PBN/LIN corpora already referenced for replay validation

Candidate public sources for the initial `bootstrap_1000` corpus:
- [Tistis PBN archive](https://www.tistis.nl/pbn/)
- [bridge_deals_db](https://github.com/ureshvahalia/bridge_deals_db)
- [BBO Handviewer / LIN documentation](https://www.bridgebase.com/tools/hvdoc.html)

Data requirement for each record:
- complete deal
- dealer / vulnerability
- full auction sequence
- ideally full play sequence for later extensions

## Experiment phases

### Phase 0 — Control baseline preservation

Purpose:
- keep the monolithic stack runnable
- preserve Modal checkpoints and manifests for later comparison

### Phase 1 — Offline bidding model

Purpose:
- predict next bids from private hand plus full auction context

Success criteria:
- stable held-out bidding loss
- sensible legal-call distributions
- training corpus limited to `1,000` full games

### Phase 2 — Offline hidden-hand belief model

Purpose:
- predict ownership distributions for unseen cards

Success criteria:
- per-card owner metrics beat simple heuristics
- calibration is measurable and reasonable
- results are achieved without increasing the bootstrap corpus beyond `1,000` games

### Phase 3 — Whole-deal sampler

Purpose:
- convert beliefs into complete legal deal hypotheses

Success criteria:
- high legal sample rate
- low repair rate
- posterior samples visibly shift with auction context

### Phase 4 — Downstream play over sampled deals

Purpose:
- use sampled deals for play-time action scoring

Success criteria:
- stronger play than simple heuristics on fixed boards
- measurable improvement over non-belief baselines

### Phase 5 — DDS integration

Purpose:
- evaluate cardplay actions using an external perfect-information solver over sampled deals

Success criteria:
- DDS interface is stable
- batched sample evaluation is tractable
- downstream score improves enough to justify integration complexity

## Experiment ledger

### 2026-03-01 to 2026-03-05

Completed under the old monolithic-baseline direction:
- environment, legality, scoring, and replay validation
- monolithic transformer policy/value training path
- determinization-aware search
- self-play / train / eval / pipeline orchestration
- manifest validation
- real LIN replay tests
- Modal smoke and scale benchmarking
- search-guided smoke validation

These runs are still useful, but now only as control baselines.

### 2026-03-06

Research direction changed:
- monolithic full-move modeling is no longer the primary research target
- the primary target is now:
  - transformer bidding prediction
  - transformer hidden-hand inference conditioned on full auction history
  - constrained whole-deal sampling
  - downstream sampled-deal play logic
  - later DDS-backed cardplay evaluation

Interpretation:
- the repository's earlier experiments were not wasted
- they established the environment, orchestration, and control baseline needed for the new path
-
  revised research stance:
  - supervised bootstrap is capped at `1,000` full games
  - self-play is now expected to improve both bidding and hidden-hand inference
  - shared-policy partnerships are the default assumption for the main training loop
- Implemented and ran the first end-to-end bidding-plus-belief smoke pipeline:
  - command:
    - `bazel run //:pipeline -- --config-path=configs/smoke.yaml`
  - setup:
    - bootstrap data from full auction records
    - small transformer (`hidden_dim=64`, `layers=2`, `heads=2`)
    - constrained whole-deal sampler over owner logits
  - result:
    - `examples=23`
    - `bid_accuracy=0.5652`
    - `bid_loss=2.0753`
    - `belief_accuracy=0.2910`
    - `belief_loss=1.3865`
    - `avg_true_owner_prob=0.2540`
    - `avg_owner_entropy=1.3717`
    - `sampler_validity_rate=1.0`
  - interpretation:
    - the new architecture is now executable end to end,
    - the bidding head is already learning non-uniform legal distributions on tiny data,
    - the belief head is still near weak-baseline quality, which is expected at this scale,
    - the sampler is structurally sound because it produced valid whole deals for every smoke attempt.
- Ran the new architecture on Modal:
  - smoke config:
    - `configs/modal_smoke.yaml`
    - result:
      - `examples=16`
      - `bid_accuracy=0.5000`
      - `bid_loss=2.1513`
      - `belief_accuracy=0.1859`
      - `belief_loss=1.4083`
      - `avg_true_owner_prob=0.2467`
      - `sampler_validity_rate=1.0`
  - small GPU config:
    - `configs/modal.yaml`
    - result:
      - `examples=17`
      - `bid_accuracy=0.5882`
      - `bid_loss=1.9502`
      - `belief_accuracy=0.2760`
      - `belief_loss=1.3690`
      - `avg_true_owner_prob=0.2563`
      - `sampler_validity_rate=1.0`
  - interpretation:
    - the new stack is Modal-compatible already,
    - a small remote GPU trainer is enough to prove the code path works,
    - output distributions are still weak but visibly non-uniform and inspectable,
    - and the constrained sampler remains stable remotely.
- Built and trained the first real `bootstrap_1000` tournament dataset:
  - source:
    - `bridge_deals_db` release asset `bridge_deals.tar.gz`
  - selected events:
    - `BermudaBowl2023`
    - `VeniceCup2023`
    - `dOrsiTrophy2023`
    - `WuhanCup2023`
  - selection policy:
    - `250` room records per event
    - `1000` total room records
  - local dataset build result:
    - `train_examples=9289`
    - `holdout_examples=2378`
  - Modal training run:
    - config: `configs/modal_bootstrap1000.yaml`
    - epochs: `10`
    - trainer: small GPU-backed Modal run
  - final held-out evaluation:
    - `examples=1024`
    - `bid_accuracy=0.7217`
    - `bid_loss=0.9524`
    - `belief_accuracy=0.3767`
    - `belief_loss=1.2708`
    - `avg_true_owner_prob=0.2924`
    - `sampler_validity_rate=1.0`
  - trajectory over training:
    - epoch `0`:
      - `bid_accuracy=0.0000`
      - `belief_accuracy=0.2599`
    - epoch `10`:
      - `bid_accuracy=0.7217`
      - `belief_accuracy=0.3767`
  - interpretation:
    - the expected direction held:
      - real tournament bootstrap improved both bid accuracy and belief accuracy
    - bidding improved much more sharply than belief, which matches the research expectation
    - belief improved from near-random to clearly above random, but is still far from "strong inference"
    - the sampler remained valid throughout, so the main remaining problem is posterior quality, not structural legality.
- Upgraded the measurement path to track belief during play as public cards are revealed:
  - dataset examples now include play-phase states with:
    - visible dummy cards
    - played-card history
    - current trick context
  - evaluator now reports:
    - `auction_belief_accuracy`
    - `play_belief_accuracy`
    - `play_belief_accuracy_by_played_count`
  - reran the Modal `bootstrap_1000` training under the upgraded path.
- Upgraded Modal `bootstrap_1000` result:
  - final held-out metrics:
    - `bid_accuracy=0.7373`
    - `belief_accuracy=0.3748`
    - `auction_belief_accuracy=0.3772`
    - `play_belief_accuracy=0.3728`
  - trajectory:
    - epoch `0`:
      - `bid_accuracy=0.1994`
      - `auction_belief_accuracy=0.2637`
      - `play_belief_accuracy=0.2460`
    - epoch `10`:
      - `bid_accuracy=0.7373`
      - `auction_belief_accuracy=0.3772`
      - `play_belief_accuracy=0.3728`
  - interpretation:
    - this confirms the earlier intuition that belief should be measured as a time-evolving posterior, not just at auction time,
    - and the current model improves in both auction and play phases over training,
    - although play-phase belief remains only modestly above the rough random baseline and still needs further work.

## Immediate next experiments

1. Preserve the active Modal run as a control checkpoint and manifest namespace.
2. Improve belief calibration and offline metrics on held-out examples.
3. Add self-play refinement with shared-policy partnerships.
4. Compare sampled-deal quality before any DDS claim.
5. Add DDS-backed downstream evaluation only after the sampler is stable.

## Failure modes to monitor

- model learns bidding patterns but hidden-card beliefs remain poorly calibrated
- beliefs collapse to unrealistic independent marginals
- sampled deals are legal only after heavy ad hoc repair
- sampled deals ignore auction information in practice
- DDS appears strong only because evaluation is too tied to solver-friendly boards rather than realistic posterior quality
- the supervised bootstrap corpus becomes the de facto training strategy instead of a small warm start

## Decision criteria

Continue if:
- held-out bidding metrics are stable
- belief calibration is measurable
- sampled deals are legal at high rate
- downstream play improves on fixed boards
- self-play refinement improves over the `bootstrap_1000` checkpoint

Pause or rework if:
- auction context does not materially improve posterior quality
- sampled deals remain inconsistent without brittle repair logic
- DDS integration cost dominates before belief quality is proven
