# Modal Setup

This repo now has a first runnable Modal path in `bridge_ai.infra.modal_app`.

What it does now:
- runs `selfplay`, `train`, and `eval` as separate Modal functions,
- persists `replays/`, `checkpoints/`, and `artifacts/` on one shared Modal Volume,
- and provides a local entrypoint that sequences those steps with one command.

What it does not do yet:
- distributed actor fleets,
- replay sharding/merging,
- or automated cost controls beyond simple compute-profile environment variables.

## 1. Install the local client

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[modal]"
```

## 2. Authenticate with Modal

```bash
modal setup
```

## 3. Create the shared volume

The code can lazily create the volume on first run, but creating it explicitly makes the setup easier to inspect:

```bash
modal volume create bridge-ai-artifacts --version=2
```

The default config mounts that volume at `/vol/bridge-ai` inside Modal containers.

## 4. Review the Modal config

Use [configs/modal.yaml](/Users/kevin/.codex/worktrees/b49c/bridge_ai/configs/modal.yaml).

Important defaults:
- artifacts live at `/vol/bridge-ai/artifacts`
- replays live at `/vol/bridge-ai/replays`
- checkpoints live at `/vol/bridge-ai/checkpoints`
- the manifest is written to `/vol/bridge-ai/artifacts/modal/manifest.json`
- training runs on CUDA by default
- self-play and training both auto-load `/vol/bridge-ai/checkpoints/latest.pt` when it exists

## 5. Run the first remote pipeline

```bash
modal run -m bridge_ai.infra.modal_app --config-path configs/modal.yaml --job pipeline
```

That command runs three remote steps in order:
- self-play
- training
- evaluation

Useful single-step runs:

```bash
modal run -m bridge_ai.infra.modal_app --config-path configs/modal.yaml --job selfplay
modal run -m bridge_ai.infra.modal_app --config-path configs/modal.yaml --job train
modal run -m bridge_ai.infra.modal_app --config-path configs/modal.yaml --job eval
```

If you want the single-container remote pipeline worker instead of local orchestration across step-specific workers:

```bash
modal run -m bridge_ai.infra.modal_app --config-path configs/modal.yaml --job pipeline_worker
```

## 6. Inspect outputs

Download the manifest locally:

```bash
modal volume get bridge-ai-artifacts artifacts/modal/manifest.json ./modal-manifest.json
```

List the stored checkpoint directory:

```bash
modal volume ls bridge-ai-artifacts checkpoints
```

## 7. Optional compute overrides

The current launcher reads these environment variables at startup:

```bash
export BRIDGE_AI_MODAL_APP_NAME=bridge-ai
export BRIDGE_AI_MODAL_VOLUME=bridge-ai-artifacts
export BRIDGE_AI_MODAL_VOLUME_MOUNT=/vol/bridge-ai
export BRIDGE_AI_MODAL_SELFPLAY_CPU=8
export BRIDGE_AI_MODAL_TRAINER_CPU=4
export BRIDGE_AI_MODAL_EVALUATOR_CPU=2
export BRIDGE_AI_MODAL_TRAINER_GPU=A100
export BRIDGE_AI_MODAL_SELFPLAY_GPU=CPU
export BRIDGE_AI_MODAL_EVALUATOR_GPU=CPU
export BRIDGE_AI_MODAL_TIMEOUT_SECONDS=7200
```

Example: switch the trainer to an `L40S` and keep everything else the same:

```bash
export BRIDGE_AI_MODAL_TRAINER_GPU=L40S
modal run -m bridge_ai.infra.modal_app --config-path configs/modal.yaml --job pipeline
```

## 8. Deploy the app for reuse

```bash
modal deploy -m bridge_ai.infra.modal_app
```

This persists the Modal app and its functions. It does not automatically start training runs by itself; use `modal run` to trigger jobs unless you add a scheduled or programmatic caller later.

## Recommended next scaling step

Keep one trainer GPU and one shared checkpoint namespace first.  
After parity is confirmed against the local baseline, split self-play into sharded replay writers so multiple actor containers can feed the same training run without overwriting `latest.json`.
