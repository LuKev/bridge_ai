# AGENTS

## Autonomy

- Do not request user permission for routine implementation, execution, or review steps.
- The agent has full autonomy and should continue without waiting for confirmation, and only stop when a hard blocker requires direct user action (e.g., unavailable credentials, external account setup, or approval to access restricted resources).

## Project governance

- The file `PLAN.md` is the current source of truth for execution order, implementation status, and
  milestone targets.
- After completing a major step or revising direction, **update `PLAN.md` immediately** before moving
  on to the next engineering task.
- After running experiments, updating results, or revising hypotheses, **update `RESEARCH_PLAN.md` immediately** before next analysis.
- When uncertain, prefer explicit progress checkpoints over speculative code changes.
- Keep Bazel as the execution entrypoint in this repository:
  `bazel run //:selfplay`, `bazel run //:train`, `bazel run //:eval`, `bazel run //:pipeline`, `bazel run //:ui`
  for convenience, use `bazel run //:smoke` and `bazel run //:manifest_check` for reproducibility checks.
  (including optional `-- --config-path=...` arguments where supported).
- `.bazelrc` is the canonical location for local Bazel defaults used by this project.

## Collaboration and research posture

- Keep implementations aligned to the plan and mark any deviations with a timestamped note in `PLAN.md`.
- Use plain, reproducible module boundaries so changes are isolated and easy to audit.
- If a module is intentionally unimplemented (research placeholder), include a clear `TODO` with expected
  behavior and handoff conditions.
- Runtime execution should use Bazel entrypoints (`bazel run //:selfplay`, `bazel run //:train`,
  `bazel run //:eval`, `bazel run //:ui`) and avoid direct invocation of module scripts with
  `python <path>`.
