# leanmt benchmark baselines

Regression baselines for `bench/leanmt-bench`. Per-PR CI runs on the reference
arm64 runner are compared against these numbers (comparison/gating lands with
the perf-tuning follow-up; today the tables are published for review).

## Files

- `arm64.json` — reference baseline from the `ubuntu-24.04-arm` GitHub runner
  (aarch64, NEON, single thread, Release build).

## Contract (do not hand-edit numbers)

1. Numbers come **only** from CI runs of the `arm64-test-bench` job on
   `ubuntu-24.04-arm`. Never record numbers from a dev box, emulator, or
   cross-run — a baseline from different hardware is worse than no baseline.
2. The run uses the pinned `ende.student.tiny11` v2 fixture
   (sha256 `93821e13b3c511b5...`, fetched by `tests/fixtures/download-fixtures.sh`)
   with `--iterations 30 --warmup 5`.
3. The file content is exactly the `--json` output of `bench/leanmt-bench`,
   copied verbatim from the run (artifact `leanmt-bench-<sha>` /
   `bench-results.json`, also embedded folded in the PR benchmark comment).
4. Refresh deliberately (dedicated commit, usually perf work on the inference
   core), not on every merge — a baseline that moves with each commit cannot
   gate regressions.

## Status

- **Pending (NEU-22):** `arm64.json` lands from the first green
  `arm64-test-bench` run on the NEU-16 build-hygiene PR — the run whose table
  the workflow publishes as a PR comment. The JSON block of that comment is
  the source; copy it here verbatim in a follow-up commit.
