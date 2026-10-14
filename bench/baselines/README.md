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
   with `--iterations 30 --warmup 5` and the translation cache disabled
   (the benchmark measures inference, not memoization — see
   `bench/bench_translate.cc`).
3. The file content is exactly the `--json` output of `bench/leanmt-bench`,
   copied verbatim from the run. Channels: git note under
   `refs/notes/bench-arm64` keyed by commit (primary; no auth needed —
   `git fetch origin refs/notes/bench-arm64:refs/notes/bench-arm64`), CI
   artifact `leanmt-bench-<sha>`, and a folded JSON block in the benchmark
   comment on PR builds. `extract-baseline.py <sha>` does the fetch +
   verbatim write + schema check.
4. Refresh deliberately (dedicated commit, usually perf work on the inference
   core), not on every merge — a baseline that moves with each commit cannot
   gate regressions.

## Status

- **Landed (NEU-22, 2026-10-13):** `arm64.json` is the verbatim `--json`
  output of the first green `arm64-test-bench` CI run — commit `40c0fb0`
  (`elin/NEU-16-build-hygiene`), Actions run 38434394203. Read out of the git
  note `refs/notes/bench-arm64` via `extract-baseline.py`, per contract §3.
  Refresh only per §4.
