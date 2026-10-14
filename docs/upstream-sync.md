# Upstream sync policy

LeanMT is a fork of `jerinphilip/slimt` (bergamot lineage). This document
defines what "upstream" means for us, what we watch and on what cadence,
and how a sync lands. Established in NEU-27; first review:
[`upstream-review-2026Q4.md`](upstream-review-2026Q4.md).

**Position (2026-10-14):** LeanMT is an independent codebase, not
a downstream mirror. We track only true upstreams (below) and deliberately
diverge. Other forks of the same base — notably `DavidVentura/slimt` — are
**reference code, not upstream**: read them for ideas, never sync from
them (see "Reference forks").

## Watch list (true upstreams)

| Repo | Relationship | Notes |
| --- | --- | --- |
| [`jerinphilip/slimt`](https://github.com/jerinphilip/slimt) | Direct parent. Our fork point is `9f0b1a2` (2024-04-11). | Dormant as of 2026-10 (main == fork point). Keep watching `main`; feature branches are not sync material. |
| [`browsermt/bergamot-translator`](https://github.com/browsermt/bergamot-translator) | Grandparent lineage (marian-based). | Dormant as of 2026-10. Scan for fixes only; cherry-picks almost never apply textually — port by hand if relevant. |
| [`google/ruy`](https://github.com/google/ruy) | Submodule (`third_party/ruy`), sole int8 GEMM backend. | Active. Semver-free rolling `master`; evaluate bumps deliberately. |
| [`browsermt/sentencepiece`](https://github.com/browsermt/sentencepiece) | Submodule (`third_party/sentencepiece`). | Bump only on concrete need; model-format events come from Versta.Models, not from here. |

## Reference forks (read-only, not upstream)

| Repo | Status |
| --- | --- |
| [`DavidVentura/slimt`](https://github.com/DavidVentura/slimt) | Parallel fork of the same slimt base. LeanMT historically integrated parts of its work (credited in README); that predates this policy. **Going forward: not tracked, not merged, not synced.** |

Rules for reference forks:

- **Inspiration only.** When we fix a bug or build a feature in a path a
  reference fork has also touched, we may study its approach. The fix is
  designed, implemented, and owned here.
- **No port-per-commit syncing.** No "keep parity with fork X" work.
  Divergence from reference forks is expected and desired.
- **Credit where due.** A fix informed by a reference fork carries an
  `Informed-by: <repo>@<sha>` trailer in the commit message and a one-line
  credit in CHANGELOG. It lands under `### Fixed`/`### Added` like any
  other own change — not under `### Sync` (that section is reserved for
  true upstreams).

## Cadence

- **Scheduled review: quarterly** (Jan / Apr / Jul / Oct), tracked as a
  Paperclip issue in the Model Platform project. Each review produces a
  dated document under `docs/upstream-review-<year>Q<n>.md`. While the
  parent repos stay dormant the scheduled review is cheap (mostly a
  verified no-op enumeration); the recurring substantive item is the ruy
  submodule evaluation.
- **Event-driven review** (ad hoc, any of):
  - Upstream security/crash/correctness fix in an in-scope path.
  - ruy or sentencepiece movement relevant to platforms we ship.
  - Model-format spec change in Versta.Models (Nikhil Rao).
  - NDK/Clang bump in our CI matrix that upstream already dealt with.

## Scope

In scope (port when correctness/robustness fixes land):

- Quant GEMM: `src/qmm/`, `src/Quantize*` paths, ruy submodule.
- Decode: `src/Model.*`, `src/Transformer.*`, `src/Batcher.*`, `src/Cache.hh`.
- Shortlist: `src/Shortlist.*`.
- I/O: `src/Io.*` (model package loading — fail-fast contract in README).
- SIMD: `src/Simd.hh`, `src/simd/`.
- Lifetime/concurrency of the public surface (`Frontend`, `Request`,
  `Response`) that the JNI bridge depends on.

Out of scope for sync (handle as their own thing):

- Feature work (new decode modes, APIs, platforms). If an upstream or a
  reference fork develops something product-relevant, file a feature issue
  with the maintainer; do not smuggle features in as "sync".
- Python/CLI/WASM/desktop packaging churn (removed from this fork).
- Build modernization that does not fix a real failure in our matrix.
- Model conversion tooling — owned by Versta.Models / Nikhil Rao.

## How a sync lands

1. **One logical fix per PR.** Never batch unrelated upstream commits into
   one PR. Each ported commit carries the upstream SHA in its commit
   message (`Upstream: <repo>@<sha>`).
2. **Tests.** Port upstream's tests when they exist and fit our harness
   (`tests/`, ctest). When upstream has none, say so in the PR and add our
   own regression coverage where cheap. A fix that cannot be observed by
   any test gets an explicit "coverage gap" note in the risk section.
3. **CHANGELOG.md entry** under `[Unreleased]` in a `### Sync` block:
   source repo, upstream commit SHAs + titles, what we took/skipped,
   and the risk note (see below). (Fixes merely *informed* by a reference
   fork are own changes and land under `### Fixed` with a credit line.)
4. **Risk note** in the PR body: blast radius, behavior-affecting
   (output changes) vs. behavior-preserving, validation performed, and
   rollback (usually: revert the PR; ruy/sentencepiece bumps roll back by
   restoring the submodule pin).
5. **Bench gate.** CI publishes the arm64 latency/RSS table on the PR
   (`bench/leanmt-bench` vs. committed baseline). Latency or peak-RSS
   regression >5% on any row needs a written justification or a fix
   before merge.
6. **Output-affecting fixes** (decode results can change) must say so in
   the PR title (`[output-affecting]`) and get golden-diff numbers on the
   fixture model attached.

## Submodule bumps (ruy, sentencepiece)

- Bumps are their own PRs, separate from source-fix ports.
- Pin to an upstream commit SHA, never a floating branch.
   The bump PR body must include: commit-range summary with per-commit
   relevance classification (port-relevant / neutral / n/a), what
   validation ran, and confirmation that anti-regression invariants still
   hold (currently: the model-destruct/reload lifecycle test and the
   pinned golden translations).
- Known good pins before any bump must be named in the PR so rollback is
  a one-line revert.

## Review document template

Each quarterly review doc (`docs/upstream-review-<year>Q<n>.md`) records:

1. Reviewer, date, fork point, and the exact commit ranges examined.
2. Per-repo enumeration: every commit in range, classified
   (ported / already-present(+evidence) / candidate / skip+reason).
3. Submodule evaluation (ruy, sentencepiece) with the same rigor.
4. Reference-fork activity worth knowing about (informational section —
   no port obligations; anything we act on is filed as own work).
5. Actions taken (PRs), follow-ups (issues created), and the next
   review's scheduled quarter.
