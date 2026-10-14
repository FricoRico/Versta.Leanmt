# Upstream sync review — 2026 Q4

- Reviewer: Elin Sørensen (NEU-27)
- Date: 2026-10-14
- Fork point: `9f0b1a2` ("Add JNI bindings along with a package",
  2024-04-11) — merge-base of `main` against `jerinphilip/slimt`.
- Policy: [`upstream-sync.md`](upstream-sync.md) — **revised 2026-10-14**
  after review: `DavidVentura/slimt` is a reference fork, not an
  upstream. This document reflects that framing: its section below is an
  informational scan; fixes identified there land as own work.

## Summary

| Source | Range examined | Findings | Action |
| --- | --- | --- | --- |
| jerinphilip/slimt `main` | `9f0b1a2..9f0b1a2` | **0 new commits** — upstream `main` tip *is* our fork point; repo dormant (last push 2024-10-24). | Watch continues quarterly. |
| jerinphilip/slimt side branches | `beam-search` (18), `perf` (16), `wasm` (26) vs. fork point | Nothing in scope; see table below. | None. |
| browsermt/bergamot-translator `main` | since 2024-04-11 | 2 commits, neither in core inference paths. | None. |
| google/ruy (submodule) | `c04e5e5..2264753` (20 commits) | 4 correctness/robustness fixes relevant to us; our `-mevex512` patch still needed; prepacked-cache code untouched. | Bump pin to `2264753` this cycle. |
| browsermt/sentencepiece (submodule) | pin `8cbdf13` | Not moved this cycle. | None; next review. |
| DavidVentura/slimt `main` (reference only) | 41 commits, 2026-05-01 → 2026-09-01 | Informational scan; two pre-existing defects in *our* tree identified by comparison. | Fixed one ourselves (on_progress lifetime); second filed as own work (NEU-30). |

## jerinphilip/slimt

`main` has not moved since the fork point — there is literally nothing to
enumerate there. Side branches:

| Branch | Commit(s) | Verdict |
| --- | --- | --- |
| `beam-search` (18) | `dc317b5`…`29cc579` | WIP beam-search rewrite ("Stash changes", "Begin beam-search"). Feature work, explicitly out of sync scope. Skip. |
| `perf` (16) | mostly `Merge branch 'main'` scaffolding around WNGT20-era benchmarking | `a714c0f` "Fix a few buffered read bugs" touches only `app/main.cc` + scripts — both removed from this fork. Rest is benchmark scaffolding. Skip. |
| `wasm` (26) | WASM/emscripten packaging, cibuildwheel bumps | Out of scope (desktop/py packaging). Only core touch is `2be8a3c` (uint64_t→size_t narrowing in `slimt/Io.cc`) — **already covered**: NEU-16 rewrote that code with a bounds-checked `Cursor` doing explicit `static_cast<size_t>` conversions (`src/Io.cc`). Skip. |

## browsermt/bergamot-translator

2 commits on `main` since the fork point: `9271618` "Update submodule" and
`34acd8d` "fix downloading of models in the python binding (#472)". Neither
touches quant GEMM / decode / shortlist / io. Bergamot's inference core
lives in its marian-dev submodule — that lineage does not produce
cherry-pickable fixes for our tree. **Nothing to port.**

## google/ruy — submodule bump evaluation

Pin: `c04e5e5` → candidate `2264753` (`master` tip, 2026-07-22). 20
commits; classification:

| Commit | Subject | Class |
| --- | --- | --- |
| `2264753` | Fix signed integer overflow in `MakeKernelParams8bit` | **Correctness (UB in 8-bit quant path)** — our sole backend. |
| `2af8886` | Fix FlatSize integer overflow | Correctness (overflow). |
| `141477a` | nullptr checks after allocations | Robustness (OOM handling). |
| `690c14c`,`857bcab`,`c08ec52` | cpuinfo null-processor / `QueryCacheParams` error handling | Robustness on exotic hardware/emulators. |
| `83fd40d` | gemmlowp null `VectorMap` in `EvalGemmlowp` | Test-only path; neutral for us. |
| `8467039` | `.inst` instead of `.word` in arm64 assembly | **Build robustness on newer LLVM IAS** — zero semantic change (same encodings). |
| `1e6e508` | android x86-64 support | Platform hygiene; matches our CI matrix leg. |
| `8a21714` | `std::ptrdiff_t` for dims/indices | Mechanical, wide. |
| rest | CMake/bazel/IWYU churn (`6ffa93a`,`8704f5f`,`576e020`,`9940fbf`,`9416e7c`,`587c2cf`,`86427e5`,`9889171`,`95484c3`,`cd7b926`) | Neutral for our build (we consume `ruy/CMakeLists.txt`; verified no flag regressions). |

Key findings against the two concerns flagged in NEU-27:

- **AVX-512/`evex512`: upstream has *not* fixed it.** `2264753` still
  compiles the AVX-512 TUs without `-mevex512`; our conditional patch in
  `third_party/CMakeLists.txt` must be **kept** after the bump. (Verified:
  no `evex512` anywhere in upstream tree.)
- **Prepacked-cache code is untouched** in `c04e5e5..2264753`
  (`prepacked_cache.*`, `ctx*`, `tr_params*` unchanged). The NEU-16
  model-identity invalidation therefore stays compatible; the lifecycle
  regression test remains the guard.

**Recommendation: bump** the pin to `2264753` this cycle. Risk: low
(largest diff is asm-syntax-only; UB fixes do not alter valid-input
behavior). Rollback: restore pin `c04e5e5`.

## Reference scan — DavidVentura/slimt (informational, not synced)

Per policy this fork is reference material, not an upstream we track.
Scanned anyway this cycle because our code historically integrated parts
of its work (pre-policy), and a content comparison is a cheap way to find
latent defects in shared code paths.

Historical integration (pre-policy, credited in README) verified present
in our tree by spot check:

| Reference commit | Subject | Status in LeanMT |
| --- | --- | --- |
| `8a70a76` | vext softmax Pass-3 tail double-division fix | **Present** — `src/Simd.hh` Pass-3 loop, comment verbatim |
| `de4bbbf` | decode independent of batch composition | **Present** — `src/Types.hh:40`, per-row shortlist groups `src/Model.cc:656` |
| `78eec77` | additive slack on per-row target-length cap | **Present** — per-row caps, `src/Model.cc:279` |
| `7496d0a` | quantize round half-to-even (supersedes `eecb061`) | **Present** — `nearbyintf`, `src/qmm/Ruy.inl.cc:172` |
| `f9574dd` | per-thread shortlist pack cache | **Present** + hardened (NEU-16 model-identity invalidation, `src/Model.cc:876`) |
| `543d59e` | NEON gemv decode fast path | **Present** — `src/TensorOps.cc:534` |
| `82b2ad9` | shortlist top-up with next-most-frequent ids | **Present** — `src/Shortlist.cc:182` (`min_candidates`) |
| `9c8c5e5`,`c3fd680` | deficit-stratified re-decode, old path dropped | **Present** — re-decode markers in `src/Model.hh`/`src/Model.cc` |
| `a8a54e4` | top-k alternatives API | **Present** — `src/Response.hh:37` |
| `ba53406` | onprogress callback API | **Present** — `src/Response.hh:80` |
| `375299b` | attention-mask fused into softmax | Perf-neutral per its author; not verified present. Low value — skip unless profiling shows the mask pass hot. |
| `7334643` | win32 support | Not applicable — Android-only fork. |
| `00c32f1` | pivot attention threading | Feature surface we do not expose via JNI. |
| `922f292` | call on_progress before set_value | **Identified a real defect in our tree** — fixed as own work (below). |
| `2beb403` | force-admit digit pieces into shortlist | **Identified a real defect in our tree** — filed as own work (NEU-30). |

### Defect 1 — `on_progress` fired after the result promise (fixed this cycle)

Our `src/Frontend.cc` `Async::translate`/`pivot` continuations called
`promise->set_value(...)` *before* `on_progress()`. A waiter waking on
`set_value` can tear down the stack frame the callback points into while
it is still running → use-after-scope on the JNI bridge. Cross-checking
the reference fork's ordering surfaced this. Fixed in our tree (four-line
reorder, `Informed-by: DavidVentura/slimt@922f292`), behavior-preserving,
lifetime-critical. Not a sync — it is our bug in our code.

### Defect 2 — digits unreproducible through the shortlist (NEU-30)

When the source contains a digit, digit-only target pieces are not
reliably admitted to the shortlist, so numbers can be dropped or doubled
in output. Root cause (per the reference fork's `2beb403`): the lexical
table comes from one deterministic segmentation while training sampled
subwords, so e.g. "404" as ▁4+04 is often unreachable. **Output-affecting**
— changes translations of numeric input. Filed as own implementation work:
NEU-30, behind a golden-diff review and a bench check (digit rows expand
the shortlist slightly).

## Tests assessment

Neither upstream brought core-inference unit tests in any examined range
(slimt: none since fork point; bergamot: none; ruy: `test.h` null-guards
only — we do not run ruy's own suite; reference fork: none).
Consequences carried into this cycle's PRs:

- ruy bump relies on our existing suite: format loader, tokenizer
  determinism, model-destruct/reload lifecycle, and pinned golden
  translations on the ende-tiny11 fixture.
- The `on_progress` fix has no unit-observable signal (thread-lifetime
  race); validated by reasoning + existing frontend determinism tests.
  Coverage gap acknowledged: a deterministic reproducer would need an
  instrumented promise/callback harness — filed as a testing idea, not
  blocking.

## Actions & follow-ups

- [x] Policy established: `docs/upstream-sync.md` (v1), revised same day
      after review (reference-fork stance).
- [x] `on_progress` use-after-scope fixed as own change — PR, CHANGELOG
      `### Fixed` entry; informed-by credit inline.
- [x] ruy pin bump `c04e5e5 → 2264753` — PR, CHANGELOG `### Sync` entry +
      risk note, CI arm64 bench gate applies.
- [ ] NEU-30: digit-forcing implementation (own work, output-affecting).
- Next scheduled review: **2027 Q1** (January). Event triggers per policy.
