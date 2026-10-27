# Changelog

All notable changes to this fork are documented here. LeanMT follows the
slimt/bergamot lineage; upstream syncs list their source commits.

## [Unreleased]

### Fixed (NEU-157)

- **CI: cmake 4 runner-image drift broke every configure step.** GitHub's
  ubuntu-latest image rolled from cmake 3.x to 4.x (2026-10-27); cmake 4
  removed compatibility with `cmake_minimum_required(VERSION < 3.5)` and
  bundled third_party/sentencepiece still declares 3.1, so all host and
  android CI legs failed at Configure (the android legs use the runner's
  cmake, not the NDK-bundled one — verified cmake 4.4.4 in run 40477149391
  logs). Workaround: `CMAKE_POLICY_VERSION_MINIMUM=3.5` set once in the
  workflow env block (cmake 4's official escape hatch; ignored by cmake
  < 4). **Note for future bumps: our builds couple the bundled third_party
  cmake-minimum floor to whatever cmake the runners ship — a runner-image
  rollover can break the repo with zero code changes on our side.** The
  proper fix remains bumping the sentencepiece pin to a release requiring
  cmake >= 3.5 itself.
- ci.yml "build-and-test" is the only affected workflow; no source changes.

### Added (NEU-50)

- `leanmt-bench translate`: batch file→file translate mode on the existing
  bench CLI binary, for the `versta.eval` quality harness — reads UTF-8
  sentences from `--input` (one per line), writes exactly one translation
  per line to `--output`, in order. New versioned CLI spec
  `versta.engine-cli/1` (mirrored in Versta.Models `versta/leanmt.py` +
  PIPELINE.md, same handshake style as NEU-25's `leanmt_binary_version`).
  Decode settings pinnable by flags (`--threads`, `--beam`, `--batch`) and
  echoed to stderr; cache disabled; blank input lines and embedded newlines
  fail fast rather than corrupting line alignment; exit codes 0/1/2
  (success / runtime failure / usage). `--version` prints engine + spec
  versions. Determinism: bit-identical output across reruns, thread counts
  and batch sizes (batching is row-wise identical — same property
  `frontend_test` pins). Coverage: `bench/translate_cli_test.sh` (ctest).
- `--target-vocab` flag in bench mode as well (two-vocabulary pairs, NEU-26
  rule — previously hard-coded empty). No behavior change when unused.

### Sync (NEU-27, 2026Q4)

- ruy submodule `c04e5e5` → `2264753` (google/ruy master, 2026-07-22).
  Takes upstream fixes for signed-overflow UB in `MakeKernelParams8bit`
  (int8 zero-point path), `FlatSize` overflow, allocator nullptr checks,
  cpuinfo null-processor/`QueryCacheParams` error handling, and arm64
  assembly (`.inst` vs `.word`, LLVM IAS compatibility). Verified: our
  `-mevex512` patch is still required (upstream does not carry the fix)
  and still applies (AVX-512 target names unchanged); ruy's prepacked
  cache code is untouched in this range, so the NEU-16 model-identity
  invalidation is unaffected. Rollback: restore pin `c04e5e5`.
  Full range classification in docs/upstream-review-2026Q4.md.

### Fixed (NEU-27)

- **Use-after-scope in `Frontend` continuations:** `Options::on_progress`
  now fires **before** the result promise is fulfilled. Previously a waiter
  waking on `set_value` could destroy the stack frame that owns the
  callback's captures while a still-running `on_progress` referenced it —
  reachable through the JNI bridge. Risk: none to output bytes (pure
  callback/promise ordering); callers that assumed the future is already
  ready inside their callback would observe the *corrected* order.
  Validation: existing frontend determinism tests; race itself is not
  unit-observable (coverage gap noted in docs/upstream-review-2026Q4.md).
  Found during the 2026Q4 review; approach informed by the reference fork
  DavidVentura/slimt@922f292 — implementation is ours.
### Added (NEU-23)

- Intra-op ruy threading knob: `leanmt::qmm::set_num_threads` /
  `qmm::num_threads`, `Model::Config::threads` (0 = keep process default),
  `LEANMT_THREADS` env seed. Bit-identical outputs at any thread count (ruy
  never splits the reduction dimension; verified empirically).
- `bench/leanmt-bench`: `--threads N` and `--batch N` (batched-decode
  measurement with a batch of N distinct deterministic sentences); results
  JSON now records the actual thread count and batch per run.
- CI `arm64-test-bench`: per-PR thread/batch sweep (1/2/4 threads × batch
  1/4/8), published to the step summary and a folded PR comment; never gating,
  canonical default-config run unchanged for baseline comparability.
- Tests: REQUIRE/CHECK_CLOSE helpers in `leanmt_test.hh`, synthetic two-vocab
  loader coverage in `io_test.cc` pinning the int8-embedding view contract and
  bit-exact dequant-on-gather formula, and a batch-independence frontend test
  (6-row batch must equal 6 single translations, bit-for-bit).

### Rejected after measurement (NEU-23)

- **Extending NEON gemv coverage from rows ≤ 4 to rows 5..8** (2-row
  interleaved sdot kernel for the shortlist-selected output projection under
  batched decode). Reference-arm64 A/B (batch 8, 1 thread): slower at every
  length (+7% at 24 words, +1% at 120) — ruy's 8-row-block kernel wins past
  the existing crossover. The pre-existing ≤4-row dotprod gemv already covers
  single-sentence and beam re-decode batch shapes; batch-of-8 stays on ruy.
  Revert commit `a005ef5` carries the numbers for future reference.

### Changed (NEU-23)

- **Default intra-op ruy threading: 1 → 2 threads.** Reference-arm64 sweep
  (PR #6, ubuntu-24.04-arm, 4 vCPU): at threads=2, single-translate medians
  improve −17% (8w), −16% (24w), −14% (60w), −3.5% (120w), batched inputs
  likewise; no small-input regression; RSS unchanged. 4 threads regressed
  batched long inputs on a 4-vCPU box (oversubscription) and was not taken:
  `LEANMT_THREADS`/`Model::Config.threads` remain the escape hatch. Output
  remains bit-identical to 1-thread (ruy never splits the reduction; the
  pre-flip canonical bench run on the PR shows the same goldens green).
  Baseline `bench/baselines/arm64.json` refreshed from the post-flip CI run
  per the baselines contract §4.

- **Wemb kept as int8 storage; dequantize on gather.** The loader no longer
  materializes f32 copies of `Wemb` / `encoder_Wemb` / `decoder_Wemb` (~32 MiB
  permanent heap for the 32k-vocab tiny11) nor a separate int8 GEMM alias copy
  (~8 MiB): the embedding tensor views the mmapped int8 payload, gathers
  (encoder `index_select`, decoder `from_sentences`) dequantize rows with the
  same `q * (1/mult)` formula bit-exactly, and `Wemb_intgemm8` is a
  transposed-shape view of the same bytes. Local (non-reference) sanity:
  −40 MiB peak RSS after load, −30% load time, latencies unchanged; the
  authoritative numbers come from the CI arm64 run attached to the PR.
  Model numerics are unchanged (pinned goldens pass; A/B vs main byte-identical
  on ende and enja fixtures).

### Fixed (NEU-23)

- `Tensor::quantization_multiplier()` centralizes the trailing-multiplier tail
  convention (previously a raw `end<int8_t>()` read in Modules.cc).

### Added (NEU-16)

- Native unit-test suite (`tests/`, ctest): model-format loader hard/soft
  failure cases, vocabulary/tokenizer determinism, text-processor wrapping,
  model package loading & two-vocab aliasing, frontend end-to-end
  determinism and pinned golden translations, and a model-destruct/reload
  lifecycle regression test.
- Latency/RSS benchmark harness (`bench/leanmt-bench`): latency vs. sentence
  length plus peak RSS (VmHWM), markdown table + JSON output.
- GitHub Actions CI: warning-gated (-Werror) Android build matrix over
  arm64-v8a/armeabi-v7a/x86/x86_64 × NDK 27.3/28.1, host unit tests on
  x86_64, and arm64 tests + benchmark with per-PR published results.
- `LEANMT_STRICT_WARNINGS` (default ON), `LEANMT_BUILD_TESTS`,
  `LEANMT_BUILD_BENCH` CMake options; debugging switches
  `LEANMT_DISABLE_PACK_CACHE` and `LEANMT_POISON_ALLOC`.

### Fixed (NEU-16)

- **Cross-model translation corruption after model destruct/reload.** ruy's
  prepacked-RHS caches are keyed by the source tensor's data pointer, which
  is only sound while the tensor stays mapped there. When a Model is
  destroyed and another constructed, the fresh tensors land on the recycled
  addresses and the per-thread "persistent" cache served the old model's
  packs — batch decodes then produced garbage translations dependent on
  allocator history. `Model::forward` now drops all pack caches on the
  calling thread when the model identity changes.
- `io::load_items` marched raw pointers with no bounds checking; a truncated
  or corrupt model file read past the mmap (SIGSEGV/UB). All reads are now
  bounds-checked and fail fast with precise `std::runtime_error`s;
  non-positive shape dims and non-NUL-terminated names are also rejected.
- Host x86 build: bundled ruy's AVX-512 dispatch units need `-mevex512` on
  modern Clang (incl. zig cc); add it conditionally.
- Include dirs used `CMAKE_SOURCE_DIR`, which leaks the *superproject's* root
  when leanmt is consumed via `add_subdirectory` (as in Versta.Android);
  switched to `PROJECT_SOURCE_DIR`/`PROJECT_BINARY_DIR`.
- Suppressed the benign stderr warning for `special:model.yml` (marian
  metadata YAML, deliberately not consumed).
- README "Using the library" example used an API that no longer exists.

### Hygiene

- Android ABI × NDK matrix verified warning-free; `-Werror` gate in CI.
- Tail of alignment-rounded heap allocations zeroed deterministically.
- Documented supported toolchains, model-format fail-fast contract, and test/
  benchmark procedures in README.
