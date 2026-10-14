# Changelog

All notable changes to this fork are documented here. LeanMT follows the
slimt/bergamot lineage; upstream syncs list their source commits.

## [Unreleased]

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
