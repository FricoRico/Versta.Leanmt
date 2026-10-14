# Changelog

All notable changes to this fork are documented here. LeanMT follows the
slimt/bergamot lineage; upstream syncs list their source commits.

## [Unreleased]

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
