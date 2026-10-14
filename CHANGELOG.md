# Changelog

All notable changes to this fork are documented here. LeanMT follows the
slimt/bergamot lineage; upstream syncs list their source commits.

## [Unreleased]

### Added (NEU-23)

- Intra-op ruy threading knob: `leanmt::qmm::set_num_threads` /
  `qmm::num_threads`, `Model::Config::threads` (0 = keep process default),
  `LEANMT_THREADS` env seed. Bit-identical outputs at any thread count (ruy
  never splits the reduction dimension; verified empirically). Default stays
  1 until the reference-arm64 sweep justifies otherwise.
- `bench/leanmt-bench`: `--threads N` and `--batch N` (batched-decode
  measurement with a batch of N distinct deterministic sentences); results
  JSON now records the actual thread count and batch per run.
- CI `arm64-test-bench`: per-PR thread/batch sweep (1/2/4 threads × batch
  1/4/8), published to the step summary and a folded PR comment; never gating,
  canonical single-thread run unchanged for baseline comparability.
- Tests: REQUIRE/CHECK_CLOSE helpers in `leanmt_test.hh`, synthetic two-vocab
  loader coverage in `io_test.cc` pinning the int8-embedding view contract and
  bit-exact dequant-on-gather formula.

### Changed (NEU-23)

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
