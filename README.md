# Versta.Leanmt

**Leanmt** is an Android-specific port of
[slimt](https://github.com/jerinphilip/slimt), the lightweight inference
frontend for tiny neural machine translation models. It lives on as a leaner, 
mobile-oriented fork. This version also contains patches made by David Ventura's
[fork](https://github.com/DavidVentura/slimt) of slimt.

Where upstream slimt aims at desktop, Python and command-line usage, leanmt
strips the project down to the parts that matter for shipping translation on
Android: a small, dependency-light C++ inference core plus the JNI surface that
lets it be called from Kotlin/Java. It is slimmed down, intended to use only
with Firefox translation models. The name is a playful nod to its parent -
*slimt* becomes *leanmt* ("lean machine translation").

## Background

[bergamot-translator](https://github.com/browsermt/bergamot-translator/) builds
on top of [marian-dev](https://github.com/marian-nmt/marian-dev) and reuses
marian's inference code-path. marian is a capable neural-network library focused
on machine translation, but most of what it carries autograd, a wide range of
sequence-to-sequence architectures, beam search is unnecessary for running
inference on client machines. leanmt is just the *tiny* part of marian, retuned
for on-device translation.

The same approach inspired contemporary efforts such as
[ggerganov/ggml](https://github.com/ggerganov/ggml) and
[karpathy/llama2.c](https://github.com/karpathy/llama2.c). The *tiny* models
roughly follow the [transformer architecture](https://arxiv.org/abs/1706.03762)
with [Simpler Simple Recurrent Units](https://aclanthology.org/D19-5632/)
(SSRU) in the decoder, and are the models used by Mozilla Firefox's
[offline translation addon](https://addons.mozilla.org/en-US/firefox/addon/firefox-translations/).

## How leanmt differs from slimt

leanmt is a deliberate slimdown of upstream slimt for the Android use case:

* **Android deployment focus.** Code and assets not relevant to shipping on
  Android have been removed, keeping the tree small and the build fast.
* **Single, embedded compute backend.** Matrix-multiply is hard-wired to
  [ruy](https://github.com/google/ruy) (the sole `int8_t` backend on every
  platform); the BLAS/intgemm/gemmology alternatives from upstream have been
  dropped.
* **NEON tuned.** SIMD paths target ARM NEON for on-device performance; x86
  SSE2/AVX2 paths are retained for development and testing on the host.
* **Self-contained vocabulary.** Vocabulary handling uses a bundled
  [sentencepiece](https://github.com/browsermt/sentencepiece); sentence
  splitting lives outside the library (in the calling Android/Kotlin layer).

The large dependency set of bergamot-translator is reduced to:

* [ruy](https://github.com/google/ruy) - the sole `int8_t` matrix-multiply
  backend.
* [sentencepiece](https://github.com/browsermt/sentencepiece) - vocabulary /
  subword decoding.

## Repository layout

```
src/              # the C++ inference library (namespace leanmt)
  leanmt.hh        # umbrella header: Frontend + Model + Version
  Frontend.hh/.cc  # translation API, batching, request/response
  Model.hh/.cc     # model loading and decoding
  Transformer.*    # encoder/decoder transformer blocks
  TensorOps.*      # quantized matmul, SIMD (NEON/SSE) kernels
  Simd.hh          # SIMD dispatch (vext namespace)
  qmm/             # ruy-backed quantized matrix-multiply
  simd/            # per-ISA SIMD implementations (neon, sse)
third_party/      # bundled ruy and sentencepiece submodules
probe/            # pack-acceptance probe (host tooling; LEANMT_BUILD_PROBE, NEU-60)
```

Public headers install into `<prefix>/include/leanmt`, and the library target
is exported as `leanmt` (with `leanmt::` namespace when packaging is enabled).

## Building

leanmt uses CMake and is meant to be consumed both as a native library in an
Android build (via the NDK + JNI).

### Desktop / host (development)

Clone with submodules:

```bash
git clone --recursive https://github.com/FricoRico/Versta.Leanmt.git
cd Versta.Leanmt
```

Configure and build. On an x86_64 host enable SSE2; on aarch64/armv7 with NEON
use `-DUSE_NEON=ON`:

```bash
ARGS=(
    -DUSE_SSE2=ON            # or -DUSE_NEON=ON on ARM
    -DCMAKE_BUILD_TYPE=Release
)

cmake -B build -S $PWD "${ARGS[@]}"
cmake --build build --target all
```

This produces the `leanmt` shared/static library under `build/`.

Host tooling is opt-in: `-DLEANMT_BUILD_TESTS=ON` (ctest unit tests),
`-DLEANMT_BUILD_BENCH=ON` (`bench/leanmt-bench` latency/RSS harness) and
`-DLEANMT_BUILD_PROBE=ON` (`probe/leanmt-probe`, the pack-acceptance probe the
Versta.Models acceptance harness drives — one bundle direction in, probe
corpus through, JSON with per-sentence translations + latency, load time and
peak RSS out; see NEU-60).

### Android (NDK + JNI)

For on-device use, build `leanmt` with the Android NDK toolchain and expose it
through the JNI bindings. A typical invocation points CMake at the NDK and
selects an ARM ABI with NEON:

```bash
cmake -B build-android -S $PWD \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24 \
    -DUSE_NEON=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build-android --target all
```

The resulting `libleanmt` is linked into the Android application and called from
Kotlin/Java through the JNI layer shipped alongside this repository in the
`Versta` packaging. Sentence-splitting and higher-level text handling are
performed on the Kotlin side; leanmt handles model loading, encoding/decoding
and the quantized inference.

### How the app consumes this repo

The Versta Android app vendors leanmt as a **git submodule** at
`app/native/jni/third_party/leanmt` (`.gitmodules` on the app's `develop`),
pinned to a commit on this repo's `main`. Per-ABI SIMD options
(`USE_NEON` / `LEANMT_ARM64_TUNE` on arm64, `USE_SSE2` on x86) are set by the
app's `app/native/jni/third_party/CMakeLists.txt`, so **option names and
semantics are part of the consumption contract — keep them stable**. Landing
engine changes: merge to `main` here, then bump the app's submodule pin;
never patch the vendored checkout in place (see the app's
`app/native/AGENTS.md`, "Vendored libraries").

## Using the library

The public API centers on `leanmt::Blocking` / `leanmt::Async`, which turn a
loaded `leanmt::Model` into translations:

```cpp
#include "leanmt/leanmt.hh"

leanmt::Package<std::string> package{
    .model = modelPath,         // mmap'd int8 model (bergamot binary layout v1)
    .vocabulary = vocabularyPath,
    .target_vocabulary = "",    // empty on shared-vocab models
    .shortlist = shortlistPath, // optional
};
auto model = std::make_shared<leanmt::Model>(leanmt::Model::Config{}, package);

leanmt::Blocking service(leanmt::Config{});
leanmt::Options options;
auto responses = service.translate(model, {"Hello world, this is a test."}, options);
std::cout << responses[0].target.text << "\n";
```

See the headers under `leanmt/` for the full request/response and annotation
surface (alignments, scores, sentence boundaries).

## Supported toolchains & build policy

The supported, CI-verified build matrix (`.github/workflows/ci.yml`):

| Axis | Values |
|------|--------|
| Android NDK | 27 (r27.3 tested), 28 (r28.1 tested — the app pins 28.1.13356709) |
| Android ABI | arm64-v8a (NEON + armv8.2-a dotprod), armeabi-v7a, x86 (SSE2), x86_64 (SSE2) |
| Host | Linux x86_64 (SSE2) and Linux arm64 (NEON), Clang, CMake ≥ 3.22 |
| C++ standard | C++20 |

LeanMT's own sources compile with `-Wall -Wextra` + fixed suppressions and
are built with `-Werror` (option `LEANMT_STRICT_WARNINGS`, default ON — keep
new code warning-clean; CI gates on it). Bundled `third_party/` deps (ruy,
sentencepiece) build with their own flag sets and are *not* in our warning
budget.

Per-ABI SIMD flags are selected by the *consuming* superproject in
`third_party/CMakeLists.txt`-equivalent glue (see Versta.Android's
`app/native/jni/third_party/CMakeLists.txt`), not by Gradle flavor arguments.

## Native tests

Assertion-based, dependency-free unit tests live in `tests/` and run via
ctest. Model-dependent cases use a pinned FirefoxMT fixture
(`ende.student.tiny11` v2, SHA-256 verified):

```bash
tests/fixtures/download-fixtures.sh   # one-time, checksum-verified
cmake -B build -DLEANMT_BUILD_TESTS=ON -DUSE_NEON=ON   # or -DUSE_SSE2=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

CI runs the suite on both a Linux x86_64 runner and a native Linux arm64
runner (real NEON kernels, no emulation).

## Benchmarks

`bench/leanmt-bench` reports translation latency vs. sentence length and peak
RSS (Linux `VmHWM`), emitting a markdown table plus an optional JSON document:

```bash
./build/bench/leanmt-bench \
    --model tests/fixtures/models/ende.student.tiny11/model.intgemm.alphas.bin \
    --vocab tests/fixtures/models/ende.student.tiny11/vocab.deen.spm \
    --shortlist tests/fixtures/models/ende.student.tiny11/lex.s2t.bin \
    --iterations 30 --warmup 5 --json bench-results.json
```

Useful extra flags: `--threads N` sets the ruy intra-op thread budget (see
below), `--batch N` translates a batch of `N` distinct deterministic sentences
per call (batched-decode measurement), `--lengths 8,24,60,120` overrides the
sentence-length set.

The CI arm64 job runs it per PR (single thread, batch 1 — the canonical
regression-comparable configuration) and publishes the table on the pull
request plus a JSON artifact — same fixture, same runner type, so numbers are
comparable across changes. A thread/batch sweep runs additionally on PR
builds (step summary + perf-sweep comment). Use these to validate latency/RSS
regressions before merging performance work. Regression baselines live in
`bench/baselines/`; only numbers produced on the CI arm64 runner are valid
there (see that directory's README for the provenance contract).

## Batch translation (file → file)

`leanmt-bench translate` is the batch file→file mode of the same binary
(NEU-50; versioned CLI spec `versta.engine-cli/1`, mirrored in
Versta.Models `versta/leanmt.py` and PIPELINE.md "leanmt translate CLI"). It
exists so the quality harness (`versta.eval`) can score the shipped engine +
shipped model bundle as a black box:

```bash
./build/bench/leanmt-bench translate \
    --model model.intgemm.alphas.bin --vocab vocab.deen.spm \
    --shortlist lex.s2t.bin \
    --input sentences.txt --output translations.txt \
    --threads 4 --beam 3
```

Contract (stable within a spec version; bumping the spec is a deliberate,
coordinated event):

- Reads UTF-8 sentences from `--input`, one per line. CRLF is tolerated; a
  blank line aborts the run with the offending line number — segmentation is
  owned upstream of this CLI and a silently-dropped blank would corrupt line
  alignment.
- Writes exactly one translation per input line to `--output`, in order,
  each followed by `\n` (including the last). A translation carrying a raw
  newline aborts the run instead of corrupting alignment. Nothing but the
  translations is written to the output file; progress and the effective
  configuration go to stderr.
- Exit codes: `0` success, `1` runtime failure (unreadable/invalid input,
  model-load error, decode failure, unwritable output), `2` usage error.
- Deterministic for a fixed model + decode settings: ruy never splits the
  reduction (bit-identical output at any `--threads` value) and batched
  decode is row-wise identical to single-line decode (pinned by
  `tests/frontend_test.cc`). The translation cache is disabled in this mode.
- Decode settings are engine defaults unless pinned by flags: `--threads N`
  (ruy intra-op budget; `LEANMT_THREADS` env when the flag is absent),
  `--beam N` (beam-width cap for the robust re-decode, default 3, `1` =
  greedy), `--batch N` (lines per `translate()` call — performance knob only,
  output is batch-size invariant). `leanmt-bench --version` prints the
  engine + CLI spec versions for the harness to record.

Two-vocabulary models (split `encoder_Wemb`/`decoder_Wemb`, e.g. ja/zh) also
need `--target-vocab` — same fail-fast rule as the library (NEU-26). The
flag exists in bench mode as well.

## Threading (intra-op)

leanmt translates on the caller's thread (`Blocking`) or per-request worker
threads (`Async`). GEMM work inside one translate call is intra-op
parallelized by ruy at most to the configured budget:

- `leanmt::Model::Config::threads` (default 0) — the deliberate, documented
  knob; 0 means "leave the process default alone".
- `leanmt::qmm::set_num_threads(n)` — explicit control, applies to the calling
  thread immediately and process-wide to threads whose contexts initialize
  afterwards.
- `LEANMT_THREADS=n` — environment default, mostly for bench/CI. Default is 1.

ruy splits GEMMs over rows/columns and never the reduction dimension, so
translation output is bit-identical at any thread count; per-GEMM fan-out
follows ruy's work-proportional heuristic clamped by the budget. Memory cost
is one pool per translating thread (workers are created lazily, spin briefly
before parking).

## Model loading contract (fail-fast errors)

`leanmt::Model` construction from a package throws `std::runtime_error` with a
precise message in these cases (no crashes, no undefined behaviour):

- model/vocab/shortlist file missing or unmappable (`io::MmapFile`);
- binary model format version mismatch (leanmt expects
  `kBinaryFileVersion` == 1, the bergamot int8 layout);
- truncated or corrupt model file — every length field read is bounds-checked
  against the mapped size;
- tensor with an unknown/incompatible element type.

A model handle outlives any translation session; sessions must be torn down
first (the app-side JNI/Kotlin contract also lives in Versta.Android's
`app/native/AGENTS.md`).

## Debugging switches

Environment levers useful when hunting memory-history-dependent issues:

- `LEANMT_DISABLE_PACK_CACHE=1` — disable ruy's pointer-keyed prepacked-RHS
  cache entirely (slower; eliminates one entire class of cross-model staleness
  while bisecting).
- `LEANMT_POISON_ALLOC=1` — fill fresh heap allocations and recycled arena
  chunks with 0xCD bytes, turning latent read-before-write bugs into visible
  output corruption.
- `LEANMT_DECODE_STATS=1` — per-decode batch/step accounting on stderr.
- `LEANMT_THREADS=n` — environment default for the intra-op ruy thread budget
  (shipped default 2, chosen from the reference-arm64 sweep — see the
  threading section); mostly for bench/CI.

## Status

leanmt is a work-in-progress port. Core text translation works for the
English–German tiny models and parity in features and speed with marian and
bergamot-translator (where relevant) is ongoing. Support for `base` models is
planned. Contributions are welcome.

## Credits & license

leanmt is derived from
[slimt](https://github.com/jerinphilip/slimt) by **Jerin Philip**
(`jerinphilip@live.in`) and **George Tom** (`georg3tom@gmail.com`), including patches
from **David Ventura**. Upstream slimt itself reuses code from
[browsermt/bergamot-translator](https://github.com/browsermt/bergamot-translator)
and [browsermt/marian-dev](https://github.com/browsermt/marian-dev).

leanmt is free software, distributed under the GNU General Public License
version 2 or (at your option) any later version. See `LICENSE` for the full text,
exceptions and third-party attributions 
(`browsermt/ssplit`, Apache-2.0; bergamot-translator / marian-dev, MPL-2.0).
