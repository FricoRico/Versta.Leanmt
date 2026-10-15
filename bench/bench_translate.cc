// Benchmark + batch translate CLI for the leanmt inference core.
//
// Bench mode (default — latency vs. sentence length and peak RSS):
//   leanmt-bench --model <model.bin> --vocab <vocab.spm> --shortlist <lex.s2t.bin>
//                [--lengths 8,24,60,120] [--iterations 25] [--warmup 5]
//                [--json results.json]
//
// Translate mode (NEU-50 — batch file→file translation; versioned CLI spec
// `versta.engine-cli/1`, mirrored in Versta.Models `versta/leanmt.py` and
// PIPELINE.md "leanmt translate CLI"):
//   leanmt-bench translate --model <model.bin> --vocab <vocab.spm>
//                [--target-vocab <vocab.spm>] --shortlist <lex.s2t.bin>
//                --input <in.txt> --output <out.txt>
//                [--threads N] [--beam N] [--batch N]
//
//   leanmt-bench --version — print engine + CLI spec versions and exit.
//
// Bench mode measures translation wall latency vs. sentence length (words)
// and peak RSS of the process. Numbers are printed as a markdown table (for
// CI step summaries) and optionally written as JSON (for regression gating).
// The corpus is generated deterministically from a fixed word list so runs
// are comparable across machines and CI builds. Timing uses the library's
// default threading (see qmm; override with --threads N or LEANMT_THREADS).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <vector>

#include "src/Frontend.hh"
#include "src/Model.hh"
#include "src/QMM.hh"
#include "src/Version.hh"

namespace {

// Versioned contract identifier for the translate subcommand (NEU-50).
// Bumped on any breaking change to the translate-mode surface (flags, I/O
// semantics, exit codes); the eval harness in Versta.Models pins against it
// so scores are never compared across contract revisions.
constexpr char kCliSpec[] = "versta.engine-cli/1";

struct Args {
  std::string model;
  std::string vocabulary;
  std::string target_vocabulary;
  std::string shortlist;
  std::vector<size_t> lengths{8, 24, 60, 120};
  size_t iterations = 25;
  size_t warmup = 5;
  size_t batch = 1;
  size_t threads = 0;  // 0 = keep the process default (1 or LEANMT_THREADS)
  std::string json_path;
};

void usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s --model <model.bin> --vocab <vocab.spm> "
               "[--target-vocab <vocab.spm>] --shortlist <lex.s2t.bin> "
               "[--lengths 8,24,60,120] [--iterations N] "
               "[--warmup N] [--batch N] [--threads N] [--json out.json]\n"
               "       %s translate ...   batch file→file translation "
               "(%s translate --help)\n"
               "       %s --version       print engine + CLI spec versions\n",
               argv0, argv0, argv0, argv0);
}

bool parse(int argc, char** argv, Args& args) {
  auto need = [&](int& i, const char* flag) -> std::string {
    if (i + 1 >= argc) {
      std::fprintf(stderr, "missing value for %s\n", flag);
      usage(argv[0]);
      std::exit(2);
    }
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model") args.model = need(i, "--model");
    else if (arg == "--vocab") args.vocabulary = need(i, "--vocab");
    else if (arg == "--target-vocab") args.target_vocabulary = need(i, "--target-vocab");
    else if (arg == "--shortlist") args.shortlist = need(i, "--shortlist");
    else if (arg == "--iterations") args.iterations = std::stoul(need(i, "--iterations"));
    else if (arg == "--warmup") args.warmup = std::stoul(need(i, "--warmup"));
    else if (arg == "--batch") args.batch = std::stoul(need(i, "--batch"));
    else if (arg == "--threads") args.threads = std::stoul(need(i, "--threads"));
    else if (arg == "--json") args.json_path = need(i, "--json");
    else if (arg == "--lengths") {
      std::string csv = need(i, "--lengths");
      args.lengths.clear();
      std::stringstream ss(csv);
      std::string item;
      while (std::getline(ss, item, ',')) args.lengths.push_back(std::stoul(item));
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      usage(argv[0]);
      return false;
    }
  }
  if (args.model.empty() || args.vocabulary.empty()) {
    usage(argv[0]);
    return false;
  }
  return true;
}

// Fixed English word pool (common nouns/verbs/adjectives) — deterministic
// pseudo-sentences, no RNG state anywhere in the benchmark. `offset` rotates
// the starting word so batched rows differ from each other while staying
// fully deterministic.
std::string sentence_of_length(size_t words, size_t offset = 0) {
  static const char* kPool[] = {
      "the",      "station",   "is",      "near",    "the",     "old",
      "market",   "and",       "we",      "can",     "walk",    "there",
      "together", "before",    "the",     "evening", "menu",    "closes",
      "a",        "friendly",  "local",   "showed",  "us",      "the",
      "shortest", "way",       "to",      "the",     "museum",  "entrance",
      "my",       "phone",     "needs",   "a",       "charge",  "before",
      "we",       "continue",  "towards", "the",     "quiet",   "harbour",
      "please",   "speak",     "slowly",  "and",     "write",   "down",
  };
  constexpr size_t kPoolSize = sizeof(kPool) / sizeof(kPool[0]);
  std::string out;
  for (size_t i = 0; i < words; ++i) {
    if (i) out += ' ';
    out += kPool[(i + offset) % kPoolSize];
  }
  out.back() = '.';  // end with a period, like real input
  return out;
}

// Peak RSS in KiB. Prefers /proc/self/status VmHWM (high-water mark, Linux) —
// ru_maxrss is the fallback. Monotonic per process: the value can only grow
// during the run, which is exactly what a regression watch wants.
long rss_peak_kib() {
  std::ifstream proc("/proc/self/status");
  std::string key;
  while (proc >> key) {
    if (key == "VmHWM:") {
      long value = -1;
      proc >> value;
      return value;
    }
    proc.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_maxrss;  // KiB on Linux, bytes on macOS — bench targets Linux
}

struct Result {
  size_t words;
  size_t batch;
  size_t iterations;
  double min_ms, median_ms, p95_ms, max_ms;
  long rss_start_kib, rss_end_kib;
};

Result run_length(leanmt::Blocking& service, const leanmt::Ptr<leanmt::Model>& model,
                  size_t words, size_t batch, size_t iterations, size_t warmup) {
  // A batch of distinct sentences (rotated word-pool offsets): batch=1 keeps
  // the historical single-sentence behaviour verbatim.
  std::vector<std::string> sentences;
  sentences.reserve(batch);
  for (size_t b = 0; b < batch; ++b) {
    sentences.push_back(sentence_of_length(words, b * 7));
  }
  leanmt::Options options;

  for (size_t i = 0; i < warmup; ++i) {
    (void)service.translate(model, sentences, options);
  }

  long rss_start = rss_peak_kib();
  std::vector<double> timings_ms;
  timings_ms.reserve(iterations);
  for (size_t i = 0; i < iterations; ++i) {
    auto started = std::chrono::steady_clock::now();
    (void)service.translate(model, sentences, options);
    auto ended = std::chrono::steady_clock::now();
    timings_ms.push_back(
        std::chrono::duration<double, std::milli>(ended - started).count());
  }
  long rss_end = rss_peak_kib();

  std::sort(timings_ms.begin(), timings_ms.end());
  auto pct = [&](double q) {
    size_t idx = static_cast<size_t>(std::ceil(q * timings_ms.size())) - 1;
    return timings_ms[std::min(idx, timings_ms.size() - 1)];
  };
  return Result{words,
                batch,
                iterations,
                timings_ms.front(),
                pct(0.5),
                pct(0.95),
                timings_ms.back(),
                rss_start,
                rss_end};
}

std::string arch_string() {
#if defined(__aarch64__)
  return "aarch64";
#elif defined(__x86_64__)
  return "x86_64";
#elif defined(__arm__)
  return "armv7";
#elif defined(__i386__)
  return "x86";
#else
  return "unknown";
#endif
}

// ---------------------------------------------------------------------------
// Translate mode (NEU-50): batch file→file translation for the versta.eval
// harness. Contract identifier `versta.engine-cli/1` (kCliSpec):
//
//  * --input holds UTF-8 sentences, one per line; CRLF tolerated; a blank
//    line aborts the run (fail-fast naming the line number — sentence
//    splitting upstream owns segmentation, a misaligned blank would corrupt
//    line alignment silently);
//  * --output receives exactly one translation per input line, in order,
//    each terminated by '\n' (including the last); nothing else is written
//    to the file;
//  * exit codes: 0 success, 1 runtime failure (unreadable input, model-load
//    error, decode failure, unwritable output), 2 usage error;
//  * deterministic for a fixed model + decode settings: ruy never splits
//    the reduction (bit-identical output at any --threads value) and batched
//    decode is row-wise identical to single-line decode (pinned by
//    frontend_test); the translation cache is disabled in this mode;
//  * decode settings are engine defaults unless pinned by flags; the
//    effective settings are echoed to stderr for the audit trail.
// ---------------------------------------------------------------------------

struct TranslateArgs {
  std::string model;
  std::string vocabulary;
  std::string target_vocabulary;
  std::string shortlist;
  std::string input;
  std::string output;
  size_t threads = 0;  // 0 = keep the process default (1 or LEANMT_THREADS)
  size_t beam = 3;     // Options::max_beam_width default; 1 = greedy
  size_t batch = 64;   // lines per translate() call; perf-only knob
};

void translate_usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s translate --model <model.bin> --vocab <vocab.spm>\n"
               "           [--target-vocab <vocab.spm>] --shortlist <lex.s2t.bin>\n"
               "           --input <sentences.txt> --output <translations.txt>\n"
               "           [--threads N] [--beam N] [--batch N]\n"
               "\n"
               "Batch file→file translation (spec %s). Reads UTF-8\n"
               "sentences from --input, one per line, and writes exactly one\n"
               "translation per input line to --output, in order. Deterministic\n"
               "for a fixed model + decode settings (see README \"Batch\n"
               "translation\"). Exit codes: 0 success, 1 runtime failure,\n"
               "2 usage error.\n"
               "\n"
               "  --threads N      ruy intra-op threads (default: LEANMT_THREADS\n"
               "                   env, else 1)\n"
               "  --beam N         beam-width cap for the robust re-decode\n"
               "                   (default 3; 1 = greedy)\n"
               "  --batch N        lines per translate() call; performance knob\n"
               "                   only, output is batch-size invariant (default 64)\n",
               argv0, kCliSpec);
}

size_t parse_size(const std::string& value, const char* flag, const char* argv0) {
  try {
    size_t parsed = std::stoul(value);
    if (parsed == 0 && std::strcmp(flag, "--threads") != 0) {
      throw std::invalid_argument("zero not allowed here");
    }
    return parsed;
  } catch (const std::exception&) {
    std::fprintf(stderr, "invalid value for %s: %s\n", flag, value.c_str());
    translate_usage(argv0);
    std::exit(2);
  }
}

bool parse_translate(int argc, char** argv, TranslateArgs& args) {
  // argv[0] here is the "translate" subcommand word; usage shows it.
  auto need = [&](int& i, const char* flag) -> std::string {
    if (i + 1 >= argc) {
      std::fprintf(stderr, "missing value for %s\n", flag);
      translate_usage(argv[0]);
      std::exit(2);
    }
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      translate_usage(argv[0]);
      std::exit(0);
    } else if (arg == "--model") args.model = need(i, "--model");
    else if (arg == "--vocab") args.vocabulary = need(i, "--vocab");
    else if (arg == "--target-vocab") args.target_vocabulary = need(i, "--target-vocab");
    else if (arg == "--shortlist") args.shortlist = need(i, "--shortlist");
    else if (arg == "--input") args.input = need(i, "--input");
    else if (arg == "--output") args.output = need(i, "--output");
    else if (arg == "--threads") args.threads = parse_size(need(i, "--threads"), "--threads", argv[0]);
    else if (arg == "--beam") args.beam = parse_size(need(i, "--beam"), "--beam", argv[0]);
    else if (arg == "--batch") args.batch = parse_size(need(i, "--batch"), "--batch", argv[0]);
    else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      translate_usage(argv[0]);
      return false;
    }
  }
  if (args.model.empty() || args.vocabulary.empty() || args.input.empty() ||
      args.output.empty()) {
    translate_usage(argv[0]);
    return false;
  }
  return true;
}

int run_translate(const TranslateArgs& args) {
  // Read the whole input up front (fail-fast before any model time is spent
  // on a file we could not parse), one sentence per line.
  std::ifstream in(args.input, std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "error: cannot open input file: %s\n", args.input.c_str());
    return 1;
  }
  std::vector<std::string> lines;
  std::string line;
  size_t line_number = 0;
  while (std::getline(in, line)) {
    ++line_number;
    if (!line.empty() && line.back() == '\r') line.pop_back();  // CRLF tolerance
    if (line.empty()) {
      std::fprintf(stderr,
                   "error: blank line %zu in %s — sentence splitting is owned "
                   "upstream of this CLI; refusing to guess alignment\n",
                   line_number, args.input.c_str());
      return 1;
    }
    lines.push_back(line);
  }
  if (in.bad()) {
    std::fprintf(stderr, "error: failed while reading %s\n", args.input.c_str());
    return 1;
  }

  try {
    if (args.threads > 0) {
      leanmt::qmm::set_num_threads(args.threads);
    }
    auto load_start = std::chrono::steady_clock::now();
    leanmt::Package<std::string> files{
        .model = args.model,
        .vocabulary = args.vocabulary,
        .target_vocabulary = args.target_vocabulary,
        .shortlist = args.shortlist,
    };
    auto model = std::make_shared<leanmt::Model>(leanmt::Model::Config{}, files);
    leanmt::Config service_config{};
    service_config.cache_size = 0;  // decode-only CLI: no memoization anywhere
    leanmt::Blocking service(service_config);
    double load_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - load_start)
                         .count();

    leanmt::Options options;
    options.max_beam_width = args.beam;

    std::fprintf(stderr,
                 "leanmt-bench translate (%s): model=%s lines=%zu "
                 "threads=%zu beam=%zu batch=%zu cache=0 load=%.1f ms\n",
                 kCliSpec, args.model.c_str(), lines.size(),
                 leanmt::qmm::num_threads(), args.beam, args.batch, load_ms);

    auto translate_start = std::chrono::steady_clock::now();
    std::vector<std::string> translations;
    translations.reserve(lines.size());
    for (size_t begin = 0; begin < lines.size(); begin += args.batch) {
      size_t end = std::min(begin + args.batch, lines.size());
      std::vector<std::string> chunk(
          lines.begin() + static_cast<std::ptrdiff_t>(begin),
          lines.begin() + static_cast<std::ptrdiff_t>(end));
      auto responses = service.translate(model, std::move(chunk), options);
      if (responses.size() != end - begin) {
        std::fprintf(stderr,
                     "error: internal invariant violated — %zu lines in, %zu "
                     "responses out (chunk starting at line %zu)\n",
                     end - begin, responses.size(), begin + 1);
        return 1;
      }
      for (const auto& response : responses) {
        // One input line must yield exactly one output line; a raw newline
        // inside a translation would silently corrupt line alignment.
        // translations.size() is the count of lines consumed so far, so the
        // failing input line (1-based) is translations.size() + 1.
        if (response.target.text.find_first_of("\r\n") != std::string::npos) {
          std::fprintf(stderr,
                       "error: translation for input line %zu contains a "
                       "newline — refusing to corrupt line alignment\n",
                       translations.size() + 1);
          return 1;
        }
        translations.push_back(response.target.text);
      }
    }
    double translate_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - translate_start)
                              .count();

    std::ofstream out(args.output, std::ios::binary | std::ios::trunc);
    if (!out) {
      std::fprintf(stderr, "error: cannot open output file: %s\n", args.output.c_str());
      return 1;
    }
    for (const auto& text : translations) {
      out << text << '\n';
    }
    out.flush();
    if (!out) {
      std::fprintf(stderr, "error: failed while writing %s\n", args.output.c_str());
      return 1;
    }
    out.close();

    std::fprintf(stderr, "translated %zu lines in %.1f ms (%.0f lines/s) -> %s\n",
                 translations.size(), translate_ms,
                 translate_ms > 0 ? 1000.0 * translations.size() / translate_ms : 0.0,
                 args.output.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}

int run_bench(const Args& args) {
  // Intra-op GEMM threads: 0 keeps the process default (1, or the
  // LEANMT_THREADS env). Set before model construction so the per-thread
  // ruy contexts never see a stale value. Latencies are bit-identical at any
  // thread count (ruy splits rows/columns, never the reduction).
  if (args.threads > 0) {
    leanmt::qmm::set_num_threads(args.threads);
  }

  auto load_start = std::chrono::steady_clock::now();
  long rss_baseline = rss_peak_kib();
  leanmt::Package<std::string> files{
      .model = args.model,
      .vocabulary = args.vocabulary,
      .target_vocabulary = args.target_vocabulary,
      .shortlist = args.shortlist,
  };
  auto model = std::make_shared<leanmt::Model>(leanmt::Model::Config{}, files);
  leanmt::Config service_config{};
  // The benchmark measures inference, not memoization: with the default
  // translation cache (cache_size 1024) every timed iteration after warmup is
  // a cache hit (~0.04 ms/sentence against a ~14 ms real forward), which would
  // silently mask inference regressions. The app keeps its own cache
  // behaviour; that is benchmarked at app level, not here.
  service_config.cache_size = 0;
  leanmt::Blocking service(service_config);
  double load_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - load_start)
                       .count();

  std::vector<Result> results;
  for (size_t words : args.lengths) {
    results.push_back(
        run_length(service, model, words, args.batch, args.iterations, args.warmup));
  }

  std::printf("\n## leanmt bench (arch=%s, threads=%zu, batch=%zu, model=%s)\n\n",
              arch_string().c_str(), leanmt::qmm::num_threads(), args.batch,
              "ende.student.tiny11 int8 (pinned)");
  std::printf("model load: %.1f ms | RSS start: %ld KiB | peak after load: %ld KiB\n\n",
              load_ms, rss_baseline, rss_peak_kib());
  std::printf("| words | batch | iters | min ms | median ms | p95 ms | max ms | RSS peak KiB |\n");
  std::printf("|------:|------:|------:|-------:|----------:|-------:|-------:|-------------:|\n");
  for (const Result& r : results) {
    std::printf("| %5zu | %5zu | %5zu | %7.3f | %10.3f | %7.3f | %7.3f | %12ld |\n",
                r.words, r.batch, r.iterations, r.min_ms, r.median_ms, r.p95_ms,
                r.max_ms, r.rss_end_kib);
  }
  std::printf("\n");

  if (!args.json_path.empty()) {
    std::ofstream json(args.json_path);
    json << "{\n  \"tool\": \"leanmt-bench\",\n"
         << "  \"arch\": \"" << arch_string() << "\",\n"
         << "  \"threads\": " << leanmt::qmm::num_threads() << ",\n"
         << "  \"batch\": " << args.batch << ",\n"
         << "  \"model\": \"ende.student.tiny11 v2 (pinned sha256)\",\n"
         << "  \"model_load_ms\": " << load_ms << ",\n"
         << "  \"rss_start_kib\": " << rss_baseline << ",\n"
         << "  \"rss_after_load_kib\": " << rss_peak_kib() << ",\n"
         << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
      const Result& r = results[i];
      json << "    {\"words\": " << r.words << ", \"batch\": " << r.batch
           << ", \"iterations\": " << r.iterations
           << ", \"min_ms\": " << r.min_ms << ", \"median_ms\": " << r.median_ms
           << ", \"p95_ms\": " << r.p95_ms << ", \"max_ms\": " << r.max_ms
           << ", \"rss_peak_kib\": " << r.rss_end_kib << "}"
           << (i + 1 < results.size() ? "," : "") << "\n";
    }
    json << "  ]\n}\n";
    std::printf("wrote %s\n", args.json_path.c_str());
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "--version") == 0) {
    std::printf("leanmt-bench %s (leanmt %s)\n", kCliSpec,
                leanmt::version().c_str());
    return 0;
  }
  if (argc > 1 && std::strcmp(argv[1], "translate") == 0) {
    TranslateArgs args;
    // Shift argv so parse_translate sees "translate" as argv[0].
    if (!parse_translate(argc - 1, argv + 1, args)) {
      return 2;
    }
    return run_translate(args);
  }
  Args args;
  if (!parse(argc, argv, args)) {
    return 2;
  }
  return run_bench(args);
}
