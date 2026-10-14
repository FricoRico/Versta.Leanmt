// Benchmark harness for the leanmt inference core.
//
//   leanmt-bench --model <model.bin> --vocab <vocab.spm> --shortlist <lex.s2t.bin>
//                [--lengths 8,24,60,120] [--iterations 25] [--warmup 5]
//                [--json results.json]
//
// Measures translation wall latency vs. sentence length (words) and peak RSS
// of the process. Numbers are printed as a markdown table (for CI step
// summaries) and optionally written as JSON (for regression gating later).
//
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

namespace {

struct Args {
  std::string model;
  std::string vocabulary;
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
               "usage: %s --model <model.bin> --vocab <vocab.spm> --shortlist "
               "<lex.s2t.bin> [--lengths 8,24,60,120] [--iterations N] "
               "[--warmup N] [--batch N] [--threads N] [--json out.json]\n",
               argv0);
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

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse(argc, argv, args)) {
    return 2;
  }

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
      .target_vocabulary = "",
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
