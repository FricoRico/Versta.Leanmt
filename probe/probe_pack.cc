// Pack-acceptance probe for Versta translation bundles (NEU-60).
//
//   leanmt-probe --model <model.bin> --vocab <vocab.spm> [--target_vocab <trg.spm>]
//                --shortlist <lex.s2t.bin>
//                --encoder_layers N --decoder_layers N --ffn_depth N --num_heads N
//                --probes <file> [--beam 4] [--max_seq 256] [--threads N]
//                [--json out.json]
//
// Loads one bundle direction exactly the way the app's JNI bridge does
// (Versta.Android app/native/jni/src/leanmt.cc): the same leanmt::Package
// fields, the same Model::Config mapping from bundle metadata, and the same
// decode options as the app's defaults (translation settings screen:
// DEFAULT_NUMBER_OF_BEAMS = 4, DEFAULT_MAX_SEQUENCE_LENGTH = 256). Probes are
// translated one sentence per translate() call so per-sentence latency is
// measurable; the translation cache is disabled so every probe measures real
// inference (same discipline as leanmt-bench).
//
// The probe corpus itself (traveller harm-class sentences with reference
// translations) and the quality checks live pipeline-side in Versta.Models
// (versta.probe); this binary stays language-agnostic: it reports the text
// the engine produced plus the measurement data, nothing more.
//
// JSON is written to stdout and, when --json is given, to that path (same
// payload). Exit codes: 0 ok, 2 argument error, 3 model load failure, 4 a
// probe failed to translate. A load failure still prints a JSON payload with
// "load_ok": false so the driving harness never has to parse stderr.

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
  std::string target_vocabulary;  // shared-vocabulary models leave this empty
  std::string shortlist;
  size_t encoder_layers = 6;
  size_t decoder_layers = 2;
  size_t ffn_depth = 2;
  size_t num_heads = 8;
  std::string probes_path;
  size_t beam = 4;      // app default: DEFAULT_NUMBER_OF_BEAMS
  size_t max_seq = 256; // app default: DEFAULT_MAX_SEQUENCE_LENGTH
  size_t threads = 0;   // 0 = keep the process default (1 or LEANMT_THREADS)
  std::string json_path;
};

void usage(const char* argv0) {
  std::fprintf(
      stderr,
      "usage: %s --model <model.bin> --vocab <vocab.spm> [--target_vocab <trg.spm>]\n"
      "          --shortlist <lex.s2t.bin>\n"
      "          --encoder_layers N --decoder_layers N --ffn_depth N --num_heads N\n"
      "          --probes <file> [--beam 4] [--max_seq 256] [--threads N]\n"
      "          [--json out.json]\n",
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
    else if (arg == "--target_vocab") args.target_vocabulary = need(i, "--target_vocab");
    else if (arg == "--shortlist") args.shortlist = need(i, "--shortlist");
    else if (arg == "--encoder_layers") args.encoder_layers = std::stoul(need(i, "--encoder_layers"));
    else if (arg == "--decoder_layers") args.decoder_layers = std::stoul(need(i, "--decoder_layers"));
    else if (arg == "--ffn_depth") args.ffn_depth = std::stoul(need(i, "--ffn_depth"));
    else if (arg == "--num_heads") args.num_heads = std::stoul(need(i, "--num_heads"));
    else if (arg == "--probes") args.probes_path = need(i, "--probes");
    else if (arg == "--beam") args.beam = std::stoul(need(i, "--beam"));
    else if (arg == "--max_seq") args.max_seq = std::stoul(need(i, "--max_seq"));
    else if (arg == "--threads") args.threads = std::stoul(need(i, "--threads"));
    else if (arg == "--json") args.json_path = need(i, "--json");
    else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      usage(argv[0]);
      return false;
    }
  }
  if (args.model.empty() || args.vocabulary.empty() || args.probes_path.empty()) {
    usage(argv[0]);
    return false;
  }
  return true;
}

// One probe sentence per line, UTF-8. Blank lines and lines starting with '#'
// are skipped so corpus files may carry comments. Lines are translated
// verbatim — no trimming — so what is measured is what is scored.
std::vector<std::string> read_probes(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot open probes file: %s\n", path.c_str());
    std::exit(2);
  }
  std::vector<std::string> sentences;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line[0] == '#') continue;
    sentences.push_back(line);
  }
  return sentences;
}

// Peak RSS in KiB via /proc/self/status VmHWM (high-water mark; Linux), with
// ru_maxrss as the fallback. Same measurement as leanmt-bench so probe numbers
// stay comparable to the bench baselines.
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
  return usage.ru_maxrss;  // KiB on Linux
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

// Minimal JSON string escaping; UTF-8 bytes pass through unchanged.
std::string json_escape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 8);
  for (unsigned char c : in) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

struct ProbeResult {
  std::string source;
  std::string target;
  double ms;
  bool ok;
};

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse(argc, argv, args)) {
    return 2;
  }
  const std::vector<std::string> sentences = read_probes(args.probes_path);

  if (args.threads > 0) {
    leanmt::qmm::set_num_threads(args.threads);
  }

  std::ostringstream json;
  json.precision(3);
  json << std::fixed;

  auto flush_json = [&]() {
    const std::string payload = json.str();
    std::printf("%s\n", payload.c_str());
    if (!args.json_path.empty()) {
      std::ofstream out(args.json_path);
      out << payload << "\n";
    }
  };

  json << "{\n  \"tool\": \"leanmt-probe\",\n"
       << "  \"arch\": \"" << arch_string() << "\",\n"
       << "  \"threads\": " << leanmt::qmm::num_threads() << ",\n"
       << "  \"decode\": {\"beam\": " << args.beam
       << ", \"max_seq\": " << args.max_seq << "},\n";

  // Model load is measured separately: it is the app's "pack ready" moment and
  // the number that makes startup cost and two-vocab overhead visible.
  leanmt::Model::Config config;
  config.encoder_layers = args.encoder_layers;
  config.decoder_layers = args.decoder_layers;
  config.feed_forward_depth = args.ffn_depth;
  config.num_heads = args.num_heads;

  leanmt::Package<std::string> files{
      .model = args.model,
      .vocabulary = args.vocabulary,
      .target_vocabulary = args.target_vocabulary,
      .shortlist = args.shortlist,
  };

  const auto load_start = std::chrono::steady_clock::now();
  std::shared_ptr<leanmt::Model> model;
  try {
    model = std::make_shared<leanmt::Model>(config, files);
  } catch (const std::exception& error) {
    json << "  \"load_ok\": false,\n"
         << "  \"load_error\": \"" << json_escape(error.what()) << "\"\n}\n";
    flush_json();
    return 3;
  } catch (...) {
    json << "  \"load_ok\": false,\n"
         << "  \"load_error\": \"unknown exception\"\n}\n";
    flush_json();
    return 3;
  }
  const double load_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - load_start)
                             .count();

  leanmt::Config service_config{};
  service_config.cache_size = 0;
  leanmt::Blocking service(service_config);

  json << "  \"load_ok\": true,\n"
       << "  \"model_load_ms\": " << load_ms << ",\n"
       << "  \"rss_after_load_kib\": " << rss_peak_kib() << ",\n"
       << "  \"sentences\": [\n";

  // One warmup translation, excluded from the measurements: first-call costs
  // (arena growth, shortlist prep) would otherwise land on probe sentence 1.
  {
    leanmt::Options options;
    options.max_beam_width = args.beam;
    options.max_sequence_length = args.max_seq;
    (void)service.translate(model, {"warmup."}, options);
  }

  std::vector<ProbeResult> results;
  results.reserve(sentences.size());
  bool any_failed = false;
  for (size_t i = 0; i < sentences.size(); ++i) {
    leanmt::Options options;
    options.max_beam_width = args.beam;
    options.max_sequence_length = args.max_seq;

    const auto started = std::chrono::steady_clock::now();
    std::string target;
    bool ok = true;
    try {
      auto responses = service.translate(model, {sentences[i]}, options);
      for (const auto& response : responses) target += response.target.text;
    } catch (...) {
      ok = false;
      any_failed = true;
    }
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();
    results.push_back(ProbeResult{sentences[i], target, ms, ok});

    json << "    {\"source\": \"" << json_escape(sentences[i]) << "\", "
         << "\"target\": \"" << json_escape(target) << "\", "
         << "\"ms\": " << ms << ", \"ok\": " << (ok ? "true" : "false") << "}"
         << (i + 1 < sentences.size() ? "," : "") << "\n";
  }

  std::vector<double> timings;
  timings.reserve(results.size());
  for (const ProbeResult& r : results) {
    if (r.ok) timings.push_back(r.ms);
  }
  std::sort(timings.begin(), timings.end());
  auto pct = [&](double q) {
    size_t idx = static_cast<size_t>(std::ceil(q * timings.size())) - 1;
    return timings[std::min(idx, timings.size() - 1)];
  };

  json << "  ],\n  \"latency\": {"
       << "\"n\": " << timings.size() << ", "
       << "\"min_ms\": " << (timings.empty() ? 0.0 : timings.front()) << ", "
       << "\"p50_ms\": " << (timings.empty() ? 0.0 : pct(0.5)) << ", "
       << "\"p95_ms\": " << (timings.empty() ? 0.0 : pct(0.95)) << ", "
       << "\"max_ms\": " << (timings.empty() ? 0.0 : timings.back()) << "},\n"
       << "  \"rss_peak_kib\": " << rss_peak_kib() << "\n}\n";

  flush_json();
  return any_failed ? 4 : 0;
}
