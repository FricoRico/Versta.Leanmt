// Numeric-fidelity gate (NEU-30): translates the pinned numeric travel
// corpus (tests/fixtures/numeric/travel_ende_100.txt, 100 sentences with
// prices, times, dates, addresses, quantities) and scores how faithfully
// the numbers survive translation.
//
// Metric (tests/digit_metrics.hh): value-level and script-normalized,
// mirroring the pack-acceptance probe (Versta.Models
// versta.probe.metrics.digit_preservation). Arabic-Indic (U+0660–0669) and
// Persian/Extended Arabic-Indic (U+06F0–06F9) digits are transliterated to
// ASCII first (۱۲ ≡ 12; ۴۵۰۰ vs 5400 fails), clocks compare as
// 12h-normalized (hour, minute) values across colon/Japanese/Korean/
// meridiem surfaces (18:45 ≡ 18時45分 ≡ 6:45 pm), and the remaining digit
// runs compare as multisets (order-free, so German word order and
// decimal-comma reformatting pass; dropped, fused ("404404") or duplicated
// numbers fail). A sentence counts as an exact match when every source
// number carrier is preserved and nothing extra appears. Both per-carrier
// recall and per-row exact match are reported; the gate is `--require-rate`
// on row exact match.
//
// Model, corpus and decode are pinned and deterministic, so the rate is
// stable in CI and per-PR runs. The translation cache is disabled: this
// gate must always exercise decode, never memoization.
//
// Usage:
//   numeric_fidelity_test [--corpus FILE] [--require-rate 0.95]
//                         [--dump before-after.tsv]
// --dump writes "source<TAB>target" per row — diffing two dumps from builds
// with and without a change is the golden-diff table attached to PRs.

#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "leanmt_test.hh"
#include "tests/digit_metrics.hh"
#include "src/Frontend.hh"
#include "src/Model.hh"

namespace {

std::string fixture(const std::string& file) {
  return leanmt::test::fixture_dir() + "/" + file;
}

bool fixture_available() {
  struct stat st;
  return ::stat(fixture("model.intgemm.alphas.bin").c_str(), &st) == 0;
}

bool corpus_available(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0;
}

struct RowResult {
  std::string source;
  std::string target;
  size_t runs_total = 0;     // expected number carriers (digit runs + clocks)
  size_t runs_matched = 0;   // source carriers present in output
  size_t runs_extra = 0;     // output carriers not in source (doubling detection)
  bool exact = false;        // value-level multiset equality of all carriers
};

RowResult score_row(const std::string& source, const std::string& target) {
  leanmt::numeric::DigitReport report =
      leanmt::numeric::compare_digit_values(source, target);
  return RowResult{source,    target,        report.expected,
                   report.matched, report.extra, report.exact};
}

}  // namespace

int main(int argc, char** argv) {
  std::string corpus_path = LEANMT_NUMERIC_DEFAULT_CORPUS;
  std::string dump_path;
  double require_rate = 0.95;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto value = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", flag);
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--corpus") corpus_path = value("--corpus");
    else if (arg == "--dump") dump_path = value("--dump");
    else if (arg == "--require-rate") require_rate = std::stod(value("--require-rate"));
    else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }

  if (!fixture_available()) {
    std::cout << "    skip: fixtures not present (run "
                 "tests/fixtures/download-fixtures.sh)\n";
    return 0;  // skip quietly, like every model-dependent test here
  }
  if (!corpus_available(corpus_path)) {
    std::fprintf(stderr, "numeric corpus not found: %s\n", corpus_path.c_str());
    return 2;
  }

  std::vector<std::string> sentences;
  {
    std::ifstream in(corpus_path);
    for (std::string line; std::getline(in, line);) {
      if (!line.empty()) sentences.push_back(line);
    }
  }
  if (sentences.empty()) {
    std::fprintf(stderr, "numeric corpus is empty: %s\n", corpus_path.c_str());
    return 2;
  }

  leanmt::Package<std::string> files{
      .model = fixture("model.intgemm.alphas.bin"),
      .vocabulary = fixture("vocab.deen.spm"),
      .target_vocabulary = "",
      .shortlist = fixture("lex.s2t.bin"),
  };
  auto model = std::make_shared<leanmt::Model>(leanmt::Model::Config{}, files);
  leanmt::Config config{};
  config.cache_size = 0;  // measure decode, not memoization (see bench)
  leanmt::Blocking service(config);
  leanmt::Options options;

  std::vector<RowResult> rows;
  rows.reserve(sentences.size());
  for (const std::string& sentence : sentences) {
    auto responses = service.translate(model, {sentence}, options);
    const std::string& target =
        (responses.size() == 1 && !responses[0].target.text.empty())
            ? responses[0].target.text
            : "";
    rows.push_back(score_row(sentence, target));
  }

  size_t exact = 0, runs_total = 0, runs_matched = 0, runs_extra = 0;
  for (const RowResult& row : rows) {
    exact += row.exact;
    runs_total += row.runs_total;
    runs_matched += row.runs_matched;
    runs_extra += row.runs_extra;
  }
  double row_rate = static_cast<double>(exact) / rows.size();
  double run_rate = static_cast<double>(runs_matched) / runs_total;

  std::printf("\nnumeric fidelity — %zu sentences (%s)\n", rows.size(),
              corpus_path.c_str());
  std::printf("row exact-match: %zu/%zu = %.1f%%\n", exact, rows.size(),
              100.0 * row_rate);
  std::printf("number-carrier recall: %zu/%zu = %.1f%% (+ %zu extra runs in "
              "output)\n",
              runs_matched, runs_total, 100.0 * run_rate, runs_extra);
  std::printf("numeric_exact_match=%.3f (rows %zu/%zu, runs %zu/%zu)\n",
              row_rate, exact, rows.size(), runs_matched, runs_total);

  std::printf("\n| # | runs | extra | source | output |\n");
  std::printf("|--:|-----:|------:|--------|--------|\n");
  size_t index = 0;
  for (const RowResult& row : rows) {
    ++index;
    if (row.exact) continue;
    std::printf("| %zu | %zu/%zu | +%zu | %s | %s |\n", index,
                row.runs_matched, row.runs_total, row.runs_extra,
                row.source.c_str(), row.target.c_str());
  }
  if (exact == rows.size()) std::printf("| — | — | — | (all rows exact) | — |\n");

  if (!dump_path.empty()) {
    std::ofstream dump(dump_path);
    for (const RowResult& row : rows) {
      dump << row.source << '\t' << row.target << '\n';
    }
    std::printf("\nwrote %s\n", dump_path.c_str());
  }

  if (row_rate + 1e-9 < require_rate) {
    std::fprintf(stderr,
                 "FAIL: numeric exact-match %.3f below required %.3f\n",
                 row_rate, require_rate);
    return 1;
  }
  return 0;
}
