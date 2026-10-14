// Unit tests for TextProcessor: tokenize + wrap into model-sized segments.
// Sentence splitting lives in the caller (Rust side), so each process() call
// yields exactly one segment per wrap_length window, EOS-terminated.

#include <string>
#include <sys/stat.h>
#include <tuple>

#include "leanmt_test.hh"
#include "src/Annotation.hh"
#include "src/TextProcessor.hh"
#include "src/Vocabulary.hh"

namespace {

std::string vocab_path() {
  return leanmt::test::fixture_dir() + "/vocab.deen.spm";
}

bool fixture_available() {
  struct stat st;
  return ::stat(vocab_path().c_str(), &st) == 0;
}

#define REQUIRE_FIXTURE()                                          \
  if (!fixture_available()) {                                      \
    std::cout << "    skip: fixtures not present (run "            \
                 "tests/fixtures/download-fixtures.sh)\n";         \
    return;                                                        \
  }

// ~40 space-separated words, >256 subwords unneeded here; only length matters.
std::string long_sentence() {
  std::string out;
  for (int i = 0; i < 64; ++i) {
    out += "linguistic ";
  }
  out += "marathon.";
  return out;
}

}  // namespace

TEST("process returns a single EOS-terminated segment for short input") {
  REQUIRE_FIXTURE();
  leanmt::Vocabulary vocab(vocab_path());
  leanmt::TextProcessor processor(vocab);

  auto [source, segments] = processor.process("Hallo Welt.", /*wrap_length=*/128);
  CHECK_EQ(segments.size(), 1u);
  CHECK(!segments[0].empty());
  CHECK_EQ(segments[0].back(), vocab.eos_id());
  CHECK_EQ(source.sentence_count(), 1u);
}

TEST("process wraps long input into additional segments") {
  REQUIRE_FIXTURE();
  leanmt::Vocabulary vocab(vocab_path());
  leanmt::TextProcessor processor(vocab);

  auto [source, segments] = processor.process(long_sentence(), /*wrap_length=*/16);
  CHECK(segments.size() > 1);
  for (const auto& segment : segments) {
    // Every wrapped segment is capped at wrap_length and EOS-terminated.
    CHECK(segment.size() <= 16);
    CHECK_EQ(segment.back(), vocab.eos_id());
  }
  CHECK_EQ(source.sentence_count(), segments.size());
}

TEST("empty input produces no segments") {
  REQUIRE_FIXTURE();
  leanmt::Vocabulary vocab(vocab_path());
  leanmt::TextProcessor processor(vocab);

  auto [source, segments] = processor.process("", /*wrap_length=*/128);
  CHECK(segments.empty());
  CHECK_EQ(source.sentence_count(), 0u);
}

int main() { return leanmt::test::run_all(); }
