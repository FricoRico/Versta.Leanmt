// Unit tests for the SentencePiece-backed Vocabulary: encoding is
// deterministic and ids stay in range; decode round-trips word text; the
// word-boundary helpers classify SentencePiece pieces correctly.

#include <string>
#include <sys/stat.h>

#include "leanmt_test.hh"
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

std::string vocab_2v_path(const std::string& file) {
  return leanmt::test::fixture_2v_dir() + "/" + file;
}

bool fixture_2v_available() {
  struct stat st;
  return ::stat(vocab_2v_path("srcvocab.enja.spm").c_str(), &st) == 0 &&
         ::stat(vocab_2v_path("trgvocab.enja.spm").c_str(), &st) == 0;
}

#define REQUIRE_FIXTURE_2V()                                       \
  if (!fixture_2v_available()) {                                   \
    std::cout << "    skip: two-vocab fixtures not present (run "  \
                 "tests/fixtures/download-fixtures.sh)\n";         \
    return;                                                        \
  }

}  // namespace

TEST("vocabulary loads and tokenizes deterministically") {
  REQUIRE_FIXTURE();
  leanmt::Vocabulary vocab(vocab_path());
  CHECK(vocab.size() > 1000);

  const std::string line = "Hello world, this is a small test.";
  auto [words_a, views_a] = vocab.encode(line, true);
  auto [words_b, views_b] = vocab.encode(line, true);
  CHECK(words_a.size() > 2);
  CHECK(words_a == words_b);
  for (leanmt::Word word : words_a) {
    CHECK(word < vocab.size());
  }
  // With add_eos=true the terminal word must be EOS.
  CHECK_EQ(words_a.back(), vocab.eos_id());
}

TEST("vocabulary decode round-trips recognizable words") {
  REQUIRE_FIXTURE();
  leanmt::Vocabulary vocab(vocab_path());
  auto [words, views] = vocab.encode("Das Haus ist klein.", false);
  std::string decoded;
  vocab.decode(words, decoded, false);
  CHECK(decoded.find("Das") != std::string::npos);
  CHECK(decoded.find("Haus") != std::string::npos);
}

TEST("word boundary helpers classify sentencepiece pieces") {
  REQUIRE_FIXTURE();
  leanmt::Vocabulary vocab(vocab_path());

  auto [words, views] = vocab.encode("industrialization.", false);
  CHECK(words.size() >= 2);
  // First piece of a sentence starts a word (metaspace prefix).
  CHECK(vocab.is_word_start(words.front()));
  // The trailing full stop is its own token and ends the word.
  CHECK(vocab.ends_word(words.back()));
  // Out-of-range lookups degrade gracefully instead of reading past tables.
  CHECK(!vocab.is_word_start(vocab.size() + 7));
  CHECK(vocab.ends_word(vocab.size() + 7));
}

TEST("two-vocab fixture ships genuinely distinct source/target vocabularies") {
  REQUIRE_FIXTURE_2V();
  leanmt::Vocabulary src(vocab_2v_path("srcvocab.enja.spm"));
  leanmt::Vocabulary trg(vocab_2v_path("trgvocab.enja.spm"));
  CHECK_EQ(src.size(), 32000u);
  CHECK_EQ(trg.size(), 32000u);

  // Same input, different ids: guards against the app accidentally wiring
  // srcvocab.spm into both package slots (which would otherwise only surface
  // as garbage Japanese output).
  const std::string line = "Where is the central station?";
  auto [src_words, src_views] = src.encode(line, true);
  auto [trg_words, trg_views] = trg.encode(line, true);
  CHECK(src_words.size() > 2);
  CHECK(trg_words.size() > 2);
  CHECK(!(src_words == trg_words));

  // The target (Japanese) model detokenizes Japanese script; the source
  // (English) model doesn't. This is the property decode relies on.
  auto [words, views] = trg.encode("東京駅はどこですか。", false);
  std::string decoded;
  trg.decode(words, decoded, false);
  CHECK(decoded.find("東京") != std::string::npos);
}

int main() { return leanmt::test::run_all(); }
