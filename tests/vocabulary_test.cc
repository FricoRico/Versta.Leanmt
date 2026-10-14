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

TEST("is_digit_piece classifies digit-only pieces (with and without ▁)") {
  REQUIRE_FIXTURE();
  leanmt::Vocabulary vocab(vocab_path());

  // Digit-only strings tokenize to digit-only pieces, both word-start
  // (▁-prefixed) and continuation pieces — every piece of every encode
  // below must classify as a digit piece.
  for (const char* digits : {"404", "7", "7392846", "06", "12000"}) {
    auto [words, views] = vocab.encode(digits, false);
    REQUIRE(!words.empty());
    for (leanmt::Word word : words) {
      CHECK(vocab.is_digit_piece(word));
    }
  }

  // Word-start and continuation digit pieces both occur in context:
  // "Room 404" yields a digit-only final piece (marked or continuation,
  // depending on context segmentation).
  auto [room, room_views] = vocab.encode("Room 404", false);
  REQUIRE(room.size() >= 2);
  CHECK(vocab.is_digit_piece(room.back()));

  // Words and punctuation carry no digit-only pieces.
  for (const char* text : {"the", "station", ".", ","}) {
    auto [words, views] = vocab.encode(text, false);
    for (leanmt::Word word : words) {
      CHECK(!vocab.is_digit_piece(word));
    }
  }

  // Mixed tokens like "B52" legitimately split into a letter piece plus a
  // digit-only continuation ("52"): at least one piece must classify as
  // non-digit (the part holding the letter), while digit-only parts may.
  auto [mixed, mixed_views] = vocab.encode("B52", false);
  bool any_nondigit = false;
  for (leanmt::Word word : mixed) {
    any_nondigit = any_nondigit || !vocab.is_digit_piece(word);
  }
  CHECK(any_nondigit);

  // Out-of-range and special ids are not digit pieces.
  CHECK(!vocab.is_digit_piece(vocab.size() + 7));
  CHECK(!vocab.is_digit_piece(vocab.eos_id()));
  CHECK(!vocab.is_digit_piece(vocab.pad_id()));
}

int main() { return leanmt::test::run_all(); }
