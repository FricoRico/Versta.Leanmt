// Unit tests for the SentencePiece-backed Vocabulary: encoding is
// deterministic and ids stay in range; decode round-trips word text; the
// word-boundary helpers classify SentencePiece pieces correctly.

#include <string>
#include <string_view>
#include <sys/stat.h>

#include "leanmt_test.hh"
#include "synthetic_vocabs.hh"
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

TEST("is_digit_text covers the shipped number scripts (table-driven)") {
  // Hermetic classification coverage — synthetic multi-byte pieces need no
  // fixture model. Covered: ASCII, Arabic-Indic (U+0660–0669), Extended
  // Arabic-Indic / Persian (U+06F0–06F9) — the scripts the shipped roster's
  // non-Latin sources (ar, fa) write prices, times and addresses in, and
  // the ones the pack-acceptance probe transliterates to ASCII (NEU-135).
  for (const char* digits :
       {"0", "9", "404", "12000", "۰", "۹۹", "۱۲", "۴۵۰۰", "۲۳", "٤٥٠٠",
        "١٢", "۱۲3" /* cross-block mix is still digit-only */}) {
    CHECK(leanmt::is_digit_text(digits));
  }
  // Excluded: mixed digit/non-digit pieces, punctuation forms adjacent to
  // the covered blocks, uncovered scripts (recorded pair-roster decisions),
  // and malformed UTF-8.
  for (const char* non_digits :
       {"", "1st", "B52", "١٢ريال", "۱۲.۵", "٪" /* U+066A percent */,
        "½" /* U+00BD vulgar fraction */, "１２３" /* fullwidth: not covered */,
        "二" /* CJK numeral: Lo, not Nd */, "१२" /* Devanagari: no shipped
        pair */, "\xC0\xB0" /* overlong '0' */, "\xD9" /* truncated */,
        "\xD9." /* bad trail */, " ", "▁"}) {
    CHECK(!leanmt::is_digit_text(non_digits));
  }
}

TEST("synthetic fa vocab classifies Eastern digit pieces (incl. ▁ stripping)") {
  // Vocabulary-level check on real SentencePiece objects: the synthetic
  // fa-style fixture vocab carries digit-only pieces in both covered
  // non-Latin Nd blocks (see tests/synthetic_vocabs.hh for why the pinned
  // ende-tiny11 vocab cannot serve here).
  const auto& vocabs = leanmt::test::synthetic_vocabs();
  REQUIRE(vocabs.ok);
  leanmt::Vocabulary fa(vocabs.fa_path);

  bool saw_arabic_indic = false, saw_extended = false, saw_marked = false;
  for (size_t i = 0; i < fa.size(); i++) {
    leanmt::Word id = static_cast<leanmt::Word>(i);
    // Decoded text tells us the script(s) in the piece; is_digit_piece must
    // agree with an independent re-derivation from the piece text.
    std::string text;
    fa.decode({id}, text, false);
    std::string_view rest = text;
    uint32_t cp = 0;
    bool has_digit = false, has_nondigit = false;
    bool in_066x = false, in_06Fx = false;
    while (leanmt::next_utf8_codepoint(rest, cp)) {
      if (cp >= 0x0660 && cp <= 0x0669) {
        has_digit = true;
        in_066x = true;
      } else if (cp >= 0x06F0 && cp <= 0x06F9) {
        has_digit = true;
        in_06Fx = true;
      } else if (cp >= 0x30 && cp <= 0x39) {
        has_digit = true;
      } else if (cp != 0x20) {
        has_nondigit = true;
      }
    }
    if (!has_digit) continue;
    if (has_nondigit) {
      // Mixed pieces (digit + non-digit, e.g. byte-fallback "<0xD9>") must
      // never classify as digit pieces.
      CHECK(!fa.is_digit_piece(id));
      continue;
    }
    CHECK(fa.is_digit_piece(id));
    saw_arabic_indic |= in_066x;
    saw_extended |= in_06Fx;
    // A digit-only piece that is also a word start carries the metaspace
    // marker — classifying it as digits exercises the marker-stripping path
    // on multi-byte digits.
    saw_marked |= fa.is_word_start(id);
  }
  CHECK(saw_arabic_indic);
  CHECK(saw_extended);
  CHECK(saw_marked);

  // Sanity on the same object: plain words and specials are not digits.
  auto [words, views] = fa.encode("قیمت ریال است", false);
  REQUIRE(!words.empty());
  for (leanmt::Word word : words) {
    CHECK(!fa.is_digit_piece(word));
  }
  CHECK(!fa.is_digit_piece(fa.eos_id()));
  CHECK(!fa.is_digit_piece(fa.size() + 7));

  // And the encode path exposes digit pieces for fa input — the
  // source-side trigger input of the shortlist test exists.
  auto [digits, dviews] = fa.encode("قیمت ۴۵۰۰ ریال است", false);
  bool any_digit = false;
  for (leanmt::Word word : digits) {
    any_digit = any_digit || fa.is_digit_piece(word);
  }
  CHECK(any_digit);
}

int main() { return leanmt::test::run_all(); }
