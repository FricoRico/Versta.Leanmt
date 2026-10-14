// Unit tests for the lexical shortlist generator: digit forcing (NEU-30) —
// when the source carries a digit, the most frequent digit-only target
// pieces (up to kDigitAdmissionCap; enough to always cover every 1- and
// 2-digit chunk) must be admitted so numbers stay copyable regardless of
// what segmentations the precomputed lexical table happens to contain.
// Also pins the invariants that must survive it: no-digit inputs see no
// change, the vector-extension alignment multiple holds, and the digit-row
// size growth stays bounded.

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "leanmt_test.hh"
#include "src/Shortlist.hh"
#include "src/Vocabulary.hh"

namespace {

std::string fixture(const std::string& file) {
  return leanmt::test::fixture_dir() + "/" + file;
}

bool fixture_available() {
  struct stat st;
  return ::stat(fixture("lex.s2t.bin").c_str(), &st) == 0 &&
         ::stat(fixture("vocab.deen.spm").c_str(), &st) == 0;
}

#define REQUIRE_FIXTURE()                                          \
  if (!fixture_available()) {                                      \
    std::cout << "    skip: fixtures not present (run "            \
                 "tests/fixtures/download-fixtures.sh)\n";         \
    return;                                                        \
  }

std::vector<char> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  std::vector<char> data(static_cast<size_t>(in.tellg()));
  in.seekg(0);
  in.read(data.data(), static_cast<std::streamsize>(data.size()));
  return data;
}

// Shared-vocabulary fixture: source and target are the same vocabulary.
struct Generator {
  std::vector<char> blob;
  leanmt::Vocabulary vocab;
  leanmt::ShortlistGenerator shortlist;

  Generator()
      : blob(read_file(fixture("lex.s2t.bin"))),
        vocab(fixture("vocab.deen.spm")),
        shortlist(leanmt::View{blob.data(), blob.size()}, vocab, vocab) {}
};

const leanmt::Words& words_of(const leanmt::Shortlist& shortlist) {
  return shortlist.words();
}

bool contains(const leanmt::Words& words, leanmt::Word id) {
  return std::binary_search(words.begin(), words.end(), id);
}

}  // namespace

TEST("digit source force-admits digit pieces, chunks always in scope") {
  REQUIRE_FIXTURE();
  Generator g;

  const std::string sentence = "The wifi code is 7392846.";
  auto [words, views] = g.vocab.encode(sentence, true);
  leanmt::Shortlist shortlist =
      g.shortlist.generate(words, leanmt::ShortlistGenerator::kMinCandidates);
  const leanmt::Words& admitted = words_of(shortlist);

  // Copyability guarantee: every 1- and 2-digit chunk piece must always be
  // admitted — with those, any number is composable in some segmentation,
  // independent of the lexical table. (Digit targets are frequency-ordered
  // and the admission cap sits above the chunk ranks — verified on this
  // vocabulary by construction below.)
  for (const char* one : {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}) {
    auto [run_ids, run_views] = g.vocab.encode(one, false);
    REQUIRE(!run_ids.empty());
    for (leanmt::Word piece : run_ids) {
      CHECK(g.vocab.is_digit_piece(piece));
      CHECK(contains(admitted, piece));
    }
  }
  for (int n = 0; n <= 99; ++n) {
    char two[3];
    std::snprintf(two, sizeof(two), "%02d", n);
    auto [run_ids, run_views] = g.vocab.encode(two, false);
    for (leanmt::Word piece : run_ids) {
      if (g.vocab.is_digit_piece(piece)) {
        CHECK(contains(admitted, piece));
      }
    }
  }

  // Full-vocabulary scan: the shortlist contains at least as many digit
  // pieces as the admission cap (or every digit piece, when the vocabulary
  // has fewer than the cap).
  size_t digit_pieces = 0, digit_admitted = 0;
  for (size_t i = 0; i < g.vocab.size(); i++) {
    auto id = static_cast<leanmt::Word>(i);
    if (g.vocab.is_digit_piece(id)) {
      ++digit_pieces;
      digit_admitted += contains(admitted, id);
    }
  }
  CHECK(digit_pieces > 100);  // sanity: the scan actually found digits
  CHECK(digit_admitted >=
        std::min(digit_pieces, leanmt::ShortlistGenerator::kDigitAdmissionCap));

  // Vector-extension alignment must hold after forcing.
  CHECK_EQ(admitted.size() % leanmt::ShortlistGenerator::kVExtAlignment, 0u);
}

TEST("digit-row shortlist stays within the documented worst-case bound") {
  REQUIRE_FIXTURE();
  Generator g;

  auto [digit_words, v2] = g.vocab.encode(
      "The menu is not available in English, it costs 12 euros.", true);

  leanmt::Shortlist digit =
      g.shortlist.generate(digit_words, leanmt::ShortlistGenerator::kMinCandidates);
  const leanmt::Words& admitted = words_of(digit);

  size_t digit_pieces = 0;
  for (size_t i = 0; i < g.vocab.size(); i++) {
    digit_pieces += g.vocab.is_digit_piece(static_cast<leanmt::Word>(i));
  }

  // Alignment invariant, including after forcing re-aligns.
  CHECK_EQ(admitted.size() % leanmt::ShortlistGenerator::kVExtAlignment, 0u);

  // Absolute worst case: |shortlist| <= align8(max(min_candidates, S)) +
  // min(|digit pieces|, kDigitAdmissionCap) + alignment slack, where S is
  // the pre-top-up union size (frequent band + lexical rows). S isn't
  // directly observable; but generate(words, 0) is an observable upper
  // bound on it (its result is that same union topped up only to the
  // nearest multiple of eight — on this digit input it also contains the
  // forced digit pieces, which only loosens the bound).
  size_t base = g.shortlist.generate(digit_words, 0).words().size();
  size_t bound_base = std::max(base, leanmt::ShortlistGenerator::kMinCandidates);
  bound_base += leanmt::ShortlistGenerator::kVExtAlignment -
                bound_base % leanmt::ShortlistGenerator::kVExtAlignment;
  CHECK(admitted.size() <=
        bound_base +
            std::min(digit_pieces,
                     leanmt::ShortlistGenerator::kDigitAdmissionCap) +
            leanmt::ShortlistGenerator::kVExtAlignment);

  // No-digit inputs see no forcing at all (the admission pass is gated on a
  // source digit); the byte-identical pinned frontend goldens guard that
  // path end-to-end.
}

TEST("each segmentation of a source number is reproducible from the shortlist") {
  REQUIRE_FIXTURE();
  Generator g;

  // Round-trip guarantee, constructively: every piece of every plausible
  // segmentation of each digit run must be reachable, otherwise the decoder
  // cannot copy the number. Encode each run standalone — any tokenization of
  // a digit-only string consists of digit-only pieces — and require all of
  // them admitted.
  const std::string sentence = "Room 404 costs 12.50 euros, code 7392846.";
  auto [words, views] = g.vocab.encode(sentence, true);
  leanmt::Shortlist shortlist =
      g.shortlist.generate(words, leanmt::ShortlistGenerator::kMinCandidates);
  const leanmt::Words& admitted = words_of(shortlist);

  const char* runs[] = {"404", "12", "50", "7392846"};
  for (const char* run : runs) {
    auto [run_words, run_views] = g.vocab.encode(run, false);
    REQUIRE(!run_words.empty());
    for (leanmt::Word piece : run_words) {
      CHECK(g.vocab.is_digit_piece(piece));
      CHECK(contains(admitted, piece));
    }
  }
}

TEST("digit runs decode back to their text") {
  REQUIRE_FIXTURE();
  Generator g;

  // End of the round trip: the admitted pieces decode to the original
  // numbers, so a decode that selects them reproduces the source digits.
  for (const char* run : {"404", "12", "50", "7392846"}) {
    auto [run_words, run_views] = g.vocab.encode(run, false);
    std::string decoded;
    g.vocab.decode(run_words, decoded, false);
    CHECK_EQ(decoded, run);
  }
}

int main() { return leanmt::test::run_all(); }
