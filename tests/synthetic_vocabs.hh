// Synthetic two-vocabulary fixture (fa-style source with Eastern Arabic /
// Persian digit scripts, en-style target with ASCII digits) trained at test
// time by the bundled SentencePiece trainer. Why synthetic: the pinned
// ende-tiny11 fixture vocab contains no Eastern-Arabic digit *pieces* (the
// multi-byte digits visible in its bytes belong to the normalization
// charsmap, not the piece inventory), so the non-Latin digit-forcing
// regression (PR #8 review / NEU-135) needs its own tiny vocabularies.
// "Synthetic multi-byte pieces need no new fixture model": two ~120-piece
// unigram models train in well under a second, hermetically, into a
// per-process temp dir.
//
// The corpora deliberately exercise both covered non-ASCII Nd blocks
// (U+0660–0669 Arabic-Indic and U+06F0–06F9 Extended/Persian): every digit
// character appears, so character_coverage=1.0 guarantees a single-char
// piece per digit and digit-only multi-char pieces show up via frequency.
// Model files land in a mkdtemp dir and are reused for the process
// lifetime.

#pragma once

#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>

#include "sentencepiece_trainer.h"

namespace leanmt::test {

struct SyntheticVocabs {
  std::string fa_path;  // fa-style vocab: carries U+0660/U+06F0 digit pieces
  std::string en_path;  // en-style vocab: carries ASCII digit pieces
  bool ok = false;
};

inline SyntheticVocabs train_synthetic_vocabs() {
  SyntheticVocabs out;
  char dir_template[] = "/tmp/leanmt-synth-vc-XXXXXX";
  const char* dir = mkdtemp(dir_template);
  if (dir == nullptr) return out;
  const std::string root = dir;

  {
    std::ofstream fa(root + "/fa.txt");
    // Persian (U+06F0) everyday number contexts + an Arabic-Indic (U+0660)
    // line; every digit char of both blocks must appear at least once.
    for (int i = 0; i < 4; ++i) {
      fa << "قیمت ۴۵۰۰ ریال است\n"
            "بلیط موزه ۱۲ یورو است\n"
            "قطار ساعت ۲۳ ۳۰ حرکت می کند\n"
            "اتاق ۴ در طبقه دوم است\n"
            "۹۸۷۶۵۴۳۲۱۰ صفر یک دو سه\n"
            "عدد ١٢ و ٣٤ و ٥٦٧٨٩٠ بزرگ است\n";
    }
  }
  {
    std::ofstream en(root + "/en.txt");
    for (int i = 0; i < 4; ++i) {
      en << "the museum ticket costs 12 euros\n"
            "the fare is 4500 rials\n"
            "my train leaves at 23 30\n"
            "room 214 is on the second floor\n"
            "breakfast is served from 6 30 to 10 00\n"
            "i need 5 tickets and 2 coffees\n"
            "0 1 2 3 4 5 6 7 8 9 the station code is 7392846\n";
    }
  }

  const char* kTrainFlags =
      "--vocab_size=128 --model_type=unigram --character_coverage=1.0 "
      "--hard_vocab_limit=false";
  const std::string fa_args = "--input=" + root + "/fa.txt --model_prefix=" +
                              root + "/fa " + kTrainFlags;
  const std::string en_args = "--input=" + root + "/en.txt --model_prefix=" +
                              root + "/en " + kTrainFlags;
  if (!sentencepiece::SentencePieceTrainer::Train(fa_args).ok()) return out;
  if (!sentencepiece::SentencePieceTrainer::Train(en_args).ok()) return out;
  out.fa_path = root + "/fa.model";
  out.en_path = root + "/en.model";
  out.ok = true;
  return out;
}

// Process-wide shared instance (both test binaries pay the training cost
// once).
inline const SyntheticVocabs& synthetic_vocabs() {
  static const SyntheticVocabs vocabs = train_synthetic_vocabs();
  return vocabs;
}

}  // namespace leanmt::test
