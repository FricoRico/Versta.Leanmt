#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <tuple>

#include "sentencepiece_processor.h"
#include "Types.hh"

namespace leanmt {

// Decodes one UTF-8 codepoint from the front of `s`, advancing past it.
// Returns false on empty input, a truncated sequence, a stray continuation
// byte or an overlong encoding — callers classify malformed UTF-8 as
// "not a digit piece", which is the safe direction (worst case is a piece
// not force-admitted, never a non-digit admitted).
inline bool next_utf8_codepoint(std::string_view &s, uint32_t &cp) {
  if (s.empty()) return false;
  const auto byte = [&](size_t i) {
    return static_cast<unsigned char>(s[i]);
  };
  const auto trail_ok = [&](size_t i) {
    return i < s.size() && (byte(i) & 0xC0) == 0x80;
  };
  unsigned char c0 = byte(0);
  if (c0 < 0x80) {
    cp = c0;
    s.remove_prefix(1);
    return true;
  }
  if ((c0 & 0xE0) == 0xC0) {
    if (c0 < 0xC2 || !trail_ok(1)) return false;  // overlong / malformed
    cp = ((c0 & 0x1F) << 6) | (byte(1) & 0x3F);
    s.remove_prefix(2);
    return true;
  }
  if ((c0 & 0xF0) == 0xE0) {
    if (!trail_ok(1) || !trail_ok(2)) return false;
    cp = ((c0 & 0x0F) << 12) | ((byte(1) & 0x3F) << 6) | (byte(2) & 0x3F);
    if (cp < 0x800) return false;  // overlong
    s.remove_prefix(3);
    return true;
  }
  if ((c0 & 0xF8) == 0xF0) {
    if (!trail_ok(1) || !trail_ok(2) || !trail_ok(3)) return false;
    cp = ((c0 & 0x07) << 18) | ((byte(1) & 0x3F) << 12) |
         ((byte(2) & 0x3F) << 6) | (byte(3) & 0x3F);
    if (cp < 0x10000 || cp > 0x10FFFF) return false;  // overlong / out of range
    s.remove_prefix(4);
    return true;
  }
  return false;
}

// Whether `cp` is a number digit the digit-forcing guarantee covers
// (NEU-30, scripts extended per the PR #8 review / NEU-135): ASCII 0-9
// (U+0030–U+0039), Arabic-Indic (U+0660–U+0669) and Extended Arabic-Indic /
// Persian (U+06F0–U+06F9) — exactly the scripts the pack-acceptance probe
// (Versta.Models versta.probe.metrics.digit_preservation) transliterates to
// ASCII before comparing number values, and the only non-ASCII Nd blocks
// the shipped pair roster's everyday orthography uses (ar writes U+0660,
// fa writes U+06F0; everything else on the roster writes ASCII).
// Explicitly not covered: fullwidth digits U+FF10–U+FF19 (the probe does
// not transliterate them either — fullwidth output already fails pack
// acceptance; cover engine and probe together if a shipped ja/zh corpus
// ever shows them) and non-Nd number characters (CJK numerals 一二三 are
// Lo, not Nd). See the digit-script coverage matrix in the changelog.
inline bool is_number_digit(uint32_t cp) {
  return (cp >= 0x0030 && cp <= 0x0039) || (cp >= 0x0660 && cp <= 0x0669) ||
         (cp >= 0x06F0 && cp <= 0x06F9);
}

// Whether `text` is a non-empty UTF-8 string of covered number digits
// (any mix of covered blocks; mixed digit/non-digit pieces and malformed
// UTF-8 return false).
inline bool is_digit_text(std::string_view text) {
  if (text.empty()) return false;
  uint32_t cp = 0;
  while (!text.empty()) {
    if (!next_utf8_codepoint(text, cp) || !is_number_digit(cp)) return false;
  }
  return true;
}

class Vocabulary {
 public:
  explicit Vocabulary(const std::string &fpath);
  explicit Vocabulary(View view);
  std::tuple<Words, Views> encode(const std::string_view &line,
                                  bool add_eos = false) const;
  Views decode(const Words &words, std::string &decoded,
               bool ignore_eos = true) const;

  Word pad_id() const { return std::max(0, processor_.pad_id()); }
  Word eos_id() const { return processor_.eos_id(); }
  size_t size() const { return processor_.GetPieceSize(); }

  // Whether `word`'s SentencePiece piece begins a new word — its piece starts
  // with the metaspace marker U+2581 ("▁", bytes E2 96 81). Continuation
  // subwords and control symbols return false.
  bool is_word_start(Word word) const {
    if (word >= size()) return false;
    const std::string &piece = processor_.IdToPiece(static_cast<int>(word));
    return piece.size() >= 3 &&
           static_cast<unsigned char>(piece[0]) == 0xE2 &&
           static_cast<unsigned char>(piece[1]) == 0x96 &&
           static_cast<unsigned char>(piece[2]) == 0x81;
  }

  // Whether `word`'s piece consists of number digits only (per
  // is_number_digit: ASCII, Arabic-Indic and Extended Arabic-Indic /
  // Persian), ignoring a leading metaspace marker: both word-start ("▁404")
  // and continuation ("04") number pieces qualify. Mixed pieces ("1st",
  // "B52", "٤٥ريال"), the bare marker, control symbols and out-of-range ids
  // do not. Numbers must stay copyable no matter what segmentations the
  // lexical shortlist table learned, so the shortlist generator keys on
  // this to admit every digit-only target piece whenever the source
  // carries one — which, with non-Latin source scripts (fa→en: ۴۵۰۰),
  // fires on the same classification.
  //
  // Single IdToPiece lookup: the marker is stripped inline (three fixed
  // bytes) rather than via is_word_start(), which would look the piece up
  // a second time.
  bool is_digit_piece(Word word) const {
    if (word >= size()) return false;
    const std::string &piece = processor_.IdToPiece(static_cast<int>(word));
    size_t begin = 0;
    if (piece.size() >= 3 &&
        static_cast<unsigned char>(piece[0]) == 0xE2 &&
        static_cast<unsigned char>(piece[1]) == 0x96 &&
        static_cast<unsigned char>(piece[2]) == 0x81) {
      begin = 3;  // strip "▁"
    }
    return is_digit_text(std::string_view(piece).substr(begin));
  }

  // Whether `word` ends the current word: it starts a new word (▁), or its
  // piece begins with ASCII punctuation (a standalone token like "." or ","
  // that attaches without a space). Mid-word continuation subwords — including
  // multi-byte accented letters — return false, so expansion keeps consuming
  // them.
  bool ends_word(Word word) const {
    if (word >= size()) return true;
    if (is_word_start(word)) return true;
    const std::string &piece = processor_.IdToPiece(static_cast<int>(word));
    if (piece.empty()) return true;
    unsigned char c0 = static_cast<unsigned char>(piece[0]);
    bool word_char = (c0 & 0x80) || std::isalnum(c0);
    return !word_char;
  }

 private:
  sentencepiece::SentencePieceProcessor processor_;
};

}  // namespace leanmt
