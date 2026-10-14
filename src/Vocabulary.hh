#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <string_view>
#include <tuple>

#include "sentencepiece_processor.h"
#include "Types.hh"

namespace leanmt {

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

  // Whether `word`'s piece consists of ASCII digits only (0-9), ignoring a
  // leading metaspace marker: both word-start ("▁404") and continuation
  // ("04") number pieces qualify. Mixed pieces ("1st", "B52"), the bare
  // marker, control symbols and out-of-range ids do not. Numbers must stay
  // copyable no matter what segmentations the lexical shortlist table
  // learned, so the shortlist generator keys on this to admit every
  // digit-only target piece whenever the source carries one.
  //
  // Single IdToPiece lookup: the marker is stripped inline (three fixed
  // bytes) rather than via is_word_start(), which would look the piece up
  // a second time. Scope is deliberately ASCII — the pairs we ship write
  // prices, times and addresses in 0-9; generalizing to Unicode Nd is a
  // separate decision (see the pair-roster risk note in NEU-30).
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
    if (begin == piece.size()) return false;
    return std::all_of(piece.begin() + static_cast<std::ptrdiff_t>(begin),
                       piece.end(), [](char c) {
                         return std::isdigit(static_cast<unsigned char>(c));
                       });
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
