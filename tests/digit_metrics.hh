// Value-level, script-normalized digit comparison for the numeric-fidelity
// gate (NEU-30; scripts and clock values extended per the PR #8 review /
// NEU-135). Mirrors the number semantics of the pack-acceptance probe
// (Versta.Models versta/probe/metrics.py `digit_preservation`):
//
//   * Digit runs compare after transliterating Arabic-Indic (U+0660–0669)
//     and Extended Arabic-Indic / Persian (U+06F0–06F9) digits to ASCII —
//     "قیمت ۴۵۰۰ ریال" → "The fare is 4500 rials." passes, "...5400..."
//     fails ("۱۲" ≡ "12" because ۱ and ۲ map to 1 and 2).
//   * Clocks compare as 12h-normalized (hour, minute) value tuples on the
//     surfaces the probe extracts: colon clocks "18:45" / "6:45 pm",
//     Japanese "18時45分" / "10時", Korean "23시 30분" (duration
//     "9時間"/"9시간" excluded by the same "間/간 does not follow" rule),
//     and bare meridiem forms "6:45 pm" / "8pm". "18:45" passes against
//     "18時45分" and "6:45 pm"; "8:30" against "8시 20분" fails.
//
// Two deliberate divergences from the probe (recorded in the digit-script
// coverage matrix in CHANGELOG.md):
//   * Number words (probe's en/ca/es/fr maps: "twelve", "doce") are NOT
//     normalized — the engine gate requires digit surfaces, which is
//     stricter; word-spelling legitimacy is a pack-level, per-language
//     decision owned by the probe.
//   * The gate tracks extra output runs as well (doubling detection), so
//     `exact` is stricter than the probe's recall-only verdict;
//     `preserved` is the probe-equivalent direction.
//
// Hermetic: pure text functions, no model/vocab fixture required.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/Vocabulary.hh"

namespace leanmt::numeric {

// Transliterates covered non-ASCII digit scripts to ASCII
// (U+0660–U+0669 → 0-9, U+06F0–U+06F9 → 0-9; see is_number_digit for the
// block roster). All other bytes — including digits in scripts we do not
// cover and malformed UTF-8 — pass through unchanged.
inline std::string normalize_digit_scripts(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  std::string_view rest = text;
  uint32_t cp = 0;
  while (!rest.empty()) {
    std::string_view before = rest;
    if (!next_utf8_codepoint(rest, cp)) {
      // Malformed byte: keep it verbatim (and make progress).
      out.push_back(before.front());
      rest.remove_prefix(1);
      continue;
    }
    size_t size = before.size() - rest.size();
    if (cp >= 0x0660 && cp <= 0x0669) {
      out.push_back(static_cast<char>('0' + (cp - 0x0660)));
    } else if (cp >= 0x06F0 && cp <= 0x06F9) {
      out.push_back(static_cast<char>('0' + (cp - 0x06F0)));
    } else {
      out.append(before.substr(0, size));
    }
  }
  return out;
}

// A clock surface on the text: byte span plus its 12h-normalized value.
struct ClockMatch {
  size_t begin = 0;
  size_t end = 0;
  int hour = 0;    // already % 12
  int minute = 0;
};

namespace detail {

inline bool ascii_digit_at(const std::string &t, size_t p) {
  return p < t.size() && t[p] >= '0' && t[p] <= '9';
}

inline bool starts_with(const std::string &t, size_t p, const char *lit) {
  const std::string_view v(lit);
  return p + v.size() <= t.size() &&
         t.compare(p, v.size(), v.data(), v.size()) == 0;
}

// Meridiem ("am", "pm", "a.m.", "p.m.") at `p` (case-insensitive) with a
// wordish boundary; returns its byte length or 0.
inline size_t meridiem_at(const std::string &t, size_t p) {
  auto lower = [](char c) { return (c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c; };
  if (p >= t.size()) return 0;
  if (lower(t[p]) != 'a' && lower(t[p]) != 'p') return 0;
  size_t len = 0;
  if (p + 1 < t.size() && lower(t[p + 1]) == 'm') {
    len = 2;
  } else if (p + 3 < t.size() && t[p + 1] == '.' && lower(t[p + 2]) == 'm' &&
             t[p + 3] == '.') {
    len = 4;
  } else {
    return 0;
  }
  size_t after = p + len;
  if (after < t.size() &&
      (ascii_digit_at(t, after) || (t[after] >= 'A' && t[after] <= 'Z') ||
       (t[after] >= 'a' && t[after] <= 'z'))) {
    return 0;  // "am" glued to more word characters is not a meridiem
  }
  return len;
}

inline size_t skip_spaces(const std::string &t, size_t p) {
  while (p < t.size() && t[p] == ' ') ++p;
  return p;
}

}  // namespace detail

// Extracts clock values with their spans, in the probe's precedence: colon
// clocks first, then Japanese 時/分, Korean 시/분, then bare meridiem
// forms. Digit runs that form no clock are left for run comparison.
inline std::vector<ClockMatch> find_clocks(const std::string &t) {
  std::vector<ClockMatch> clocks;
  size_t i = 0;
  const size_t n = t.size();
  while (i < n) {
    if (!detail::ascii_digit_at(t, i)) {
      ++i;
      continue;
    }
    size_t d0 = i;
    while (i < n && detail::ascii_digit_at(t, i)) ++i;
    size_t d1 = i;
    const size_t run_len = d1 - d0;
    if (run_len <= 2) {
      const int value = std::stoi(t.substr(d0, run_len));
      // Colon clock: H?H \s*[:|：]\s* MM (optional meridiem).
      size_t colon_bytes = 0;
      if (detail::starts_with(t, i, ":")) colon_bytes = 1;
      else if (detail::starts_with(t, i, "\xEF\xBC\x9A")) colon_bytes = 3;
      size_t m0 = detail::skip_spaces(t, i + colon_bytes);
      if (colon_bytes && detail::ascii_digit_at(t, m0) &&
          detail::ascii_digit_at(t, m0 + 1)) {
        int minute = std::stoi(t.substr(m0, 2));
        size_t end = m0 + 2;
        // Optional meridiem directly after the minutes.
        size_t p = detail::skip_spaces(t, end);
        size_t mer = detail::meridiem_at(t, p);
        if (mer) end = p + mer;
        clocks.push_back({d0, end, value % 12, minute});
        i = end;
        continue;
      }
      // Japanese: H?H \s*時 (not \s*間) [\s*M?M\s*分].
      if (detail::starts_with(t, i, "\xE6\x99\x82") &&
          !detail::starts_with(t, detail::skip_spaces(t, i + 3),
                               "\xE9\x96\x93")) {
        size_t end = i + 3;
        int minute = 0;
        size_t p = detail::skip_spaces(t, end);
        if (detail::ascii_digit_at(t, p)) {
          size_t m0 = p;
          while (p < n && detail::ascii_digit_at(t, p) && p - m0 < 2) ++p;
          size_t after_digits = detail::skip_spaces(t, p);
          if (detail::starts_with(t, after_digits, "\xE5\x88\x86")) {
            minute = std::stoi(t.substr(m0, p - m0));
            end = after_digits + 3;
          }
        }
        clocks.push_back({d0, end, value % 12, minute});
        i = end;
        continue;
      }
      // Korean: H?H \s*시 (not \s*간) [\s*M?M\s*분].
      if (detail::starts_with(t, i, "\xEC\x8B\x9C") &&
          !detail::starts_with(t, detail::skip_spaces(t, i + 3),
                               "\xEA\xB0\x84")) {
        size_t end = i + 3;
        int minute = 0;
        size_t p = detail::skip_spaces(t, end);
        if (detail::ascii_digit_at(t, p)) {
          size_t m0 = p;
          while (p < n && detail::ascii_digit_at(t, p) && p - m0 < 2) ++p;
          size_t after_digits = detail::skip_spaces(t, p);
          if (detail::starts_with(t, after_digits, "\xEB\xB6\x84")) {
            minute = std::stoi(t.substr(m0, p - m0));
            end = after_digits + 3;
          }
        }
        clocks.push_back({d0, end, value % 12, minute});
        i = end;
        continue;
      }
      // Bare meridiem: the digit run is an hour when followed by am/pm and
      // not glued to a preceding digit or colon ("8:30 pm" already matched
      // as a colon clock above; a second match on "30 pm" is excluded;
      // 0xBA is the tail byte of the fullwidth colon U+FF1A).
      size_t p = detail::skip_spaces(t, i);
      size_t mer = detail::meridiem_at(t, p);
      bool preceded_by_clock = false;
      if (d0 > 0) {
        unsigned char prev = static_cast<unsigned char>(t[d0 - 1]);
        preceded_by_clock =
            (prev >= '0' && prev <= '9') || prev == ':' || prev == 0xBA;
      }
      if (mer && !preceded_by_clock) {
        clocks.push_back({d0, p + mer, value % 12, 0});
        i = p + mer;
        continue;
      }
    }
    // Not a clock: leave the run to the plain digit comparison.
  }
  return clocks;
}

// `text` with the given byte spans excised.
inline std::string without_spans(const std::string &text,
                                 std::vector<std::pair<size_t, size_t>> spans) {
  std::string out;
  size_t cursor = 0;
  for (const auto &[begin, end] : spans) {
    if (begin < cursor || end > text.size()) continue;  // overlap guard
    out.append(text, cursor, begin - cursor);
    cursor = end;
  }
  out.append(text, cursor, std::string::npos);
  return out;
}

// Maximal runs of ASCII digits with multiplicity: "12.50 euros" →
// {"12": 1, "50": 1}. Grouping/decimal-separator rewrites (12.50 → 12,50)
// do not register as digit loss; dropped, altered, fused ("404404") or
// duplicated runs do.
using DigitRuns = std::map<std::string, size_t>;

inline DigitRuns digit_runs(const std::string &text) {
  DigitRuns runs;
  size_t i = 0;
  while (i < text.size()) {
    if (text[i] >= '0' && text[i] <= '9') {
      size_t begin = i;
      while (i < text.size() && text[i] >= '0' && text[i] <= '9') ++i;
      ++runs[text.substr(begin, i - begin)];
    } else {
      ++i;
    }
  }
  return runs;
}

// Value-level comparison result for one (source, output) pair.
struct DigitReport {
  size_t expected = 0;   // source-side number carriers (runs + clock values)
  size_t matched = 0;    // present in the output with the same value
  size_t extra = 0;      // output-side carriers the source does not carry
  bool preserved = false;  // every source carrier survived (probe direction)
  bool exact = false;      // preserved && nothing extra (gate direction)
};

// Compares the number values of `source` and `output` after digit-script
// normalization. Every source clock value must appear among the output's
// clock values; matched clock spans are excised before plain digit runs
// compare as multisets.
inline DigitReport compare_digit_values(std::string_view source0,
                                        std::string_view output0) {
  const std::string source = normalize_digit_scripts(source0);
  const std::string output = normalize_digit_scripts(output0);

  const std::vector<ClockMatch> source_clocks = find_clocks(source);
  const std::vector<ClockMatch> output_clocks = find_clocks(output);

  // Every source clock must find an equal-valued output clock; matched
  // output spans (only matched ones, mirroring the probe) are excluded from
  // the plain-run comparison.
  std::map<std::pair<int, int>, size_t> unmatched_values;
  std::vector<std::pair<size_t, size_t>> source_spans;
  for (const ClockMatch &clock : source_clocks) {
    ++unmatched_values[{clock.hour, clock.minute}];
    source_spans.emplace_back(clock.begin, clock.end);
  }
  std::vector<std::pair<size_t, size_t>> consumed_output_spans;
  for (const ClockMatch &clock : output_clocks) {
    auto key = std::make_pair(clock.hour, clock.minute);
    auto it = unmatched_values.find(key);
    if (it != unmatched_values.end() && it->second > 0) {
      --it->second;
      consumed_output_spans.emplace_back(clock.begin, clock.end);
    }
  }

  DigitReport report;
  size_t clocks_missing = 0;
  for (const auto &[value, count] : unmatched_values) {
    (void)value;
    clocks_missing += count;
  }
  // Clock values count once each as expected carriers; matched ones count
  // as matched.
  report.expected = source_clocks.size();
  report.matched = source_clocks.size() - clocks_missing;

  const std::string source_rest = without_spans(source, source_spans);
  const std::string output_rest = without_spans(output, consumed_output_spans);

  DigitRuns want = digit_runs(source_rest);
  DigitRuns got = digit_runs(output_rest);
  size_t missing_runs = 0;
  for (const auto &[run, count] : want) {
    report.expected += count;
    size_t present = got.count(run) ? got[run] : 0;
    report.matched += std::min(count, present);
    missing_runs += count - std::min(count, present);
    if (present > count) report.extra += present - count;
  }
  for (const auto &[run, count] : got) {
    if (!want.count(run)) report.extra += count;
  }
  report.preserved = (missing_runs == 0 && clocks_missing == 0);
  report.exact = report.preserved && report.extra == 0;
  return report;
}

}  // namespace leanmt::numeric
