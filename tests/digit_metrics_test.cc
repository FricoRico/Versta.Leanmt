// Unit tests for the value-level, script-normalized digit metric
// (tests/digit_metrics.hh; NEU-30, extended for NEU-135 non-Latin digit
// scripts per the PR #8 review). The (source, output) vectors are the
// pack-acceptance probe's own cases (Versta.Models
// translation/tests/test_probe.py::test_digit_preservation) — the engine
// gate and the pack probe must read number corruption the same way.
//
// Divergences are deliberate and pinned as such: where the probe passes on
// spelled-out number words (its en/ca/es/fr maps), the engine gate
// requires digit surfaces and fails — see the coverage matrix in
// CHANGELOG.md. Those cases are marked "words:" below.
//
// Hermetic: pure text functions — no model or vocab fixture needed.

#include <string>
#include <utility>
#include <vector>

#include "leanmt_test.hh"
#include "tests/digit_metrics.hh"

namespace {

using leanmt::numeric::DigitReport;
using leanmt::numeric::compare_digit_values;
using leanmt::numeric::digit_runs;
using leanmt::numeric::find_clocks;
using leanmt::numeric::normalize_digit_scripts;

bool preserved(const std::string& source, const std::string& output) {
  return compare_digit_values(source, output).preserved;
}

}  // namespace

TEST("digit-script transliteration maps covered blocks and passes the rest through") {
  CHECK_EQ(normalize_digit_scripts("۴۵۰۰"), "4500");   // Extended AI/Persian
  CHECK_EQ(normalize_digit_scripts("٤٥٠٠"), "4500");   // Arabic-Indic
  CHECK_EQ(normalize_digit_scripts("۱۲"), "12");
  CHECK_EQ(normalize_digit_scripts("١۲3"), "123");     // cross-block mix
  CHECK_EQ(normalize_digit_scripts("قیمت ۴۵۰۰ ریال"),
           "قیمت 4500 ریال");                          // text untouched
  CHECK_EQ(normalize_digit_scripts("no digits"), "no digits");
  CHECK_EQ(normalize_digit_scripts(""), "");
  // Deliberately NOT covered (see is_number_digit's roster decision):
  // fullwidth digits stay as they are (the pack probe also rejects them).
  CHECK_EQ(normalize_digit_scripts("１２３"), "１２３");
  // Malformed UTF-8 passes through byte-verbatim rather than eating input.
  CHECK_EQ(normalize_digit_scripts(std::string("\xD9")), std::string("\xD9"));
}

TEST("plain digit-run multisets") {
  auto runs = digit_runs("12.50 euros");
  CHECK_EQ(runs.size(), 2u);
  CHECK_EQ(runs["12"], 1u);
  CHECK_EQ(runs["50"], 1u);
  CHECK(digit_runs("no digits").empty());
}

TEST("clock extraction across locales (colon / meridiem / ja / ko)") {
  auto colon = find_clocks("18:45");
  REQUIRE(colon.size() == 1u);
  CHECK_EQ(colon[0].hour, 6);
  CHECK_EQ(colon[0].minute, 45);

   auto mer = find_clocks("6:45 pm");
  REQUIRE(mer.size() == 1u);
  CHECK_EQ(mer[0].hour, 6);
  CHECK_EQ(mer[0].minute, 45);

  auto bare = find_clocks("from 6pm to 10pm");
  REQUIRE(bare.size() == 2u);
  CHECK_EQ(bare[0].hour, 6);
  CHECK_EQ(bare[0].minute, 0);
  CHECK_EQ(bare[1].hour, 10);

  auto ja = find_clocks("18時45分に");
  REQUIRE(ja.size() == 1u);
  CHECK_EQ(ja[0].hour, 6);
  CHECK_EQ(ja[0].minute, 45);

  auto ko = find_clocks("23시 30분에");
  REQUIRE(ko.size() == 1u);
  CHECK_EQ(ko[0].hour, 11);
  CHECK_EQ(ko[0].minute, 30);

  // Durations stay out (9時間 / 9시간).
  CHECK(find_clocks("9時間").empty());
  CHECK(find_clocks("여행은 9시간 걸립니다.").empty());
  // German dot-style "18.45 Uhr" is not a clock surface (probe parity).
  CHECK(find_clocks("18.45 Uhr").empty());
}

TEST("probe vectors: digit and clock cases keep the probe verdict") {
  struct Case {
    std::string source;
    std::string output;
    bool expected;
  };
  const std::vector<Case> cases = {
      // verbatim digit copy
      {"The museum ticket costs 12 euros.",
       "Das Museumsticket kostet 12 Euro.", true},
      {"Room 214 is on the second floor.", "214号室は2階です。", true},
      // locale clock rendering
      {"My train leaves at 18:45 from platform 7.",
       "私の電車は18時45分に7番線から出ます。", true},
      {"Breakfast is served from 6:30 to 10:00.",
       "朝食は6時30分から10時まで提供されます。", true},
      {"私の電車は18時45分に出発します。",
       "My train leaves at 6:45 pm from platform 7.", true},
      // Eastern Arabic / Persian digit scripts (fa; NEU-135)
      {"بلیط موزه ۱۲ یورو است.", "The museum ticket costs 12 euros.", true},
      {"بلیط موزه ۱۲ یورو است.", "The museum ticket costs ۱۲ euros.", true},
      {"قیمت ۴۵۰۰ ریال است.", "The fare is 4500 rials.", true},
      {"قطار من ساعت ۲۳:۳۰ حرکت می‌کند.", "My train leaves at 23:30.", true},
      // Korean clock forms (en-ko; NEU-135)
      {"My train leaves at 23:30.", "제 기차는 23시 30분에 출발합니다.", true},
      {"The museum opens at 10:00.", "박물관은 10시에 문을 엽니다.", true},
      // duration stays out of the clock extractor
      {"여행은 9시간 걸립니다.", "The journey takes 9 hours.", true},
      // genuine corruption cases stay failing
      {"قیمت ۴۵۰۰ ریال است.", "The fare is 5400 rials.", false},
      {"The meeting starts at 8:30.", "회의는 8시 20분에 시작됩니다.", false},
      {"朝食は6時30分から10時までですか。", "Is breakfast from 6pm to 10pm?",
       false},
      {"The flight takes 9 hours and 25 minutes.",
       "El vuelo dura 9 horas y 5 minutos.", false},
      {"料金は4500円です。", "The fare is 5400 yen.", false},
      {"Someone stole my bag with 200 euros inside.",
       "Me robaron la bolsa con 20 euros dentro.", false},
      {"I need 3 tickets.", "Necesito 33 billetes.", false},
      {"Room 214.", "Habitación 1214.", false},
      {"Turn left and walk 200 meters.", "Gire a la izquierda.", false},
  };
  for (const Case& c : cases) {
    CHECK(preserved(c.source, c.output) == c.expected);
  }
}

TEST("probe divergence: spelled-out number words do not satisfy the engine gate") {
  // The probe passes these via its en/ca/es/fr number-word maps; the engine
  // gate deliberately requires digit surfaces (coverage matrix, CHANGELOG).
  const std::vector<std::pair<std::string, std::string>> word_cases = {
      {"子供2人を含めて、全部で5人です。",
       "There are five of us, including two children."},
      {"私の部屋は2階の214号室です。",
       "My room is the 214th room on the second floor."},
      {"10月3日に着いて、10月9日に出ます。",
       "I arrive on October 3 and leave on October 9."},
      {"博物館から徒歩15分です。", "The museum is a fifteen-minute walk away."},
      {"I need 5 tickets and 2 coffees.", "Necessito cinc bitllets i dos cafès."},
      {"I arrive on October 3.", "Llego el tres de octubre."},
      {"The ticket costs 12 euros.", "Le billet coûte douze euros."},
      {"I need 5 tickets and 2 coffees.", "Necessito sis bitllets i dos cafès."},
  };
  for (const auto& [source, output] : word_cases) {
    CHECK(!preserved(source, output));
  }
}

TEST("gate-only strictness: extra output carriers fail exactness, not recall") {
  // Doubling: "12 12" preserves the source value but doubles it — the probe
  // is recall-only; the gate's `exact` (and `extra`) still catches it.
  DigitReport doubled = compare_digit_values("The ticket costs 12 euros.",
                                             "Das Ticket kostet 12 12 Euro.");
  CHECK(doubled.preserved);
  CHECK(!doubled.exact);
  CHECK_EQ(doubled.extra, 1u);

  DigitReport fused = compare_digit_values("Room 404, code 404.",
                                           "Zimmer 404404.");
  CHECK(!fused.preserved);
  CHECK(!fused.exact);

  DigitReport clean = compare_digit_values("The ticket costs 12 euros.",
                                           "Das Ticket kostet 12 Euro.");
  CHECK(clean.preserved);
  CHECK(clean.exact);

  // fa output legitimately keeps Persian script — exact after normalization.
  DigitReport fa = compare_digit_values("قیمت ۴۵۰۰ ریال است.",
                                        "The fare is 4500 rials.");
  CHECK(fa.preserved);
  CHECK(fa.exact);

  // Long digit runs (codes, card numbers): must not overflow the clock
  // parser's hour probe.
  DigitReport longrun = compare_digit_values("code 73928465550192837746 ok",
                                             "Code 73928465550192837746.");
  CHECK(longrun.exact);
}

int main() { return leanmt::test::run_all(); }
