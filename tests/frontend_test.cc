// End-to-end Frontend tests: block translate through the real (pinned,
// checksum-verified) FirefoxMT tiny11 model. These pin the contract the JNI
// bridge and app rely on: same model + same input -> same output, one input
// sentence -> one response sentence, alignments well-formed when requested.

#include <cmath>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "leanmt_test.hh"
#include "src/Frontend.hh"
#include "src/Model.hh"

namespace {

std::string fixture(const std::string& file) {
  return leanmt::test::fixture_dir() + "/" + file;
}

bool fixture_available() {
  struct stat st;
  return ::stat(fixture("model.intgemm.alphas.bin").c_str(), &st) == 0;
}

#define REQUIRE_FIXTURE()                                          \
  if (!fixture_available()) {                                      \
    std::cout << "    skip: fixtures not present (run "            \
                 "tests/fixtures/download-fixtures.sh)\n";         \
    return;                                                        \
  }

leanmt::Ptr<leanmt::Model> load_model() {
  leanmt::Package<std::string> files{
      .model = fixture("model.intgemm.alphas.bin"),
      .vocabulary = fixture("vocab.deen.spm"),
      .target_vocabulary = "",
      .shortlist = fixture("lex.s2t.bin"),
  };
  return std::make_shared<leanmt::Model>(leanmt::Model::Config{}, files);
}

}  // namespace

TEST("translate is deterministic across repeated runs") {
  REQUIRE_FIXTURE();
  auto model = load_model();
  // Two separate Blocking services: the per-service translation cache would
  // otherwise answer the second run without decoding, and we'd be testing
  // the cache, not decode determinism.
  leanmt::Blocking first_service(leanmt::Config{});
  leanmt::Blocking second_service(leanmt::Config{});
  leanmt::Options options;

  const std::string input = "The menu is not available in English.";
  auto first = first_service.translate(model, {input}, options);
  auto second = second_service.translate(model, {input}, options);

  CHECK_EQ(first.size(), 1u);
  CHECK_EQ(second.size(), 1u);
  CHECK_EQ(first[0].target.text, second[0].target.text);
  CHECK(!first[0].target.text.empty());
}

TEST("translate pins the golden output of the reference model (en-de tiny)") {
  REQUIRE_FIXTURE();
  auto model = load_model();
  leanmt::Blocking service(leanmt::Config{});
  leanmt::Options options;

  // Golden recorded from the pinned fixture (en-de tiny11 v2). If a SIMD
  // dispatch change flips an argmax here it wants a human eye, not a silent
  // update of this string.
  const std::string input = "Hello world, this is a small test.";
  auto responses = service.translate(model, {input}, options);
  CHECK_EQ(responses.size(), 1u);
  CHECK_EQ(responses[0].target.text, "Hallo Welt, das ist ein kleiner Test.");
  CHECK_EQ(responses[0].source.sentence_count(), 1u);
  CHECK_EQ(responses[0].target.sentence_count(), 1u);
}

TEST("translate handles a batch preserving order and one response per input") {
  REQUIRE_FIXTURE();
  auto model = load_model();
  leanmt::Blocking service(leanmt::Config{});
  leanmt::Options options;

  std::vector<std::string> batch = {
      "Hello world, this is a small test.",
      "The menu is not available in English.",
      "Where is the central station?",
  };
  auto responses = service.translate(model, batch, options);
  CHECK_EQ(responses.size(), batch.size());
  for (const auto& response : responses) {
    CHECK_EQ(response.size(), 1u);
    CHECK(!response.target.text.empty());
  }
  // First row must still be the golden translation — batch composition must
  // not change a sentence's own candidates.
  CHECK_EQ(responses[0].target.text, "Hallo Welt, das ist ein kleiner Test.");
}

TEST("batched translation is row-wise identical to single translations") {
  REQUIRE_FIXTURE();
  auto model = load_model();
  leanmt::Options options;

  // Six rows: past the NEON gemv row limit, so on aarch64 the batch's GEMMs
  // leave the single-row sdot path and go through ruy. Either way, each
  // sentence's output must equal its single-translate result exactly: batch
  // composition must not change a sentence's decode.
  std::vector<std::string> batch = {
      "Hello world, this is a small test.",
      "The menu is not available in English.",
      "Where is the central station?",
      "I would like to pay by card, please.",
      "Does this train stop at the harbour?",
      "My hotel reservation starts tomorrow.",
  };

  leanmt::Blocking singles_service(leanmt::Config{});
  std::vector<std::string> singles;
  for (const auto& sentence : batch) {
    auto r = singles_service.translate(model, {sentence}, options);
    REQUIRE(r.size() == 1u);
    singles.push_back(r[0].target.text);
  }

  leanmt::Blocking batch_service(leanmt::Config{});
  auto responses = batch_service.translate(model, batch, options);
  REQUIRE(responses.size() == batch.size());
  for (size_t i = 0; i < batch.size(); ++i) {
    CHECK_EQ(responses[i].target.text, singles[i]);
  }
}

TEST("alignments, when requested, are well-formed") {
  REQUIRE_FIXTURE();
  auto model = load_model();
  leanmt::Blocking service(leanmt::Config{});
  leanmt::Options options;
  options.alignment = true;

  const std::string input = "The menu is not available in English.";
  auto responses = service.translate(model, {input}, options);
  CHECK_EQ(responses.size(), 1u);

  const auto& response = responses[0];
  CHECK_EQ(response.alignments.size(), response.size());
  for (const auto& matrix : response.alignments) {
    CHECK(!matrix.empty());
    for (const auto& row : matrix) {
      for (float p : row) {
        CHECK(p >= 0.0f);
        CHECK(p <= 1.0f);
        CHECK(std::isfinite(p));
      }
    }
  }
}

TEST("translation stays correct across model destruct/reload cycles") {
  REQUIRE_FIXTURE();
  // Regression test: ruy's prepacked-RHS cache keys on the source tensor's
  // data pointer. Destroying a Model and loading another recycles the same
  // heap/mmap addresses; the never-cleared per-thread cache then served the
  // previous model's packs to the new one, corrupting batched decodes with
  // the old model's weights. The pattern below (two single-sentence models
  // destroyed, then a batch translate on a third) reproduced it 100%.
  const std::string other = "The menu is not available in English.";
  const std::string golden = "Hello world, this is a small test.";
  leanmt::Options options;
  for (int round = 0; round < 2; ++round) {
    auto model = load_model();
    leanmt::Blocking service(leanmt::Config{});
    (void)service.translate(model, {other}, options);
  }

  auto model = load_model();
  leanmt::Blocking service(leanmt::Config{});
  auto responses = service.translate(
      model, {golden, other, "Where is the central station?"}, options);

  CHECK_EQ(responses.size(), 3u);
  CHECK_EQ(responses[0].target.text, "Hallo Welt, das ist ein kleiner Test.");
  CHECK_EQ(responses[1].target.text,
           "Das Menü ist nicht auf Englisch verfügbar.");
  CHECK_EQ(responses[2].target.text, "Wo ist der Hauptbahnhof?");
}

int main() { return leanmt::test::run_all(); }
