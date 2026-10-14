// Unit tests for Model construction: package wiring (mmap or View),
// vocabulary aliasing, shortlist loading, ids, and fail-fast behaviour on
// missing or corrupt model files.

#include <fstream>
#include <memory>
#include <string>
#include <sys/stat.h>

#include "leanmt_test.hh"
#include "src/Model.hh"

namespace {

using leanmt::Model;

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

std::string fixture_2v(const std::string& file) {
  return leanmt::test::fixture_2v_dir() + "/" + file;
}

bool fixture_2v_available() {
  struct stat st;
  return ::stat(fixture_2v("model.enja.intgemm.alphas.bin").c_str(), &st) ==
         0;
}

#define REQUIRE_FIXTURE_2V()                                       \
  if (!fixture_2v_available()) {                                   \
    std::cout << "    skip: two-vocab fixtures not present (run "  \
                 "tests/fixtures/download-fixtures.sh)\n";         \
    return;                                                        \
  }

// en-ja v2.3 (Firefox remote-settings): 6 encoder layers, **4** decoder
// layers, 8 heads, ffn depth 2 per the embedded model.yml. Decoder depth
// differs from the tiny11 shared-vocab fixture (2) — pin it explicitly.
leanmt::Model::Config enja_config() {
  leanmt::Model::Config config;
  config.encoder_layers = 6;
  config.decoder_layers = 4;
  config.feed_forward_depth = 2;
  config.num_heads = 8;
  return config;
}

}  // namespace

TEST("model loads from mmap package with shared-vocab aliasing") {
  REQUIRE_FIXTURE();
  leanmt::Package<std::string> files{
      .model = fixture("model.intgemm.alphas.bin"),
      .vocabulary = fixture("vocab.deen.spm"),
      .target_vocabulary = "",
      .shortlist = fixture("lex.s2t.bin"),
  };
  auto model = std::make_shared<Model>(Model::Config{}, files);

  CHECK_EQ(model->config().encoder_layers, 6u);
  CHECK(model->vocabulary().size() > 1000);
  // Shared-vocab model: target vocabulary aliases the source vocabulary.
  CHECK_EQ(&model->target_vocabulary(), &model->vocabulary());
  // The pinned fixture ships a shortlist (lex.s2t.bin).
  CHECK(model->shortlist_generator().has_value());
}

TEST("model without shortlist still loads, generator is empty") {
  REQUIRE_FIXTURE();
  leanmt::Package<std::string> files{
      .model = fixture("model.intgemm.alphas.bin"),
      .vocabulary = fixture("vocab.deen.spm"),
      .target_vocabulary = "",
      .shortlist = "",
  };
  auto model = std::make_shared<Model>(Model::Config{}, files);
  CHECK(!model->shortlist_generator().has_value());
}

TEST("model ids are unique per construction") {
  REQUIRE_FIXTURE();
  leanmt::Package<std::string> files{
      .model = fixture("model.intgemm.alphas.bin"),
      .vocabulary = fixture("vocab.deen.spm"),
      .target_vocabulary = "",
      .shortlist = fixture("lex.s2t.bin"),
  };
  auto first = std::make_shared<Model>(Model::Config{}, files);
  auto second = std::make_shared<Model>(Model::Config{}, files);
  CHECK(first->id() != second->id());
}

TEST("missing model file fails fast with a clear error") {
  leanmt::Package<std::string> files{
      .model = "/definitely/missing/model.bin",
      .vocabulary = "",
      .target_vocabulary = "",
      .shortlist = "",
  };
  CHECK_THROWS(Model(Model::Config{}, files), "Failed to open file");
}

TEST("corrupt model content fails fast, not with a crash") {
  // Starts with the right version but is otherwise garbage: the loader must
  // reject it deterministically (bounds-checked reads), no SIGSEGV.
  const char* path = "leanmt_model_test_corrupt.bin";
  {
    std::ofstream out(path, std::ios::binary);
    uint64_t version = leanmt::kBinaryFileVersion;
    char garbage[256];
    for (int i = 0; i < 256; ++i) garbage[i] = static_cast<char>(i * 31 + 7);
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(garbage, sizeof(garbage));
  }
  leanmt::Package<std::string> files{
      .model = path,
      .vocabulary = "",
      .target_vocabulary = "",
      .shortlist = "",
  };
  bool threw = false;
  try {
    Model model(Model::Config{}, files);
  } catch (const std::exception&) {
    threw = true;
  }
  std::remove(path);
  CHECK(threw);
}

TEST("two-vocab model loads with separate source/target vocabularies") {
  REQUIRE_FIXTURE_2V();
  leanmt::Package<std::string> files{
      .model = fixture_2v("model.enja.intgemm.alphas.bin"),
      .vocabulary = fixture_2v("srcvocab.enja.spm"),
      .target_vocabulary = fixture_2v("trgvocab.enja.spm"),
      .shortlist = fixture_2v("lex.50.50.enja.s2t.bin"),
  };
  auto model = std::make_shared<Model>(enja_config(), files);

  CHECK_EQ(model->config().encoder_layers, 6u);
  CHECK_EQ(model->config().decoder_layers, 4u);
  // Two-vocab model: target vocabulary must NOT alias the source.
  CHECK(&model->target_vocabulary() != &model->vocabulary());
  CHECK_EQ(model->vocabulary().size(), 32000u);
  CHECK_EQ(model->target_vocabulary().size(), 32000u);
  CHECK(model->shortlist_generator().has_value());
}

TEST("two-vocab model with empty target vocabulary fails fast") {
  REQUIRE_FIXTURE_2V();
  leanmt::Package<std::string> files{
      .model = fixture_2v("model.enja.intgemm.alphas.bin"),
      .vocabulary = fixture_2v("srcvocab.enja.spm"),
      .target_vocabulary = "",  // bug the JNI bridge must not get away with
      .shortlist = fixture_2v("lex.50.50.enja.s2t.bin"),
  };
  CHECK_THROWS(Model(enja_config(), files),
               "two-vocabulary model");
}

int main() { return leanmt::test::run_all(); }
