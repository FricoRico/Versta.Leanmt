// Unit tests for Model construction: package wiring (mmap or View),
// vocabulary aliasing, shortlist loading, ids, and fail-fast behaviour on
// missing or corrupt model files.

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "leanmt_test.hh"
#include "src/Model.hh"

namespace {

using leanmt::Model;

std::string fixture(const std::string& file) {
  return leanmt::test::fixture_dir() + "/" + file;
}

// Writes a minimal bergamot-layout binary model whose only tensors are one
// f32 marker per requested layer (`encoder_l<N>_marker` / `decoder_l<N>_marker`
// for N in 1..count). This is all architecture reconciliation reads; real
// weights are absent by design — these files exist to exercise the NEU-134
// depth-mismatch fail-fast path, which throws before any weight load.
void write_synthetic_model(const std::string& path, size_t encoder_layers,
                           size_t decoder_layers) {
  std::vector<std::string> names;
  for (size_t i = 1; i <= encoder_layers; ++i) {
    names.push_back("encoder_l" + std::to_string(i) + "_marker");
  }
  for (size_t i = 1; i <= decoder_layers; ++i) {
    names.push_back("decoder_l" + std::to_string(i) + "_marker");
  }

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  auto put_u64 = [&out](uint64_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  };
  put_u64(leanmt::kBinaryFileVersion);
  put_u64(names.size());
  for (const std::string& name : names) {  // headers
    put_u64(name.size() + 1);              // name length incl. NUL
    put_u64(0x0400 + sizeof(float));       // element type: f32
    put_u64(1);                            // shape length (one dim)
    put_u64(sizeof(float));                // data length
  }
  for (const std::string& name : names) {  // NUL-terminated names
    out.write(name.c_str(), name.size() + 1);
  }
  for (size_t i = 0; i < names.size(); ++i) {  // shapes: [1]
    int32_t dim = 1;
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
  }
  put_u64(0);  // no padding before the data section
  for (size_t i = 0; i < names.size(); ++i) {  // payloads
    float value = 0.0F;
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  }
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

TEST("deeper model file than config fails fast, naming decoder_layers") {
  // Shipped-generation packs carry a dec4 decoder (NEU-134); a default
  // enc6/dec2 config must abort at load, not translate into exit-0 garbage.
  const char* path = "leanmt_model_test_arch_deeper.bin";
  write_synthetic_model(path, /*encoder_layers=*/6, /*decoder_layers=*/4);
  leanmt::Package<std::string> files{
      .model = path,
      .vocabulary = "",
      .target_vocabulary = "",
      .shortlist = "",
  };
  CHECK_THROWS(Model(Model::Config{}, files), "Model::Config.decoder_layers");
  CHECK_THROWS(Model(Model::Config{}, files), "carries 4 decoder_l layer(s)");
  std::remove(path);
}

TEST("shallower model file than config fails fast, naming encoder_layers") {
  // The opposite direction: config requests a deeper encoder than the file
  // ships. Same fail-fast contract, named on the encoder side.
  const char* path = "leanmt_model_test_arch_shallower.bin";
  write_synthetic_model(path, /*encoder_layers=*/4, /*decoder_layers=*/2);
  Model::Config config;
  config.encoder_layers = 6;
  leanmt::Package<std::string> files{
      .model = path,
      .vocabulary = "",
      .target_vocabulary = "",
      .shortlist = "",
  };
  CHECK_THROWS(Model(config, files), "Model::Config.encoder_layers");
  CHECK_THROWS(Model(config, files), "carries 4 encoder_l layer(s)");
  std::remove(path);
}

int main() { return leanmt::test::run_all(); }
