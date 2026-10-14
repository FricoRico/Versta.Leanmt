// Unit tests for MmapFile and the binary model-format loader (Io.cc).
// The loader must fail fast with a precise error on truncated/corrupt input —
// it used to march past the end of the mmap and crash.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "leanmt_test.hh"
#include "src/Io.hh"

namespace {

using leanmt::io::Header;

// Assemble a minimal well-formed model-format buffer: one f32 tensor named
// "Wemb" with shape (2, 4) and 32 bytes of payload (header.data_length is
// padded to 64 to mimic the 256-byte-aligned real format).
std::vector<char> minimal_model() {
  std::vector<char> out;
  auto put = [&](const void* p, size_t n) {
    const char* c = static_cast<const char*>(p);
    out.insert(out.end(), c, c + n);
  };
  auto put_u64 = [&](uint64_t v) { put(&v, sizeof(v)); };

  put_u64(leanmt::kBinaryFileVersion);  // version
  put_u64(1);                           // num_headers
  Header header{/*name_length=*/5, /*type=*/0x0404, /*shape_length=*/2,
                /*data_length=*/64};
  put(&header, sizeof(header));
  put("Wemb\0", 5);                          // name (NUL-terminated)
  int shape[2] = {2, 4};
  put(shape, sizeof(shape));                 // shape
  put_u64(0);                                // padding to data section
  std::vector<char> payload(64, '\x1a');
  put(payload.data(), payload.size());       // payload
  return out;
}

// Assemble a minimal two-vocabulary model buffer: ig8 `encoder_Wemb`,
// ig8 `decoder_Wemb` (both shape (2, 4), payload = 8 int8 + f32 mult), and
// the ig8 alpha marker `decoder_Wemb_QuantMultA` (shape (1, 1): one int8 +
// f32 mult), mirroring what Firefox ships for en-ja/en-zh.
constexpr uint64_t kIg8 = 0x4101;  // signed | intgemm marker | 1 byte

void put_ig8(std::vector<char>& out, std::initializer_list<int8_t> values,
             float mult) {
  size_t begin = out.size();
  for (int8_t v : values) out.push_back(static_cast<char>(v));
  const char* m = reinterpret_cast<const char*>(&mult);
  out.insert(out.end(), m, m + sizeof(mult));
  // Pad the payload to the advertised data_length of 64 (alignment slop the
  // loader must skip past, never read).
  out.insert(out.end(), 64 - (out.size() - begin), '\0');
}

std::vector<char> two_vocab_model() {
  std::vector<char> out;
  auto put = [&](const void* p, size_t n) {
    const char* c = static_cast<const char*>(p);
    out.insert(out.end(), c, c + n);
  };
  auto put_u64 = [&](uint64_t v) { put(&v, sizeof(v)); };

  put_u64(leanmt::kBinaryFileVersion);  // version
  put_u64(3);                           // num_headers
  Header enc{/*name_length=*/13, /*type=*/kIg8, /*shape_length=*/2,
             /*data_length=*/64};
  Header dec{/*name_length=*/13, /*type=*/kIg8, /*shape_length=*/2,
             /*data_length=*/64};
  Header alpha{/*name_length=*/24, /*type=*/kIg8, /*shape_length=*/2,
               /*data_length=*/64};
  for (const Header* h : {&enc, &dec, &alpha}) put(h, sizeof(*h));
  put("encoder_Wemb\0", 13);  // names (NUL-terminated)
  put("decoder_Wemb\0", 13);
  put("decoder_Wemb_QuantMultA\0", 24);
  int shape12[2] = {2, 4};
  int shape11[2] = {1, 1};
  put(shape12, sizeof(shape12));  // shapes
  put(shape12, sizeof(shape12));
  put(shape11, sizeof(shape11));
  put_u64(0);  // padding to data section
  put_ig8(out, {1, 2, 3, 4, 5, 6, 7, 8}, 2.0f);       // encoder_Wemb
  put_ig8(out, {9, 10, 11, 12, 13, 14, 15, 16}, 4.0f);  // decoder_Wemb
  put_ig8(out, {127}, 19.198595f);  // decoder_Wemb_QuantMultA (alpha ≈ 6.6)
  return out;
}

}  // namespace

TEST("load_items parses a minimal well-formed buffer") {
  auto buffer = minimal_model();
  auto items = leanmt::io::load_items(buffer.data(), buffer.size());
  // The two trailing entries are the (empty-defaulted) prepared-embedding
  // aliases appended unconditionally by the loader.
  CHECK(items.size() >= 3);
  CHECK_EQ(items[0].name, "Wemb");
  CHECK_EQ(items[0].shape.dim(-1), 4);
  CHECK(items[0].type == leanmt::Type::f32);
}

TEST("load_items handles a two-vocabulary model (encoder/decoder_Wemb)") {
  auto buffer = two_vocab_model();
  auto items = leanmt::io::load_items(buffer.data(), buffer.size());

  auto by_name = [&](const std::string& name) -> const leanmt::io::Item* {
    for (const auto& item : items) {
      if (item.name == name) return &item;
    }
    return nullptr;
  };

  // Encoder embedding: no GEMM alias. The embedding's own storage form
  // (int8 view vs unquantized f32 copy) is an internal detail — pin the
  // name/shape contract here and values end-to-end via the frontend
  // goldens instead.
  const auto* enc = by_name("encoder_Wemb");
  REQUIRE(enc != nullptr);
  CHECK_EQ(enc->shape.dim(-2), 2);
  CHECK_EQ(enc->shape.dim(-1), 4);
  CHECK(!by_name("encoder_Wemb_intgemm8"));

  // Decoder embedding: an int8 alias named `decoder_Wemb_intgemm8` with
  // transposed shape (cols, rows) is what the decoder output projection
  // GEMMs against.
  const auto* dec = by_name("decoder_Wemb");
  REQUIRE(dec != nullptr);
  CHECK_EQ(dec->shape.dim(-2), 2);
  CHECK_EQ(dec->shape.dim(-1), 4);
  const auto* dec_alias = by_name("decoder_Wemb_intgemm8");
  REQUIRE(dec_alias != nullptr);
  CHECK(dec_alias->type == leanmt::Type::i8);
  CHECK_EQ(dec_alias->shape.dim(-2), 4);  // transposed (cols, rows)
  CHECK_EQ(dec_alias->shape.dim(-1), 2);

  // Two-vocab models ship `decoder_Wemb_QuantMultA` as an ig8 [1,1] marker
  // (int8=127 + f32 quantMult) instead of the shared-vocab variant's f32
  // `none_QuantMultA`. It must materialize as the real f32 alpha — this is
  // the only calibrated source the decoder output projection gets on these
  // models. (Verified against Firefox's en-ja v2.3 file: 127/19.1986 ≈ 6.6.)
  const auto* alpha = by_name("decoder_Wemb_QuantMultA");
  REQUIRE(alpha != nullptr);
  CHECK(alpha->type == leanmt::Type::f32);
  REQUIRE(alpha->view.data != nullptr);
  CHECK_CLOSE(static_cast<const float*>(alpha->view.data)[0], 127.0f / 19.198595f, 1e-4f);
}

TEST("load_items rejects a buffer smaller than the version field") {
  std::vector<char> buffer(4, '\0');
  CHECK_THROWS(leanmt::io::load_items(buffer.data(), buffer.size()),
               "corrupt or truncated");
}

TEST("load_items rejects a wrong binary file version") {
  auto buffer = minimal_model();
  uint64_t bad_version = 42;
  std::memcpy(buffer.data(), &bad_version, sizeof(bad_version));
  CHECK_THROWS(leanmt::io::load_items(buffer.data(), buffer.size()),
               "binary version mismatch");
}

TEST("load_items rejects a header overhang (declared count beyond end)") {
  auto buffer = minimal_model();
  uint64_t count = 6;  // only one header present
  std::memcpy(buffer.data() + 8, &count, sizeof(count));
  CHECK_THROWS(leanmt::io::load_items(buffer.data(), buffer.size()),
               "exceeds the file's remaining");
}

TEST("load_items rejects a non-NUL-terminated tensor name") {
  auto buffer = minimal_model();
  // Name sits at offset 16 + 32 = 48; clobber its terminator.
  buffer[48 + 4] = 'x';
  CHECK_THROWS(leanmt::io::load_items(buffer.data(), buffer.size()),
               "NUL-terminated");
}

TEST("load_items rejects non-positive shape dimensions") {
  auto buffer = minimal_model();
  // Shape ints sit at 16 + 32 + 5 = 53.
  int zero = 0;
  std::memcpy(buffer.data() + 53, &zero, sizeof(zero));
  CHECK_THROWS(leanmt::io::load_items(buffer.data(), buffer.size()),
               "non-positive shape dimension");
}

TEST("load_items rejects an unknown tensor type") {
  auto buffer = minimal_model();
  Header mutated{/*name_length=*/5, /*type=*/0xDEAD, /*shape_length=*/2,
                 /*data_length=*/64};
  std::memcpy(buffer.data() + 16, &mutated, sizeof(mutated));
  CHECK_THROWS(leanmt::io::load_items(buffer.data(), buffer.size()),
               "incompatible tensor type");
}

TEST("load_items rejects a payload running past the end of file") {
  auto buffer = minimal_model();
  Header mutated{/*name_length=*/5, /*type=*/0x0404, /*shape_length=*/2,
                 /*data_length=*/1 << 20};
  std::memcpy(buffer.data() + 16, &mutated, sizeof(mutated));
  CHECK_THROWS(leanmt::io::load_items(buffer.data(), buffer.size()),
               "exceeds the file's remaining");
}

TEST("MmapFile maps a real file and reports its size") {
  const char* path = "leanmt_io_test_mmap_tmp.bin";
  {
    std::ofstream out(path, std::ios::binary);
    out << "0123456789abcdef";
  }
  leanmt::io::MmapFile file(path);
  CHECK_EQ(file.size(), 16u);
  CHECK(std::memcmp(file.data(), "0123", 4) == 0);
  std::remove(path);
}

TEST("MmapFile throws on a missing path") {
  CHECK_THROWS(leanmt::io::MmapFile("/definitely/missing/model.bin"),
               "Failed to open file");
}

int main() { return leanmt::test::run_all(); }
