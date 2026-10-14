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
