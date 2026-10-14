#include "Io.hh"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Aligned.hh"
#include "QMM.hh"
#include "Tensor.hh"
#include "Types.hh"

namespace leanmt::io {

namespace {

// Bounds-checked cursor over the model file buffer. Model files arrive from
// `mmap` of an on-device download/packaged asset; a length field past the end
// of the mapping must surface as a clear error (fail-fast), never a SIGSEGV
// from walking off the mapped region.
class Cursor {
 public:
  Cursor(void* data, size_t size)
      : head_(static_cast<char*>(data)),
        begin_(static_cast<char*>(data)),
        end_(head_ + size) {}

  template <typename Element>
  Element read(const char* what) {
    require(sizeof(Element), what);
    Element value;
    std::memcpy(&value, head_, sizeof(Element));
    head_ += sizeof(Element);
    return value;
  }

  template <typename Element>
  std::vector<Element> read_vector(uint64_t count, const char* what) {
    if (count > remaining() / sizeof(Element)) {
      fail(what, count * sizeof(Element));
    }
    std::vector<Element> values(count);
    if (count > 0) {
      std::memcpy(values.data(), head_, count * sizeof(Element));
    }
    head_ += count * sizeof(Element);
    return values;
  }

  // Returns the current position and advances `size` bytes (caller keeps a
  // raw view into the buffer; the view is validated to be in-bounds).
  char* read_bytes(uint64_t size, const char* what) {
    require(size, what);
    char* begin = head_;
    head_ += size;
    return begin;
  }

  size_t offset() const { return static_cast<size_t>(head_ - begin_); }
  size_t remaining() const { return static_cast<size_t>(end_ - head_); }

 private:
  void require(uint64_t bytes, const char* what) const {
    if (bytes > remaining()) {
      fail(what, bytes);
    }
  }

  [[noreturn]] void fail(const char* what, uint64_t bytes) const {
    throw std::runtime_error(
        "[leanmt] corrupt or truncated model file: reading " + std::string(what) +
        " (" + std::to_string(bytes) + " bytes) at offset " +
        std::to_string(offset()) + " exceeds the file's remaining " +
        std::to_string(remaining()) + " bytes");
  }

  char* head_;
  char* begin_;
  char* end_;
};

// clang-format off
// NOLINTBEGIN
// Internal to types.h, don't use. Use test functions below.
enum class TypeClass : size_t {
  signed_type   = 0x0100,
  unsigned_type = 0x0200,
  float_type    = 0x0400,

  intgemm_type = 0x4000, // legacy marian int8-quantized model format marker


  size_mask     = 0x00FF,
  class_mask    = 0xFF00
};

constexpr inline size_t operator+(TypeClass typeClass, size_t val) {
  return (size_t)typeClass + val;
}

constexpr inline size_t operator+(size_t val, TypeClass typeClass) {
  return val + (size_t)typeClass;
}



enum class OGType : size_t {
  int8     = TypeClass::signed_type + 1u,
  float32  = TypeClass::float_type + 4u,

  // Marian's legacy int8-quantized model format marker (not the intgemm
  // backend, which was removed). Maps to leanmt's `Type::ig8`.
  intgemm8      = TypeClass::signed_type + 1u + TypeClass::intgemm_type
};

// clang-format on
// NOLINTEND

Type intercept(uint64_t value) {
  auto type = static_cast<OGType>(value);
  switch (type) {
    case OGType::intgemm8:
      return Type::ig8;
    case OGType::int8:
      return Type::i8;
    case OGType::float32:
      return Type::f32;
    default:
      throw std::runtime_error(
          "[leanmt] model file contains an incompatible tensor type (" +
          std::to_string(value) + ")");
  }
}

}  // namespace

void set_item(Item& item, Aligned&& aligned) {
  item.aligned = std::move(aligned);
  item.view = View{
      .data = reinterpret_cast<char*>(item.aligned.data()),  //
      .size = item.aligned.size()                            //
  };
}

std::vector<io::Item> load_items(void* data, size_t file_size) {
  Cursor cursor(data, file_size);

  uint64_t binary_file_version = cursor.read<uint64_t>("binary file version");
  if (binary_file_version != kBinaryFileVersion) {
    throw std::runtime_error(
        "[leanmt] model file binary version mismatch: file is v" +
        std::to_string(binary_file_version) + ", leanmt expects v" +
        std::to_string(kBinaryFileVersion) +
        " (re-download or regenerate the model)");
  }

  // Read number of headers and based on the information, the headers.
  uint64_t num_headers = cursor.read<uint64_t>("tensor header count");
  std::vector<Header> headers =
      cursor.read_vector<Header>(num_headers, "tensor headers");

  // prepopulate items with meta data from headers
  std::vector<io::Item> items;
  items.resize(num_headers);
  for (uint64_t i = 0; i < num_headers; ++i) {
    items[i].type = intercept(headers[i].type);

    // Names are written NUL-terminated; -1 strips the terminator.
    size_t length = headers[i].name_length;
    char* name = cursor.read_bytes(length, "tensor name");
    if (length < 1 || name[length - 1] != '\0') {
      throw std::runtime_error(
          "[leanmt] corrupt model file: tensor name at header index " +
          std::to_string(i) + " is empty or not NUL-terminated");
    }
    items[i].name = std::string(name, length - 1);
  }

  // read in actual shape and data
  for (uint64_t i = 0; i < num_headers; ++i) {
    Item& item = items[i];

    uint64_t size = headers[i].shape_length;
    std::vector<int> shape = cursor.read_vector<int>(size, "tensor shape");
    for (int dim : shape) {
      if (dim <= 0) {
        throw std::runtime_error(
            "[leanmt] corrupt model file: tensor '" + item.name +
            "' has non-positive shape dimension " + std::to_string(dim));
      }
    }

    // This copy has to be incurred, because metadata.
    item.shape.set(shape.data(), shape.data() + shape.size());
  }

  // move by offset bytes, aligned to 256-bytes boundary
  uint64_t offset = cursor.read<uint64_t>("data section offset");
  cursor.read_bytes(offset, "alignment padding to data section");

  // Keep extra items for the int8-prepared (PrepareB-form) versions of any
  // embeddings that double as the decoder output projection. For shared-vocab
  // models that's a single `Wemb_intgemm8`. For two-vocab models with
  // separate `decoder_Wemb`, that's `decoder_Wemb_intgemm8`. Encoder-only
  // embeddings (`encoder_Wemb`) don't need a PrepareB form since they're
  // never used as a GEMM B-matrix. (The `_intgemm8` suffix is the marian
  // model-file name for the legacy int8 format; the runtime multiplies these
  // weights with ruy.)
  Item embedding_processed;
  Item decoder_embedding_processed;

  // Recognises the three embedding tensor names leanmt knows about. Anything
  // else with name ending in "_QuantMultA" stays a quantization-multiplier
  // marker (no-op view); other ig8 tensors are GEMM B-matrices that need
  // PrepareBQuantizedTransposed.
  auto is_embedding_name = [](const std::string& name) {
    return name == "Wemb" || name == "encoder_Wemb" || name == "decoder_Wemb";
  };
  auto is_quant_mult_a_marker = [](const std::string& name) {
    return name == "Wemb_QuantMultA" ||
           name == "encoder_Wemb_QuantMultA" ||
           name == "decoder_Wemb_QuantMultA";
  };

  for (uint64_t i = 0; i < num_headers; ++i) {
    Item& item = items[i];
    uint64_t size = headers[i].data_length;
    char* ptr = cursor.read_bytes(size, "tensor payload");
    // We're about to read-data.
    // We can either make it point to mmap, which is aligned,
    // or we can create a new aligned.
    if (item.type == Type::ig8) {
      // ig8 payloads are `shape.elements()` int8 values followed by an f32
      // quantization multiplier (and possibly 256-byte alignment padding,
      // which `size` covers). Validate the logical payload fits in the
      // advertised data_length before touching it.
      size_t num_elements = item.shape.elements();
      if (num_elements + sizeof(float) > static_cast<size_t>(size)) {
        throw std::runtime_error(
            "[leanmt] corrupt model file: data_length of ig8 tensor '" +
            item.name + "' (" + std::to_string(size) +
            " bytes) cannot hold " + std::to_string(num_elements) +
            " int8 values plus the f32 quantization multiplier");
      }
      // since Embedding layer quantized weights need to be dequantised, we
      // have a special case for the embedding tensors. Two-vocab models
      // (en-zh, en-ja, ...) have separate `encoder_Wemb` / `decoder_Wemb`;
      // single-vocab models have a single `Wemb`. The handling is the same:
      // the tensor keeps its int8 storage (embedding gathers dequantize rows
      // on the fly with the trailing multiplier), and (for the embedding that
      // doubles as the decoder output projection) a second item aliases the
      // same bytes with the transposed shape for the GEMM. No f32 copy is
      // materialized — that used to hold ~32 MiB of permanent heap for the
      // 32k-vocab tiny11 models, plus another ~8 MiB for the alias copy.
      if (is_quant_mult_a_marker(item.name)) {
        // `*_Wemb_QuantMultA` tensors aren't empty placeholders despite
        // their shape-[1,1] size — marian stores them in the ig8 layout
        // (`shape.elements()` int8 values followed by the f32 `quantMult`)
        // and `marian-dev/.../integer_common.h::unquantizeWemb` reads them
        // back as `alpha[i] = int8[i] / quantMult`. For these alpha
        // tensors that's one int8 (=127, the saturated post-quantize value)
        // plus the multiplier, which un-quantizes to the original
        // calibrated activation alpha (≈4–8 for tiny11 bergamot models).
        //
        // We only need to materialize the value for `decoder_Wemb_QuantMultA`
        // — on shared-vocab models the same alpha is also shipped as a
        // real f32 `none_QuantMultA` that the parameter map already
        // consumes, and `encoder_Wemb` is an embedding-lookup-only tensor
        // with no output-projection GEMM. Drop those by clearing the name;
        // for `decoder_Wemb_QuantMultA` (the only source on two-vocab
        // models like en-ja / zh_hant-en) un-quantize and store as f32.
        char* mult_addr = ptr + num_elements;
        float quant_mult;
        std::memcpy(&quant_mult, mult_addr, sizeof(float));

        Aligned aligned(kAlignWidth, sizeof(float));
        auto* out = reinterpret_cast<float*>(aligned.data());
        out[0] =
            static_cast<float>(reinterpret_cast<int8_t*>(ptr)[0]) / quant_mult;
        set_item(item, std::move(aligned));
        item.type = Type::f32;
      } else if (is_embedding_name(item.name)) {  // NOLINT
        // Keep the embedding table in its int8 payload exactly as mmapped:
        // `num_elements` int8 values row-major (vocabulary × embedding
        // dimension) followed by the f32 quantization multiplier. Gathers
        // dequantize rows on the fly with the same `q[i] * (1/mult)` f32
        // formula the old load-time pass used (bit-exact), so model outputs
        // are unchanged. Viewing instead of copying removes the ~32 MiB f32
        // materialization from peak RSS and the quantize/dequantize pass
        // from model load.
        item.view = View{.data = ptr, .size = static_cast<size_t>(size)};
        item.type = Type::i8;

        size_t rows = item.shape.dim(-2);
        size_t cols = item.shape.dim(-1);

        // Decide which name the GEMM-shape alias should take. Skip the alias
        // entirely for `encoder_Wemb` — encoder embeddings are never used as
        // a GEMM B-matrix. For `decoder_Wemb` and the legacy single `Wemb`,
        // the decoder output projection GEMMs against this table. The alias
        // is a view of the same bytes with the dims swapped: the file's
        // row-major (vocab × dim) int8 payload read as a col-major
        // (dim × vocab) matrix IS the transpose — no data movement, the old
        // requantize-transposed copy recovered these same values bit-exactly.
        const bool needs_prepared_alias = (item.name != "encoder_Wemb");
        if (needs_prepared_alias) {
          Item& processed = (item.name == "decoder_Wemb")
                                ? decoder_embedding_processed
                                : embedding_processed;
          processed.name = (item.name == "decoder_Wemb")
                               ? "decoder_Wemb_intgemm8"
                               : "Wemb_intgemm8";
          processed.shape = Shape({cols, rows});
          processed.type = Type::i8;
          processed.view = item.view;
        }
      } else {
        // The matrix has to be processed to the format expected by the QMM backend.
        size_t rows = item.shape.dim(-2);
        size_t cols = item.shape.dim(-1);
        auto* input = reinterpret_cast<int8_t*>(ptr);

        Aligned aligned(kAlignWidth, rows * cols + sizeof(float));

        auto* output = reinterpret_cast<int8_t*>(aligned.data());
        qmm::prepare_weight_quantized_transposed(input, output, rows, cols);

        // Set b_quant at end.
        auto* output_end = reinterpret_cast<float*>(output + rows * cols);
        auto* input_end = reinterpret_cast<float*>(input + rows * cols);
        *output_end = *input_end;

        set_item(item, std::move(aligned));
        item.type = Type::i8;
      }
    } else {
      item.view = View{
          .data = ptr,                       //
          .size = static_cast<size_t>(size)  //
      };
    }
  }

  // Append both prepared aliases. Items left default-constructed (e.g. when
  // the model has no `decoder_Wemb`) appear as no-name entries that the
  // Transformer's parameter map never looks up — harmless.
  items.push_back(std::move(embedding_processed));
  items.push_back(std::move(decoder_embedding_processed));
  return items;
}

void unquantize_embedding_weights(const int8_t* quantized_weights,
                                  float quantization_multiplier, size_t size,
                                  float* weights) {
  // Canonical int8→f32 embedding dequantization, applied per gathered row at
  // lookup time (the loader keeps embedding tables in their int8 storage).
  // Gather sites call this same helper so results stay bit-exact with the
  // load-time materialization the loader used to do — same expression, same
  // evaluation order: reciprocal as a float division, then a per-element
  // float multiply.
  for (size_t i = 0; i < size; i++) {
    weights[i] = static_cast<float>(quantized_weights[i]) *
                 (1 / quantization_multiplier);
  }
}

std::ostream& operator<<(std::ostream& out, const Item& item) {
  out << "Item(" << item.name << ", ";
  out << to_string(item.type) << ", ";
  out << item.shape << ")";
  return out;
}

MmapFile::MmapFile(const std::string& filepath) {
  fd_ = open(filepath.c_str(), O_RDONLY);
  if (fd_ == -1) {
    throw std::runtime_error("Failed to open file: " + filepath);
  }

  struct stat st;
  if (fstat(fd_, &st) == -1) {
    close(fd_);
    throw std::runtime_error("Failed to get file size: " + filepath);
  }
  size_ = st.st_size;

  data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (data_ == MAP_FAILED) {  // NOLINT
    close(fd_);
    throw std::runtime_error("Failed to mmap file: " + filepath);
  }
}

MmapFile::~MmapFile() {
  if (data_ != nullptr) {
    munmap(data_, size_);
  }
  if (fd_ != -1) {
    close(fd_);
  }
}

MmapFile::MmapFile(MmapFile&& from) noexcept
    : fd_(from.fd_), data_(from.data_), size_(from.size_) {
  from.reset();
}

MmapFile& MmapFile::operator=(MmapFile&& from) noexcept {
  if (this == &from) {
    return *this;
  }
  consume(from);
  return *this;
}

void MmapFile::consume(MmapFile& from) {
  fd_ = (from.fd_);
  data_ = (from.data_);
  size_ = (from.size_);
  from.reset();
}

void MmapFile::reset() {
  fd_ = -1;
  data_ = nullptr;
  size_ = 0;
}

}  // namespace leanmt::io
