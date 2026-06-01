#include "gguf_callbacks.h"

#include <string>

namespace {

constexpr int kStatusInvalidArguments = -2;

} // namespace

namespace cogentlm::wasm {

GgufReadAt::GgufReadAt(void * user_data, GgufReadAtCallback read_at)
    : user_data_(user_data), read_at_(read_at) {}

int GgufReadAt::read_at(std::uint64_t offset, rust::Slice<std::uint8_t> dst) {
  return read_at_(user_data_, offset, dst.data(), dst.size());
}

GgufShardWriter::GgufShardWriter(
    void * user_data,
    GgufWriteShardCallback write_shard,
    GgufCloseShardCallback close_shard)
    : user_data_(user_data), write_shard_(write_shard), close_shard_(close_shard) {}

int GgufShardWriter::write_shard(rust::Slice<const std::uint8_t> bytes) {
  return write_shard_(user_data_, bytes.data(), bytes.size());
}

int GgufShardWriter::close_shard() {
  return close_shard_(user_data_);
}

GgufShardSink::GgufShardSink(
    void * user_data,
    GgufOpenShardCallback open_shard,
    GgufWriteShardCallback write_shard,
    GgufCloseShardCallback close_shard)
    : user_data_(user_data),
      open_shard_(open_shard),
      write_shard_(write_shard),
      close_shard_(close_shard) {}

int GgufShardSink::open_shard(rust::Str path, std::uint16_t index, std::uint16_t count) {
  const std::string copy(path.data(), path.size());
  if (copy.find('\0') != std::string::npos) {
    return kStatusInvalidArguments;
  }
  return open_shard_(user_data_, copy.c_str(), index, count);
}

std::unique_ptr<GgufShardWriter> GgufShardSink::create_writer() {
  return std::make_unique<GgufShardWriter>(user_data_, write_shard_, close_shard_);
}

} // namespace cogentlm::wasm
