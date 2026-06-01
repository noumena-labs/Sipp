#pragma once

#include <cstdint>
#include <memory>

#include "rust/cxx.h"

namespace cogentlm::wasm {

using GgufReadAtCallback =
    int (*)(void *, std::uint64_t, std::uint8_t *, std::uintptr_t);
using GgufOpenShardCallback =
    int (*)(void *, const char *, std::uint16_t, std::uint16_t);
using GgufWriteShardCallback =
    int (*)(void *, const std::uint8_t *, std::uintptr_t);
using GgufCloseShardCallback = int (*)(void *);

class GgufReadAt {
public:
  GgufReadAt(void * user_data, GgufReadAtCallback read_at);

  int read_at(std::uint64_t offset, rust::Slice<std::uint8_t> dst);

private:
  void * user_data_;
  GgufReadAtCallback read_at_;
};

class GgufShardWriter {
public:
  GgufShardWriter(
      void * user_data,
      GgufWriteShardCallback write_shard,
      GgufCloseShardCallback close_shard);

  int write_shard(rust::Slice<const std::uint8_t> bytes);
  int close_shard();

private:
  void * user_data_;
  GgufWriteShardCallback write_shard_;
  GgufCloseShardCallback close_shard_;
};

class GgufShardSink {
public:
  GgufShardSink(
      void * user_data,
      GgufOpenShardCallback open_shard,
      GgufWriteShardCallback write_shard,
      GgufCloseShardCallback close_shard);

  int open_shard(rust::Str path, std::uint16_t index, std::uint16_t count);
  std::unique_ptr<GgufShardWriter> create_writer();

private:
  void * user_data_;
  GgufOpenShardCallback open_shard_;
  GgufWriteShardCallback write_shard_;
  GgufCloseShardCallback close_shard_;
};

} // namespace cogentlm::wasm
