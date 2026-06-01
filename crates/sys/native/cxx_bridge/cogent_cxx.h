#pragma once

#include <cstdint>
#include <memory>

#include "rust/cxx.h"

namespace cogentlm::sys {

class NativeBatch;
class NativeRuntime;
class CommonSampler;

void backend_init();
void backend_load_all();
void set_llama_log_quiet(bool quiet);
rust::String backend_observability_json(bool include_details);
rust::String mtmd_default_marker();

std::unique_ptr<NativeRuntime> load_native_runtime(rust::Str model_path, rust::Str args_json);

class NativeRuntime {
public:
  ~NativeRuntime();

  NativeRuntime(const NativeRuntime &) = delete;
  NativeRuntime & operator=(const NativeRuntime &) = delete;
  NativeRuntime(NativeRuntime &&) noexcept;
  NativeRuntime & operator=(NativeRuntime &&) noexcept;

  std::int32_t n_ctx() const;
  std::int32_t n_batch() const;
  std::int32_t n_ubatch() const;
  std::int32_t n_seq_max() const;
  std::int32_t n_threads() const;
  std::int32_t n_threads_batch() const;
  std::int32_t n_embd_out() const;
  std::int32_t n_cls_out() const;
  std::int32_t pooling_type() const;
  bool has_encoder() const;
  bool has_decoder() const;
  bool has_chat_template() const;
  bool is_recurrent() const;
  bool is_hybrid() const;
  bool kv_unified() const;
  rust::String flash_attention() const;
  rust::String cache_type_k() const;
  rust::String cache_type_v() const;
  rust::String chat_template_source() const;
  std::int32_t bos_token() const;
  std::int32_t eos_token() const;
  std::int32_t decoder_start_token() const;
  bool is_eog(std::int32_t token) const;
  bool mtmd_ready() const;
  rust::Vec<std::int32_t> tokenize(rust::Str text, bool add_special, bool parse_special) const;
  rust::String token_to_piece(std::int32_t token, bool special) const;
  rust::Vec<std::uint8_t> token_to_piece_bytes(std::int32_t token, bool special) const;
  rust::String apply_chat_template_json(rust::Str messages_json, bool add_assistant) const;
  std::int32_t decode(const NativeBatch & batch);
  std::int32_t encode(const NativeBatch & batch);
  bool synchronize();
  bool clear_sequence(std::int32_t seq_id, std::int32_t p0, std::int32_t p1);
  void add_sequence_delta(std::int32_t seq_id, std::int32_t p0, std::int32_t p1, std::int32_t delta);
  rust::Vec<float> embeddings_seq(std::int32_t seq_id) const;
  rust::Vec<std::uint8_t> state_seq(std::int32_t seq_id) const;
  bool set_state_seq(std::int32_t seq_id, rust::Slice<const std::uint8_t> data);
  bool init_mtmd(rust::Str projector_path, bool use_gpu, std::int32_t n_threads);
  bool mtmd_support_vision() const;
  std::int32_t mtmd_eval_images(
      rust::Str prompt,
      rust::Slice<const std::uint8_t> image_bytes,
      rust::Slice<const std::int32_t> image_sizes,
      bool add_special,
      bool parse_special,
      std::int32_t n_past,
      std::int32_t seq_id,
      std::int32_t n_batch,
      bool logits_last);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit NativeRuntime(std::unique_ptr<Impl> impl);

  friend std::unique_ptr<NativeRuntime> load_native_runtime(rust::Str model_path, rust::Str args_json);
  friend std::unique_ptr<CommonSampler> create_sampler(
      const NativeRuntime & runtime,
      rust::Str sampling_json,
      rust::Str grammar,
      rust::Str json_schema);
  friend std::int32_t sampler_sample(CommonSampler & sampler, NativeRuntime & runtime, std::int32_t idx);
  friend bool sampler_attach(CommonSampler & sampler, NativeRuntime & runtime, std::int32_t seq_id);
  friend bool sampler_detach(NativeRuntime & runtime, std::int32_t seq_id);
};

std::unique_ptr<NativeBatch> make_native_batch();

class NativeBatch {
public:
  NativeBatch();
  ~NativeBatch();

  NativeBatch(const NativeBatch &) = delete;
  NativeBatch & operator=(const NativeBatch &) = delete;
  NativeBatch(NativeBatch &&) noexcept;
  NativeBatch & operator=(NativeBatch &&) noexcept;

  void ensure_capacity(std::int32_t max_tokens, std::int32_t max_sequences);
  void reset();
  bool add_token(std::int32_t token, std::int32_t pos, std::int32_t seq_id, bool logits);
  std::int32_t n_tokens() const;
  std::int32_t token(std::int32_t index) const;
  std::int32_t pos(std::int32_t index) const;
  std::int32_t seq_id(std::int32_t index) const;
  bool logits(std::int32_t index) const;
  void clear_logits();
  void set_last_logits();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend class NativeRuntime;
};

std::unique_ptr<CommonSampler> create_sampler(
    const NativeRuntime & runtime,
    rust::Str sampling_json,
    rust::Str grammar,
    rust::Str json_schema);

class CommonSampler {
public:
  ~CommonSampler();

  CommonSampler(const CommonSampler &) = delete;
  CommonSampler & operator=(const CommonSampler &) = delete;
  CommonSampler(CommonSampler &&) noexcept;
  CommonSampler & operator=(CommonSampler &&) noexcept;

  bool sampler_accept(std::int32_t token, bool accept_grammar);
  void sampler_reset();
  bool sampler_backend_sampling() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit CommonSampler(std::unique_ptr<Impl> impl);

  friend std::unique_ptr<CommonSampler> create_sampler(
      const NativeRuntime & runtime,
      rust::Str sampling_json,
      rust::Str grammar,
      rust::Str json_schema);
  friend std::int32_t sampler_sample(CommonSampler & sampler, NativeRuntime & runtime, std::int32_t idx);
  friend bool sampler_attach(CommonSampler & sampler, NativeRuntime & runtime, std::int32_t seq_id);
};

std::int32_t sampler_sample(CommonSampler & sampler, NativeRuntime & runtime, std::int32_t idx);
bool sampler_attach(CommonSampler & sampler, NativeRuntime & runtime, std::int32_t seq_id);
bool sampler_detach(NativeRuntime & runtime, std::int32_t seq_id);

} // namespace cogentlm::sys
