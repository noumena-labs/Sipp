#include "sipp_cxx.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "sipp_shim.h"
#include "llama.h"

namespace sipp::sys {
namespace {

constexpr std::uint32_t kStateSeqFlagsNone = 0;

struct FreeDeleter {
  template <typename T>
  void operator()(T * value) const {
    std::free(value);
  }
};

using OwnedCString = std::unique_ptr<char, FreeDeleter>;
using OwnedBytes = std::unique_ptr<std::uint8_t, FreeDeleter>;

struct MtmdBitmapDeleter {
  void operator()(sipp_mtmd_bitmap * bitmap) const {
    sipp_mtmd_bitmap_free(bitmap);
  }
};

struct MtmdChunksDeleter {
  void operator()(sipp_mtmd_input_chunks * chunks) const {
    sipp_mtmd_input_chunks_free(chunks);
  }
};

using OwnedMtmdBitmap = std::unique_ptr<sipp_mtmd_bitmap, MtmdBitmapDeleter>;
using OwnedMtmdChunks = std::unique_ptr<sipp_mtmd_input_chunks, MtmdChunksDeleter>;

std::string to_string(rust::Str value) {
  return std::string(value.data(), value.size());
}

void reject_interior_nul(const std::string & value, const char * label) {
  if (value.find('\0') != std::string::npos) {
    throw std::runtime_error(std::string(label) + " contains an interior NUL byte");
  }
}

std::string to_c_string_argument(rust::Str value, const char * label) {
  std::string text = to_string(value);
  reject_interior_nul(text, label);
  return text;
}

rust::String to_rust_string(const std::string & value) {
  return rust::String(value);
}

rust::String take_owned_string(char * value, const char * fallback) {
  OwnedCString owned(value);
  if (value == nullptr) {
    return to_rust_string(fallback == nullptr ? "" : fallback);
  }
  std::string copy(owned.get());
  return to_rust_string(copy);
}

std::string take_error(char * value, const char * fallback) {
  OwnedCString owned(value);
  if (value == nullptr) {
    return fallback == nullptr ? "native FFI call failed" : fallback;
  }
  std::string copy(owned.get());
  if (copy.empty() && fallback != nullptr) {
    return fallback;
  }
  return copy;
}

std::vector<std::string> parse_args_json(rust::Str args_json) {
  const std::string text = to_string(args_json);
  if (text.empty()) {
    return {};
  }
  auto args = nlohmann::ordered_json::parse(text).get<std::vector<std::string>>();
  for (const auto & arg : args) {
    reject_interior_nul(arg, "llama runtime argument");
  }
  return args;
}

rust::Vec<std::uint8_t> copy_bytes(const std::uint8_t * data, std::size_t size) {
  rust::Vec<std::uint8_t> out;
  for (std::size_t i = 0; i < size; ++i) {
    out.push_back(data[i]);
  }
  return out;
}

rust::Vec<float> copy_floats(const float * data, std::size_t size) {
  rust::Vec<float> out;
  for (std::size_t i = 0; i < size; ++i) {
    out.push_back(data[i]);
  }
  return out;
}

rust::Vec<std::uint8_t> copy_string_bytes(const std::string & value) {
  rust::Vec<std::uint8_t> out;
  for (unsigned char byte : value) {
    out.push_back(static_cast<std::uint8_t>(byte));
  }
  return out;
}

std::string token_to_piece_string(
    const llama_vocab * vocab,
    std::int32_t token,
    bool special) {
  std::array<char, 32> stack_buffer{};
  std::int32_t written =
      llama_token_to_piece(vocab, token, stack_buffer.data(), stack_buffer.size(), 0, special);
  if (written >= 0) {
    return std::string(stack_buffer.data(), static_cast<std::size_t>(written));
  }
  if (written == INT32_MIN) {
    throw std::runtime_error("llama_token_to_piece overflowed");
  }

  std::vector<char> buffer(static_cast<std::size_t>(-written));
  written = llama_token_to_piece(
      vocab,
      token,
      buffer.data(),
      static_cast<std::int32_t>(buffer.size()),
      0,
      special);
  if (written < 0) {
    throw std::runtime_error("llama_token_to_piece failed");
  }
  return std::string(buffer.data(), static_cast<std::size_t>(written));
}

} // namespace

struct NativeRuntime::Impl {
  sipp_common_init * init = nullptr;
  sipp_chat_templates * chat_templates = nullptr;
  sipp_mtmd_context * mtmd = nullptr;

  ~Impl() {
    if (mtmd != nullptr) {
      sipp_mtmd_free(mtmd);
    }
    if (chat_templates != nullptr) {
      sipp_chat_templates_free(chat_templates);
    }
    if (init != nullptr) {
      sipp_common_init_free(init);
    }
  }

  llama_model * model() const {
    return sipp_common_init_model(init);
  }

  llama_context * context() const {
    return sipp_common_init_context(init);
  }

  const llama_vocab * vocab() const {
    return sipp_common_init_vocab(init);
  }
};

struct NativeBatch::Impl {
  llama_batch batch{};
  std::int32_t capacity_tokens = 0;
  std::int32_t capacity_sequences = 0;
  bool allocated = false;

  ~Impl() {
    release();
  }

  void release() {
    if (!allocated) {
      return;
    }
    llama_batch_free(batch);
    batch = {};
    capacity_tokens = 0;
    capacity_sequences = 0;
    allocated = false;
  }

  bool has_storage() const {
    return batch.token != nullptr && batch.pos != nullptr && batch.n_seq_id != nullptr &&
           batch.seq_id != nullptr && batch.logits != nullptr;
  }
};

struct CommonSampler::Impl {
  sipp_common_sampler * sampler = nullptr;

  ~Impl() {
    if (sampler != nullptr) {
      sipp_common_sampler_free(sampler);
    }
  }
};

void backend_init() {
  llama_backend_init();
}

void backend_load_all() {
  sipp_backend_load_all();
}

void set_llama_log_quiet(bool quiet) {
  sipp_set_llama_log_quiet(quiet);
}

rust::String backend_observability_json(bool include_details) {
  return take_owned_string(sipp_backend_observability_json(include_details), "{}");
}

rust::String mtmd_default_marker() {
  const char * marker = sipp_mtmd_default_marker();
  return to_rust_string(marker == nullptr ? "" : marker);
}

std::unique_ptr<NativeRuntime> load_native_runtime(rust::Str model_path, rust::Str args_json) {
  const std::string model = to_c_string_argument(model_path, "model path");
  const auto args = parse_args_json(args_json);
  if (args.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::runtime_error("too many llama runtime arguments");
  }
  std::vector<const char *> argv;
  argv.reserve(args.size());
  for (const auto & arg : args) {
    argv.push_back(arg.c_str());
  }

  char * error = nullptr;
  sipp_common_params * params = sipp_common_params_parse_server(
      model.c_str(), static_cast<std::int32_t>(argv.size()), argv.data(), &error);
  if (params == nullptr) {
    throw std::runtime_error(take_error(error, "failed to parse llama runtime parameters"));
  }

  sipp_common_init * init = sipp_common_init_from_params(params, &error);
  sipp_common_params_free(params);
  if (init == nullptr) {
    throw std::runtime_error(take_error(error, "failed to initialize llama runtime"));
  }

  auto impl = std::make_unique<NativeRuntime::Impl>();
  impl->init = init;
  impl->chat_templates = sipp_chat_templates_init(sipp_common_init_model(init), "");
  if (impl->chat_templates == nullptr) {
    throw std::runtime_error("failed to initialize chat templates");
  }

  return std::unique_ptr<NativeRuntime>(new NativeRuntime(std::move(impl)));
}

NativeRuntime::NativeRuntime(std::unique_ptr<NativeRuntime::Impl> impl) : impl_(std::move(impl)) {}

NativeRuntime::~NativeRuntime() = default;
NativeRuntime::NativeRuntime(NativeRuntime &&) noexcept = default;
NativeRuntime & NativeRuntime::operator=(NativeRuntime &&) noexcept = default;

std::int32_t NativeRuntime::n_ctx() const {
  return sipp_common_init_n_ctx(impl_->init);
}

std::int32_t NativeRuntime::n_batch() const {
  return sipp_common_init_n_batch(impl_->init);
}

std::int32_t NativeRuntime::n_ubatch() const {
  return sipp_common_init_n_ubatch(impl_->init);
}

std::int32_t NativeRuntime::n_seq_max() const {
  return sipp_common_init_n_parallel(impl_->init);
}

std::int32_t NativeRuntime::n_threads() const {
  return llama_n_threads(impl_->context());
}

std::int32_t NativeRuntime::n_threads_batch() const {
  return llama_n_threads_batch(impl_->context());
}

std::int32_t NativeRuntime::n_embd_out() const {
  return sipp_common_init_n_embd_out(impl_->init);
}

std::int32_t NativeRuntime::n_cls_out() const {
  return sipp_common_init_n_cls_out(impl_->init);
}

std::int32_t NativeRuntime::pooling_type() const {
  return sipp_common_init_pooling_type(impl_->init);
}

bool NativeRuntime::has_encoder() const {
  return sipp_common_init_model_has_encoder(impl_->init);
}

bool NativeRuntime::has_decoder() const {
  return sipp_common_init_model_has_decoder(impl_->init);
}

bool NativeRuntime::has_chat_template() const {
  return sipp_common_init_model_has_chat_template(impl_->init);
}

bool NativeRuntime::is_recurrent() const {
  return llama_model_is_recurrent(impl_->model());
}

bool NativeRuntime::is_hybrid() const {
  return llama_model_is_hybrid(impl_->model());
}

bool NativeRuntime::kv_unified() const {
  return sipp_common_init_kv_unified(impl_->init);
}

rust::String NativeRuntime::flash_attention() const {
  return take_owned_string(sipp_common_init_flash_attention(impl_->init), "unknown");
}

rust::String NativeRuntime::cache_type_k() const {
  return take_owned_string(sipp_common_init_cache_type_k(impl_->init), "unknown");
}

rust::String NativeRuntime::cache_type_v() const {
  return take_owned_string(sipp_common_init_cache_type_v(impl_->init), "unknown");
}

rust::String NativeRuntime::chat_template_source() const {
  return take_owned_string(sipp_chat_templates_source(impl_->chat_templates), "");
}

std::int32_t NativeRuntime::bos_token() const {
  return llama_vocab_bos(impl_->vocab());
}

std::int32_t NativeRuntime::eos_token() const {
  return llama_vocab_eos(impl_->vocab());
}

std::int32_t NativeRuntime::decoder_start_token() const {
  return sipp_common_init_decoder_start_token(impl_->init);
}

bool NativeRuntime::is_eog(std::int32_t token) const {
  return llama_vocab_is_eog(impl_->vocab(), token);
}

bool NativeRuntime::mtmd_ready() const {
  return impl_->mtmd != nullptr;
}

rust::Vec<std::int32_t> NativeRuntime::tokenize(
    rust::Str text,
    bool add_special,
    bool parse_special) const {
  const std::string input = to_string(text);
  if (input.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::runtime_error("tokenizer input is too large");
  }
  std::int32_t required = llama_tokenize(
      impl_->vocab(),
      input.data(),
      static_cast<std::int32_t>(input.size()),
      nullptr,
      0,
      add_special,
      parse_special);
  if (required == INT32_MIN) {
    throw std::runtime_error("llama_tokenize overflowed");
  }
  if (required < 0) {
    required = -required;
  }

  std::vector<llama_token> buffer(static_cast<std::size_t>(required));
  const std::int32_t written = llama_tokenize(
      impl_->vocab(),
      input.data(),
      static_cast<std::int32_t>(input.size()),
      buffer.data(),
      required,
      add_special,
      parse_special);
  if (written < 0) {
    throw std::runtime_error("llama_tokenize failed");
  }

  rust::Vec<std::int32_t> out;
  for (std::int32_t i = 0; i < written; ++i) {
    out.push_back(buffer[static_cast<std::size_t>(i)]);
  }
  return out;
}

rust::String NativeRuntime::token_to_piece(std::int32_t token, bool special) const {
  return to_rust_string(token_to_piece_string(impl_->vocab(), token, special));
}

rust::Vec<std::uint8_t> NativeRuntime::token_to_piece_bytes(
    std::int32_t token,
    bool special) const {
  return copy_string_bytes(token_to_piece_string(impl_->vocab(), token, special));
}

rust::String NativeRuntime::apply_chat_template_json(
    rust::Str messages_json,
    bool add_assistant) const {
  const std::string messages = to_c_string_argument(messages_json, "chat messages JSON");
  char * rendered = sipp_apply_chat_template(
      impl_->chat_templates,
      messages.c_str(),
      add_assistant);
  if (rendered == nullptr) {
    throw std::runtime_error("failed to apply chat template");
  }
  return take_owned_string(rendered, "");
}

std::int32_t NativeRuntime::decode(const NativeBatch & batch) {
  const std::int32_t status = sipp_llama_decode(impl_->context(), &batch.impl_->batch);
  if (status < 0) {
    throw std::runtime_error("llama decode failed");
  }
  return status;
}

std::int32_t NativeRuntime::encode(const NativeBatch & batch) {
  const std::int32_t status = sipp_llama_encode(impl_->context(), &batch.impl_->batch);
  if (status < 0) {
    throw std::runtime_error("llama encode failed");
  }
  return status;
}

bool NativeRuntime::synchronize() {
  return sipp_llama_synchronize(impl_->context());
}

bool NativeRuntime::clear_sequence(std::int32_t seq_id, std::int32_t p0, std::int32_t p1) {
  return llama_memory_seq_rm(llama_get_memory(impl_->context()), seq_id, p0, p1);
}

void NativeRuntime::add_sequence_delta(
    std::int32_t seq_id,
    std::int32_t p0,
    std::int32_t p1,
    std::int32_t delta) {
  llama_memory_seq_add(llama_get_memory(impl_->context()), seq_id, p0, p1, delta);
}

rust::Vec<float> NativeRuntime::embeddings_seq(std::int32_t seq_id) const {
  const float * values = sipp_llama_embeddings_seq(impl_->context(), seq_id);
  if (values == nullptr) {
    throw std::runtime_error("llama embeddings sequence output is unavailable");
  }

  std::int32_t dimensions = sipp_common_init_n_embd_out(impl_->init);
  if (sipp_common_init_pooling_type(impl_->init) == LLAMA_POOLING_TYPE_RANK) {
    dimensions = sipp_common_init_n_cls_out(impl_->init);
  }
  if (dimensions <= 0) {
    throw std::runtime_error("model reports zero embedding dimensions");
  }
  return copy_floats(values, static_cast<std::size_t>(dimensions));
}

rust::Vec<std::uint8_t> NativeRuntime::state_seq(std::int32_t seq_id) const {
  std::uint8_t * data = nullptr;
  std::size_t size = 0;
  const bool ok = sipp_llama_state_seq_get_data_ext_alloc(
      impl_->context(),
      seq_id,
      kStateSeqFlagsNone,
      &data,
      &size);
  if (!ok || data == nullptr) {
    throw std::runtime_error("failed to capture llama sequence state");
  }
  OwnedBytes owned_data(data);
  return copy_bytes(owned_data.get(), size);
}

bool NativeRuntime::set_state_seq(std::int32_t seq_id, rust::Slice<const std::uint8_t> data) {
  return sipp_llama_state_seq_set_data_ext(
      impl_->context(),
      seq_id,
      kStateSeqFlagsNone,
      data.data(),
      data.size());
}

bool NativeRuntime::init_mtmd(rust::Str projector_path, bool use_gpu, std::int32_t n_threads) {
  const std::string path = to_c_string_argument(projector_path, "multimodal projector path");
  if (path.empty()) {
    return false;
  }
  if (impl_->mtmd != nullptr) {
    sipp_mtmd_free(impl_->mtmd);
    impl_->mtmd = nullptr;
  }
  impl_->mtmd = sipp_mtmd_init_from_file(path.c_str(), impl_->model(), use_gpu, n_threads);
  return impl_->mtmd != nullptr;
}

bool NativeRuntime::mtmd_support_vision() const {
  return impl_->mtmd != nullptr && sipp_mtmd_support_vision(impl_->mtmd);
}

std::int32_t NativeRuntime::mtmd_eval_images(
    rust::Str prompt,
    rust::Slice<const std::uint8_t> image_bytes,
    rust::Slice<const std::int32_t> image_sizes,
    bool add_special,
    bool parse_special,
    std::int32_t n_past,
    std::int32_t seq_id,
    std::int32_t n_batch,
    bool logits_last) {
  if (impl_->mtmd == nullptr) {
    throw std::runtime_error("multimodal context is not initialized");
  }

  std::size_t expected_image_bytes = 0;
  for (std::size_t i = 0; i < image_sizes.size(); ++i) {
    if (image_sizes[i] <= 0) {
      throw std::runtime_error("multimodal image payload size must be positive");
    }
    const auto len = static_cast<std::size_t>(image_sizes[i]);
    if (expected_image_bytes > image_bytes.size() ||
        len > image_bytes.size() - expected_image_bytes) {
      throw std::runtime_error("multimodal image sizes exceed payload length");
    }
    expected_image_bytes += len;
  }
  if (expected_image_bytes != image_bytes.size()) {
    throw std::runtime_error("multimodal image sizes do not match payload length");
  }

  std::vector<OwnedMtmdBitmap> owned_bitmaps;
  std::vector<const sipp_mtmd_bitmap *> bitmap_refs;
  owned_bitmaps.reserve(image_sizes.size());
  bitmap_refs.reserve(image_sizes.size());

  std::size_t offset = 0;
  for (std::size_t i = 0; i < image_sizes.size(); ++i) {
    const auto len = static_cast<std::size_t>(image_sizes[i]);
    sipp_mtmd_bitmap * bitmap =
        sipp_mtmd_bitmap_init_from_buf(impl_->mtmd, image_bytes.data() + offset, len);
    if (bitmap == nullptr) {
      throw std::runtime_error("failed to decode multimodal image payload");
    }
    owned_bitmaps.emplace_back(bitmap);
    bitmap_refs.push_back(bitmap);
    offset += len;
  }

  OwnedMtmdChunks chunks(sipp_mtmd_input_chunks_init());
  if (chunks == nullptr) {
    throw std::runtime_error("failed to allocate multimodal input chunks");
  }

  const std::string text = to_c_string_argument(prompt, "multimodal prompt");
  const bool tokenized = sipp_mtmd_tokenize(
      impl_->mtmd,
      chunks.get(),
      text.c_str(),
      add_special,
      parse_special,
      bitmap_refs.data(),
      bitmap_refs.size());
  if (!tokenized) {
    throw std::runtime_error("failed to tokenize multimodal input");
  }

  std::int32_t new_n_past = n_past;
  const std::int32_t status = sipp_mtmd_eval_chunks(
      impl_->mtmd,
      impl_->context(),
      chunks.get(),
      n_past,
      seq_id,
      n_batch,
      logits_last,
      &new_n_past);
  if (status != 0) {
    throw std::runtime_error("failed to evaluate multimodal chunks");
  }
  if (!sipp_llama_synchronize(impl_->context())) {
    throw std::runtime_error("failed to synchronize after multimodal evaluation");
  }

  return new_n_past;
}

std::unique_ptr<NativeBatch> make_native_batch() {
  return std::make_unique<NativeBatch>();
}

NativeBatch::NativeBatch() : impl_(std::make_unique<NativeBatch::Impl>()) {}
NativeBatch::~NativeBatch() = default;
NativeBatch::NativeBatch(NativeBatch &&) noexcept = default;
NativeBatch & NativeBatch::operator=(NativeBatch &&) noexcept = default;

void NativeBatch::ensure_capacity(std::int32_t max_tokens, std::int32_t max_sequences) {
  if (max_tokens <= 0 || max_sequences <= 0) {
    throw std::runtime_error("llama batch capacity must be positive");
  }

  if (impl_->allocated && impl_->capacity_tokens >= max_tokens &&
      impl_->capacity_sequences >= max_sequences) {
    reset();
    return;
  }

  impl_->release();
  impl_->batch = llama_batch_init(max_tokens, 0, max_sequences);
  if (!impl_->has_storage()) {
    llama_batch_free(impl_->batch);
    impl_->batch = {};
    throw std::runtime_error("llama_batch_init failed");
  }

  impl_->capacity_tokens = max_tokens;
  impl_->capacity_sequences = max_sequences;
  impl_->allocated = true;
  reset();
}

void NativeBatch::reset() {
  if (!impl_->allocated || !impl_->has_storage()) {
    return;
  }
  impl_->batch.n_tokens = 0;
}

bool NativeBatch::add_token(
    std::int32_t token,
    std::int32_t pos,
    std::int32_t seq_id,
    bool logits) {
  if (!impl_->allocated || !impl_->has_storage() || impl_->batch.n_tokens < 0 ||
      impl_->batch.n_tokens >= impl_->capacity_tokens || impl_->capacity_sequences <= 0) {
    return false;
  }

  const std::int32_t index = impl_->batch.n_tokens;
  llama_seq_id * seq_ids = impl_->batch.seq_id[index];
  if (seq_ids == nullptr) {
    return false;
  }
  impl_->batch.token[index] = token;
  impl_->batch.pos[index] = pos;
  impl_->batch.n_seq_id[index] = 1;
  seq_ids[0] = seq_id;
  impl_->batch.logits[index] = logits ? 1 : 0;
  impl_->batch.n_tokens += 1;
  return true;
}

std::int32_t NativeBatch::n_tokens() const {
  return impl_->batch.n_tokens;
}

std::int32_t NativeBatch::token(std::int32_t index) const {
  if (!impl_->allocated || !impl_->has_storage() || index < 0 || index >= impl_->batch.n_tokens) {
    return 0;
  }
  return impl_->batch.token[index];
}

std::int32_t NativeBatch::pos(std::int32_t index) const {
  if (!impl_->allocated || !impl_->has_storage() || index < 0 || index >= impl_->batch.n_tokens) {
    return 0;
  }
  return impl_->batch.pos[index];
}

std::int32_t NativeBatch::seq_id(std::int32_t index) const {
  if (!impl_->allocated || !impl_->has_storage() || index < 0 || index >= impl_->batch.n_tokens ||
      impl_->batch.seq_id[index] == nullptr) {
    return 0;
  }
  return impl_->batch.seq_id[index][0];
}

bool NativeBatch::logits(std::int32_t index) const {
  if (!impl_->allocated || !impl_->has_storage() || index < 0 || index >= impl_->batch.n_tokens) {
    return false;
  }
  return impl_->batch.logits[index] != 0;
}

void NativeBatch::clear_logits() {
  if (!impl_->allocated || !impl_->has_storage()) {
    return;
  }
  for (std::int32_t i = 0; i < impl_->batch.n_tokens; ++i) {
    impl_->batch.logits[i] = 0;
  }
}

void NativeBatch::set_last_logits() {
  if (!impl_->allocated || !impl_->has_storage() || impl_->batch.n_tokens <= 0) {
    return;
  }
  clear_logits();
  impl_->batch.logits[impl_->batch.n_tokens - 1] = 1;
}

std::unique_ptr<CommonSampler> create_sampler(
  const NativeRuntime & runtime,
  rust::Str sampling_json,
  rust::Str grammar,
  rust::Str json_schema) {
  const std::string sampling = to_c_string_argument(sampling_json, "sampler JSON");
  const std::string grammar_text = to_c_string_argument(grammar, "sampler grammar");
  const std::string schema_text = to_c_string_argument(json_schema, "sampler JSON schema");
  char * error = nullptr;
  sipp_common_sampler * sampler = sipp_common_sampler_init_from_json(
      runtime.impl_->init,
      sampling.c_str(),
      grammar_text.c_str(),
      schema_text.c_str(),
      &error);
  if (sampler == nullptr) {
    throw std::runtime_error(take_error(error, "failed to initialize sampler"));
  }

  auto impl = std::make_unique<CommonSampler::Impl>();
  impl->sampler = sampler;
  return std::unique_ptr<CommonSampler>(new CommonSampler(std::move(impl)));
}

CommonSampler::CommonSampler(std::unique_ptr<CommonSampler::Impl> impl) : impl_(std::move(impl)) {}

CommonSampler::~CommonSampler() = default;
CommonSampler::CommonSampler(CommonSampler &&) noexcept = default;
CommonSampler & CommonSampler::operator=(CommonSampler &&) noexcept = default;

bool CommonSampler::sampler_accept(std::int32_t token, bool accept_grammar) {
  return sipp_common_sampler_accept(impl_->sampler, token, accept_grammar);
}

void CommonSampler::sampler_reset() {
  sipp_common_sampler_reset(impl_->sampler);
}

bool CommonSampler::sampler_backend_sampling() const {
  return sipp_common_sampler_backend_sampling(impl_->sampler);
}

std::int32_t sampler_sample(CommonSampler & sampler, NativeRuntime & runtime, std::int32_t idx) {
  return sipp_common_sampler_sample(sampler.impl_->sampler, runtime.impl_->context(), idx);
}

bool sampler_attach(CommonSampler & sampler, NativeRuntime & runtime, std::int32_t seq_id) {
  return sipp_llama_set_sampler(
      runtime.impl_->context(),
      seq_id,
      sipp_common_sampler_raw(sampler.impl_->sampler));
}

bool sampler_detach(NativeRuntime & runtime, std::int32_t seq_id) {
  return sipp_llama_set_sampler(runtime.impl_->context(), seq_id, nullptr);
}

} // namespace sipp::sys
