#include "cogent_shim.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <atomic>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

#include <nlohmann/json.hpp>

#include "arg.h"
#include "chat.h"
#include "ggml-backend.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"

struct cogent_chat_templates {
    common_chat_templates_ptr inner;
};

struct cogent_common_params {
    common_params inner;
};

struct cogent_common_init {
    common_params params;
    common_init_result_ptr inner;
};

struct cogent_common_sampler {
    common_params_sampling sampling;
    common_sampler * inner = nullptr;
};

struct cogent_common_checkpoint {
    std::vector<uint8_t> data;
    int32_t seq_id = -1;
    uint32_t flags = 0;
};

struct cogent_mtmd_context {
    mtmd_context * inner;
};

struct cogent_mtmd_bitmap {
    mtmd_bitmap * inner;
};

struct cogent_mtmd_input_chunks {
    mtmd_input_chunks * inner;
};

namespace {

char * copy_string(const std::string & value) {
    char * out = static_cast<char *>(std::malloc(value.size() + 1));
    if (out == nullptr) {
        return nullptr;
    }
    std::memcpy(out, value.data(), value.size());
    out[value.size()] = '\0';
    return out;
}

void set_error(char ** error_out, const std::string & value) {
    if (error_out == nullptr) {
        return;
    }
    *error_out = copy_string(value);
}

std::vector<std::string> build_common_argv(
    const char * model_path,
    int32_t argc,
    const char * const * argv) {
    std::vector<std::string> args;
    args.emplace_back("cogentlm");
    if (model_path != nullptr && model_path[0] != '\0') {
        args.emplace_back("--model");
        args.emplace_back(model_path);
    }
    for (int32_t i = 0; i < argc; ++i) {
        if (argv != nullptr && argv[i] != nullptr) {
            args.emplace_back(argv[i]);
        }
    }
    return args;
}

std::vector<char *> mutable_argv(std::vector<std::string> & args) {
    std::vector<char *> out;
    out.reserve(args.size());
    for (auto & arg : args) {
        out.push_back(arg.data());
    }
    return out;
}

template <typename T>
void set_if_present(const nlohmann::ordered_json & json, const char * key, T & target) {
    auto it = json.find(key);
    if (it != json.end() && !it->is_null()) {
        target = it->get<T>();
    }
}

void apply_sampling_json(
    common_params_sampling & sampling,
    const char * sampling_json,
    const char * grammar,
    const char * json_schema) {
    using json = nlohmann::ordered_json;
    if (sampling_json != nullptr && sampling_json[0] != '\0') {
        const json parsed = json::parse(sampling_json);

        if (auto it = parsed.find("samplers");
            it != parsed.end() && it->is_array() && !it->empty()) {
            std::vector<std::string> names;
            names.reserve(it->size());
            for (const auto & item : *it) {
                auto name = item.get<std::string>();
                if (name == "typical_p") {
                    name = "typ_p";
                }
                names.push_back(std::move(name));
            }
            sampling.samplers = common_sampler_types_from_names(names, true);
        }

        set_if_present(parsed, "seed", sampling.seed);
        set_if_present(parsed, "top_k", sampling.top_k);
        set_if_present(parsed, "top_p", sampling.top_p);
        set_if_present(parsed, "min_p", sampling.min_p);
        set_if_present(parsed, "typical_p", sampling.typ_p);
        set_if_present(parsed, "xtc_probability", sampling.xtc_probability);
        set_if_present(parsed, "xtc_threshold", sampling.xtc_threshold);
        set_if_present(parsed, "top_n_sigma", sampling.top_n_sigma);
        set_if_present(parsed, "temperature", sampling.temp);
        set_if_present(parsed, "dynatemp_range", sampling.dynatemp_range);
        set_if_present(parsed, "dynatemp_exponent", sampling.dynatemp_exponent);
        set_if_present(parsed, "repeat_last_n", sampling.penalty_last_n);
        set_if_present(parsed, "repeat_penalty", sampling.penalty_repeat);
        set_if_present(parsed, "frequency_penalty", sampling.penalty_freq);
        set_if_present(parsed, "presence_penalty", sampling.penalty_present);
        set_if_present(parsed, "dry_multiplier", sampling.dry_multiplier);
        set_if_present(parsed, "dry_base", sampling.dry_base);
        set_if_present(parsed, "dry_allowed_length", sampling.dry_allowed_length);
        set_if_present(parsed, "dry_penalty_last_n", sampling.dry_penalty_last_n);
        set_if_present(parsed, "mirostat", sampling.mirostat);
        set_if_present(parsed, "mirostat_tau", sampling.mirostat_tau);
        set_if_present(parsed, "mirostat_eta", sampling.mirostat_eta);
        set_if_present(parsed, "min_keep", sampling.min_keep);
        set_if_present(parsed, "n_probs", sampling.n_probs);
        set_if_present(parsed, "ignore_eos", sampling.ignore_eos);
        set_if_present(parsed, "grammar_lazy", sampling.grammar_lazy);
        set_if_present(parsed, "backend_sampling", sampling.backend_sampling);

        if (auto it = parsed.find("dry_sequence_breakers");
            it != parsed.end() && it->is_array() && !it->empty()) {
            sampling.dry_sequence_breakers.clear();
            for (const auto & item : *it) {
                sampling.dry_sequence_breakers.push_back(item.get<std::string>());
            }
        }

        if (auto it = parsed.find("logit_bias"); it != parsed.end() && it->is_array()) {
            sampling.logit_bias.clear();
            for (const auto & item : *it) {
                if (!item.is_object()) {
                    continue;
                }
                sampling.logit_bias.push_back({
                    item.value("token", LLAMA_TOKEN_NULL),
                    item.value("bias", 0.0f),
                });
            }
        }

        if (auto it = parsed.find("preserved_tokens"); it != parsed.end() && it->is_array()) {
            sampling.preserved_tokens.clear();
            for (const auto & item : *it) {
                sampling.preserved_tokens.insert(item.get<llama_token>());
            }
        }
    }

    if (grammar != nullptr && grammar[0] != '\0') {
        sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, grammar};
    }
    if (json_schema != nullptr && json_schema[0] != '\0') {
        sampling.grammar = {
            COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT,
            json_schema_to_grammar(nlohmann::ordered_json::parse(json_schema)),
        };
    }
}

bool parse_messages(const char * messages_json, std::vector<common_chat_msg> & out) {
    out.clear();
    if (messages_json == nullptr || messages_json[0] == '\0') {
        return false;
    }

    using json = nlohmann::ordered_json;
    const json parsed = json::parse(messages_json, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_array()) {
        return false;
    }

    out = common_chat_msgs_parse_oaicompat(parsed);
    return true;
}

const char * backend_dev_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
    case GGML_BACKEND_DEVICE_TYPE_CPU:
        return "CPU";
    case GGML_BACKEND_DEVICE_TYPE_GPU:
        return "GPU";
    case GGML_BACKEND_DEVICE_TYPE_IGPU:
        return "IGPU";
    case GGML_BACKEND_DEVICE_TYPE_ACCEL:
        return "ACCEL";
    case GGML_BACKEND_DEVICE_TYPE_META:
        return "META";
    default:
        return "UNKNOWN";
    }
}

std::atomic_bool g_llama_log_quiet{false};

void quiet_llama_log_callback(enum ggml_log_level, const char *, void *) {}

std::string linux_executable_directory() {
#if defined(__linux__)
    std::vector<char> path(4096);
    const ssize_t len = readlink("/proc/self/exe", path.data(), path.size() - 1);
    if (len <= 0) {
        return {};
    }

    path[static_cast<size_t>(len)] = '\0';
    std::string full_path(path.data(), static_cast<size_t>(len));
    const size_t separator = full_path.find_last_of('/');
    if (separator == std::string::npos) {
        return {};
    }
    return full_path.substr(0, separator);
#else
    return {};
#endif
}

void restore_llama_log_callback() {
#if defined(__EMSCRIPTEN__)
    if (g_llama_log_quiet.load()) {
        common_log_set_verbosity_thold(-1);
        llama_log_set(quiet_llama_log_callback, nullptr);
    } else {
        common_log_set_verbosity_thold(LOG_DEFAULT_LLAMA);
        llama_log_set(nullptr, nullptr);
    }
#else
    if (g_llama_log_quiet.load()) {
        common_log_pause(common_log_main());
        llama_log_set(quiet_llama_log_callback, nullptr);
    } else {
        common_log_resume(common_log_main());
        llama_log_set(nullptr, nullptr);
    }
#endif
}

struct llama_log_capture {
    std::string text;
};

void capture_llama_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    if (level < GGML_LOG_LEVEL_WARN || text == nullptr || user_data == nullptr) {
        return;
    }
    auto * capture = static_cast<llama_log_capture *>(user_data);
    if (capture->text.size() >= 4096) {
        return;
    }
    capture->text.append(text);
}

struct scoped_llama_log_capture {
    llama_log_capture capture;

    scoped_llama_log_capture() {
        llama_log_set(capture_llama_log_callback, &capture);
    }

    ~scoped_llama_log_capture() {
        restore_llama_log_callback();
    }
};

std::string trim_log_detail(std::string detail) {
    while (!detail.empty() && (detail.back() == '\n' || detail.back() == '\r' || detail.back() == ' ' || detail.back() == '\t')) {
        detail.pop_back();
    }
    size_t start = 0;
    while (start < detail.size() && (detail[start] == '\n' || detail[start] == '\r' || detail[start] == ' ' || detail[start] == '\t')) {
        ++start;
    }
    if (start > 0) {
        detail.erase(0, start);
    }
    if (detail.size() > 2048) {
        detail.resize(2048);
        detail.append("...");
    }
    return detail;
}

std::string common_init_failure_message(
    const common_params & params,
    const common_init_result_ptr & init,
    const std::string & log_detail) {
    std::string message = "common_init_from_params failed to load ";
    message += (!init || init->model() == nullptr) ? "model" : "context";
    if (!params.model.path.empty()) {
        message += " '";
        message += params.model.path;
        message += "'";
    }
    const std::string detail = trim_log_detail(log_detail);
    if (!detail.empty()) {
        message += ": ";
        message += detail;
    }
    return message;
}

} // namespace

cogent_common_params * cogent_common_params_parse_server(
    const char * model_path,
    int32_t argc,
    const char * const * argv,
    char ** error_out) {
    try {
        common_params params;
        auto args = build_common_argv(model_path, argc, argv);
        auto cargs = mutable_argv(args);
        if (!common_params_parse(
                static_cast<int>(cargs.size()),
                cargs.data(),
                params,
                LLAMA_EXAMPLE_SERVER)) {
            set_error(error_out, "llama.cpp common parameter parsing failed");
            return nullptr;
        }
        auto * out = new cogent_common_params();
        out->inner = std::move(params);
        return out;
    } catch (const std::exception & e) {
        set_error(error_out, e.what());
        return nullptr;
    } catch (...) {
        set_error(error_out, "unknown llama.cpp common parameter parsing failure");
        return nullptr;
    }
}

void cogent_common_params_free(cogent_common_params * params) {
    delete params;
}

cogent_common_init * cogent_common_init_from_params(
    const cogent_common_params * params,
    char ** error_out) {
    if (params == nullptr) {
        set_error(error_out, "common params pointer is null");
        return nullptr;
    }

    try {
        auto * out = new cogent_common_init();
        out->params = params->inner;
        scoped_llama_log_capture log_capture;
        out->inner = common_init_from_params(out->params);
        if (!out->inner || out->inner->model() == nullptr || out->inner->context() == nullptr) {
            const std::string error = common_init_failure_message(
                out->params,
                out->inner,
                log_capture.capture.text);
            delete out;
            set_error(error_out, error);
            return nullptr;
        }
        return out;
    } catch (const std::exception & e) {
        set_error(error_out, e.what());
        return nullptr;
    } catch (...) {
        set_error(error_out, "unknown common_init_from_params failure");
        return nullptr;
    }
}

void cogent_common_init_free(cogent_common_init * init) {
    delete init;
}

llama_model * cogent_common_init_model(cogent_common_init * init) {
    return init != nullptr && init->inner ? init->inner->model() : nullptr;
}

llama_context * cogent_common_init_context(cogent_common_init * init) {
    return init != nullptr && init->inner ? init->inner->context() : nullptr;
}

const llama_vocab * cogent_common_init_vocab(cogent_common_init * init) {
    llama_model * model = cogent_common_init_model(init);
    return model != nullptr ? llama_model_get_vocab(model) : nullptr;
}

int32_t cogent_common_init_n_parallel(const cogent_common_init * init) {
    if (init == nullptr || !init->inner || init->inner->context() == nullptr) {
        return 0;
    }
    return static_cast<int32_t>(llama_n_seq_max(init->inner->context()));
}

int32_t cogent_common_init_n_batch(const cogent_common_init * init) {
    if (init == nullptr || !init->inner || init->inner->context() == nullptr) {
        return 0;
    }
    return static_cast<int32_t>(llama_n_batch(init->inner->context()));
}

int32_t cogent_common_init_n_ubatch(const cogent_common_init * init) {
    if (init == nullptr || !init->inner || init->inner->context() == nullptr) {
        return 0;
    }
    return static_cast<int32_t>(llama_n_ubatch(init->inner->context()));
}

int32_t cogent_common_init_n_ctx(const cogent_common_init * init) {
    if (init == nullptr || !init->inner || init->inner->context() == nullptr) {
        return 0;
    }
    return static_cast<int32_t>(llama_n_ctx(init->inner->context()));
}

int32_t cogent_common_init_n_embd_out(const cogent_common_init * init) {
    if (init == nullptr || !init->inner || init->inner->model() == nullptr) {
        return 0;
    }
    return llama_model_n_embd_out(init->inner->model());
}

int32_t cogent_common_init_n_cls_out(const cogent_common_init * init) {
    if (init == nullptr || !init->inner || init->inner->model() == nullptr) {
        return 0;
    }
    return static_cast<int32_t>(llama_model_n_cls_out(init->inner->model()));
}

int32_t cogent_common_init_pooling_type(const cogent_common_init * init) {
    if (init == nullptr || !init->inner || init->inner->context() == nullptr) {
        return static_cast<int32_t>(LLAMA_POOLING_TYPE_UNSPECIFIED);
    }
    return static_cast<int32_t>(llama_pooling_type(init->inner->context()));
}

int32_t cogent_common_init_decoder_start_token(const cogent_common_init * init) {
    if (init == nullptr || !init->inner || init->inner->model() == nullptr) {
        return LLAMA_TOKEN_NULL;
    }
    return llama_model_decoder_start_token(init->inner->model());
}

bool cogent_common_init_model_has_encoder(const cogent_common_init * init) {
    return init != nullptr && init->inner && init->inner->model() != nullptr &&
        llama_model_has_encoder(init->inner->model());
}

bool cogent_common_init_model_has_decoder(const cogent_common_init * init) {
    return init != nullptr && init->inner && init->inner->model() != nullptr &&
        llama_model_has_decoder(init->inner->model());
}

bool cogent_common_init_model_has_chat_template(const cogent_common_init * init) {
    // Probe `tokenizer.chat_template` directly from GGUF metadata, NOT the
    // common_chat_templates fallback chain — we want to know whether the
    // model itself was trained with a chat template (and is therefore usable
    // through chat()), not whether llama.cpp can synthesize one.
    if (init == nullptr || !init->inner || init->inner->model() == nullptr) {
        return false;
    }
    return llama_model_chat_template(init->inner->model(), nullptr) != nullptr;
}

bool cogent_common_init_kv_unified(const cogent_common_init * init) {
    return init != nullptr && init->params.kv_unified;
}

char * cogent_common_init_flash_attention(const cogent_common_init * init) {
    if (init == nullptr) {
        return nullptr;
    }
    return copy_string(llama_flash_attn_type_name(init->params.flash_attn_type));
}

char * cogent_common_init_cache_type_k(const cogent_common_init * init) {
    if (init == nullptr) {
        return nullptr;
    }
    return copy_string(ggml_type_name(init->params.cache_type_k));
}

char * cogent_common_init_cache_type_v(const cogent_common_init * init) {
    if (init == nullptr) {
        return nullptr;
    }
    return copy_string(ggml_type_name(init->params.cache_type_v));
}

cogent_common_sampler * cogent_common_sampler_init_from_json(
    cogent_common_init * init,
    const char * sampling_json,
    const char * grammar,
    const char * json_schema,
    char ** error_out) {
    if (init == nullptr || !init->inner || init->inner->model() == nullptr) {
        set_error(error_out, "common init pointer is null");
        return nullptr;
    }

    try {
        auto * out = new cogent_common_sampler();
        out->sampling = init->params.sampling;
        apply_sampling_json(out->sampling, sampling_json, grammar, json_schema);
        out->inner = common_sampler_init(init->inner->model(), out->sampling);
        if (out->inner == nullptr) {
            delete out;
            set_error(error_out, "common_sampler_init returned null");
            return nullptr;
        }
        return out;
    } catch (const std::exception & e) {
        set_error(error_out, e.what());
        return nullptr;
    } catch (...) {
        set_error(error_out, "unknown common_sampler_init failure");
        return nullptr;
    }
}

void cogent_common_sampler_free(cogent_common_sampler * sampler) {
    if (sampler == nullptr) {
        return;
    }
    if (sampler->inner != nullptr) {
        common_sampler_free(sampler->inner);
    }
    delete sampler;
}

llama_sampler * cogent_common_sampler_raw(cogent_common_sampler * sampler) {
    return sampler != nullptr && sampler->inner != nullptr
        ? common_sampler_get(sampler->inner)
        : nullptr;
}

bool cogent_common_sampler_backend_sampling(const cogent_common_sampler * sampler) {
    return sampler != nullptr && sampler->sampling.backend_sampling;
}

char * cogent_common_sampler_print(const cogent_common_sampler * sampler) {
    if (sampler == nullptr || sampler->inner == nullptr) {
        return nullptr;
    }
    try {
        return copy_string(common_sampler_print(sampler->inner));
    } catch (const std::exception &) {
        return nullptr;
    }
}

int32_t cogent_common_sampler_sample(
    cogent_common_sampler * sampler,
    llama_context * context,
    int32_t idx) {
    if (sampler == nullptr || sampler->inner == nullptr || context == nullptr) {
        return LLAMA_TOKEN_NULL;
    }
    try {
        return common_sampler_sample(sampler->inner, context, idx);
    } catch (const std::exception &) {
        return LLAMA_TOKEN_NULL;
    } catch (...) {
        return LLAMA_TOKEN_NULL;
    }
}

bool cogent_common_sampler_accept(
    cogent_common_sampler * sampler,
    int32_t token,
    bool is_generated) {
    if (sampler == nullptr || sampler->inner == nullptr || token == LLAMA_TOKEN_NULL) {
        return false;
    }
    try {
        common_sampler_accept(sampler->inner, token, is_generated);
        return true;
    } catch (const std::exception &) {
        return false;
    } catch (...) {
        return false;
    }
}

bool cogent_llama_state_seq_get_data_ext_alloc(
    llama_context * context,
    int32_t seq_id,
    uint32_t flags,
    uint8_t ** data_out,
    size_t * size_out) {
    if (context == nullptr || seq_id < 0 || data_out == nullptr || size_out == nullptr) {
        return false;
    }
    *data_out = nullptr;
    *size_out = 0;
    try {
        const auto typed_flags = static_cast<llama_state_seq_flags>(flags);
        const size_t size = llama_state_seq_get_size_ext(context, seq_id, typed_flags);
        if (size == 0) {
            return false;
        }
        auto * data = static_cast<uint8_t *>(std::malloc(size));
        if (data == nullptr) {
            return false;
        }
        const size_t copied =
            llama_state_seq_get_data_ext(context, data, size, seq_id, typed_flags);
        if (copied != size) {
            std::free(data);
            return false;
        }
        *data_out = data;
        *size_out = size;
        return true;
    } catch (const std::exception &) {
        return false;
    } catch (...) {
        return false;
    }
}

bool cogent_llama_state_seq_set_data_ext(
    llama_context * context,
    int32_t seq_id,
    uint32_t flags,
    const uint8_t * data,
    size_t size) {
    if (context == nullptr || seq_id < 0 || data == nullptr || size == 0) {
        return false;
    }
    try {
        const auto typed_flags = static_cast<llama_state_seq_flags>(flags);
        const size_t restored =
            llama_state_seq_set_data_ext(context, data, size, seq_id, typed_flags);
        return restored == size;
    } catch (const std::exception &) {
        return false;
    } catch (...) {
        return false;
    }
}

cogent_common_checkpoint * cogent_common_checkpoint_capture(
    llama_context * context,
    int32_t seq_id,
    uint32_t flags,
    char ** error_out) {
    if (context == nullptr || seq_id < 0) {
        set_error(error_out, "checkpoint capture received null context or invalid sequence id");
        return nullptr;
    }
    try {
        const auto typed_flags = static_cast<llama_state_seq_flags>(flags);
        const size_t size = llama_state_seq_get_size_ext(context, seq_id, typed_flags);
        if (size == 0) {
            set_error(error_out, "checkpoint capture returned empty state");
            return nullptr;
        }
        auto * checkpoint = new cogent_common_checkpoint();
        checkpoint->data.resize(size);
        checkpoint->seq_id = seq_id;
        checkpoint->flags = flags;
        const size_t copied = llama_state_seq_get_data_ext(
            context,
            checkpoint->data.data(),
            checkpoint->data.size(),
            seq_id,
            typed_flags);
        if (copied != checkpoint->data.size()) {
            delete checkpoint;
            set_error(error_out, "checkpoint capture copied an unexpected byte count");
            return nullptr;
        }
        return checkpoint;
    } catch (const std::exception & e) {
        set_error(error_out, e.what());
        return nullptr;
    } catch (...) {
        set_error(error_out, "unknown checkpoint capture failure");
        return nullptr;
    }
}

bool cogent_common_checkpoint_restore(
    const cogent_common_checkpoint * checkpoint,
    llama_context * context,
    int32_t seq_id,
    uint32_t flags,
    char ** error_out) {
    if (checkpoint == nullptr || context == nullptr || seq_id < 0) {
        set_error(error_out, "checkpoint restore received null checkpoint/context or invalid sequence id");
        return false;
    }
    try {
        const auto typed_flags = static_cast<llama_state_seq_flags>(flags);
        const size_t restored = llama_state_seq_set_data_ext(
            context,
            checkpoint->data.data(),
            checkpoint->data.size(),
            seq_id,
            typed_flags);
        if (restored != checkpoint->data.size()) {
            set_error(error_out, "checkpoint restore copied an unexpected byte count");
            return false;
        }
        return true;
    } catch (const std::exception & e) {
        set_error(error_out, e.what());
        return false;
    } catch (...) {
        set_error(error_out, "unknown checkpoint restore failure");
        return false;
    }
}

size_t cogent_common_checkpoint_size(const cogent_common_checkpoint * checkpoint) {
    return checkpoint != nullptr ? checkpoint->data.size() : 0;
}

void cogent_common_checkpoint_free(cogent_common_checkpoint * checkpoint) {
    delete checkpoint;
}

cogent_chat_templates * cogent_chat_templates_init(
    const llama_model * model,
    const char * chat_template_override) {
    if (model == nullptr) {
        return nullptr;
    }

    try {
        auto * out = new cogent_chat_templates();
        out->inner = common_chat_templates_init(
            model,
            chat_template_override != nullptr ? chat_template_override : "");
        if (!out->inner) {
            delete out;
            return nullptr;
        }
        return out;
    } catch (const std::exception &) {
        return nullptr;
    }
}

void cogent_chat_templates_free(cogent_chat_templates * templates) {
    delete templates;
}

char * cogent_chat_templates_source(const cogent_chat_templates * templates) {
    if (templates == nullptr || !templates->inner) {
        return nullptr;
    }

    try {
        return copy_string(common_chat_templates_source(templates->inner.get()));
    } catch (const std::exception &) {
        return nullptr;
    }
}

char * cogent_apply_chat_template(
    const cogent_chat_templates * templates,
    const char * messages_json,
    bool add_assistant) {
    if (templates == nullptr || !templates->inner) {
        return nullptr;
    }

    try {
        std::vector<common_chat_msg> messages;
        if (!parse_messages(messages_json, messages)) {
            return nullptr;
        }

        common_chat_templates_inputs inputs;
        inputs.messages = std::move(messages);
        inputs.add_generation_prompt = add_assistant;
        inputs.use_jinja = true;

        return copy_string(common_chat_templates_apply(templates->inner.get(), inputs).prompt);
    } catch (const std::exception &) {
        return nullptr;
    }
}

void cogent_set_llama_log_quiet(bool quiet) {
    g_llama_log_quiet.store(quiet);
    restore_llama_log_callback();
}

void cogent_backend_load_all(void) {
#ifdef GGML_BACKEND_DL
    const std::string executable_dir = linux_executable_directory();
    if (!executable_dir.empty()) {
        ggml_backend_load_all_from_path(executable_dir.c_str());
        if (ggml_backend_reg_by_name("CPU") != nullptr) {
            return;
        }
    }
#endif
    ggml_backend_load_all();
}

char * cogent_backend_observability_json(bool include_details) {
    try {
        using json = nlohmann::ordered_json;
        json out;
        json compiled;

#ifdef GGML_BACKEND_DL
        out["dynamicBackendLoading"] = true;
#else
        out["dynamicBackendLoading"] = false;
#endif

#ifdef GGML_USE_CUDA
        compiled["cuda"] = true;
#else
        compiled["cuda"] = false;
#endif
#ifdef GGML_USE_METAL
        compiled["metal"] = true;
#else
        compiled["metal"] = false;
#endif
#ifdef GGML_USE_VULKAN
        compiled["vulkan"] = true;
#else
        compiled["vulkan"] = false;
#endif
#ifdef GGML_USE_OPENMP
        compiled["openmp"] = true;
#else
        compiled["openmp"] = false;
#endif
#ifdef GGML_USE_WEBGPU
        compiled["webgpu"] = true;
#else
        compiled["webgpu"] = false;
#endif

        out["compiled"] = compiled;
        out["gpuOffloadSupported"] = llama_supports_gpu_offload();
        out["backendCount"] = ggml_backend_reg_count();
        out["deviceCount"] = ggml_backend_dev_count();

        json backends = json::array();
        if (include_details) {
            const size_t backend_count = ggml_backend_reg_count();
            for (size_t i = 0; i < backend_count; ++i) {
                ggml_backend_reg_t reg = ggml_backend_reg_get(i);
                json item;
                item["name"] = ggml_backend_reg_name(reg) != nullptr
                                   ? ggml_backend_reg_name(reg)
                                   : "";
                item["deviceCount"] = ggml_backend_reg_dev_count(reg);
                backends.push_back(std::move(item));
            }
        }
        out["availableBackends"] = std::move(backends);

        json devices = json::array();
        if (include_details) {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                ggml_backend_dev_props props{};
                ggml_backend_dev_get_props(dev, &props);
                ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);

                json item;
                item["name"] = props.name != nullptr ? props.name : "";
                item["description"] = props.description != nullptr ? props.description : "";
                item["type"] = backend_dev_type_name(props.type);
                item["backendName"] =
                    reg != nullptr && ggml_backend_reg_name(reg) != nullptr
                        ? ggml_backend_reg_name(reg)
                        : "";
                if (props.device_id != nullptr && props.device_id[0] != '\0') {
                    item["deviceId"] = props.device_id;
                } else {
                    item["deviceId"] = nullptr;
                }
                item["memoryFreeBytes"] = props.memory_free;
                item["memoryTotalBytes"] = props.memory_total;
                item["capabilities"] = {
                    {"async", props.caps.async},
                    {"hostBuffer", props.caps.host_buffer},
                    {"bufferFromHostPtr", props.caps.buffer_from_host_ptr},
                    {"events", props.caps.events},
                };
                devices.push_back(std::move(item));
            }
        }
        out["devices"] = std::move(devices);
        return copy_string(out.dump());
    } catch (const std::exception &) {
        return nullptr;
    }
}

bool cogent_llama_set_sampler(
    llama_context * context,
    int32_t seq_id,
    llama_sampler * sampler) {
    if (context == nullptr || seq_id < 0) {
        return false;
    }

    try {
        return llama_set_sampler(context, static_cast<llama_seq_id>(seq_id), sampler);
    } catch (const std::exception &) {
        return false;
    } catch (...) {
        return false;
    }
}

int32_t cogent_llama_decode(llama_context * context, const llama_batch * batch) {
    if (context == nullptr || batch == nullptr) {
        return -1;
    }

    try {
        return llama_decode(context, *batch);
    } catch (const std::exception &) {
        return -1;
    } catch (...) {
        return -1;
    }
}

int32_t cogent_llama_encode(llama_context * context, const llama_batch * batch) {
    if (context == nullptr || batch == nullptr) {
        return -1;
    }

    try {
        return llama_encode(context, *batch);
    } catch (const std::exception &) {
        return -1;
    } catch (...) {
        return -1;
    }
}

const float * cogent_llama_embeddings_seq(llama_context * context, int32_t seq_id) {
    if (context == nullptr || seq_id < 0) {
        return nullptr;
    }

    try {
        return llama_get_embeddings_seq(context, static_cast<llama_seq_id>(seq_id));
    } catch (const std::exception &) {
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

const float * cogent_llama_embeddings_ith(llama_context * context, int32_t i) {
    if (context == nullptr) {
        return nullptr;
    }

    try {
        return llama_get_embeddings_ith(context, i);
    } catch (const std::exception &) {
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

bool cogent_llama_synchronize(llama_context * context) {
    if (context == nullptr) {
        return false;
    }

    try {
        llama_synchronize(context);
        return true;
    } catch (const std::exception &) {
        return false;
    } catch (...) {
        return false;
    }
}

int32_t cogent_llama_sampler_sample(
    llama_sampler * sampler,
    llama_context * context,
    int32_t idx) {
    if (sampler == nullptr || context == nullptr) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        return llama_sampler_sample(sampler, context, idx);
    } catch (const std::exception &) {
        return LLAMA_TOKEN_NULL;
    } catch (...) {
        return LLAMA_TOKEN_NULL;
    }
}

bool cogent_llama_sampler_accept(llama_sampler * sampler, int32_t token) {
    if (sampler == nullptr || token == LLAMA_TOKEN_NULL) {
        return false;
    }

    try {
        llama_sampler_accept(sampler, token);
        return true;
    } catch (const std::exception &) {
        return false;
    } catch (...) {
        return false;
    }
}

const char * cogent_mtmd_default_marker(void) {
    return mtmd_default_marker();
}

cogent_mtmd_context * cogent_mtmd_init_from_file(
    const char * mmproj_path,
    const llama_model * text_model,
    bool use_gpu,
    int n_threads) {
    if (mmproj_path == nullptr || mmproj_path[0] == '\0' || text_model == nullptr) {
        return nullptr;
    }

    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = use_gpu;
    params.print_timings = false;
    params.n_threads = n_threads;

    mtmd_context * inner = mtmd_init_from_file(mmproj_path, text_model, params);
    if (inner == nullptr) {
        return nullptr;
    }

    auto * context = new cogent_mtmd_context();
    context->inner = inner;
    return context;
}

void cogent_mtmd_free(cogent_mtmd_context * context) {
    if (context == nullptr) {
        return;
    }

    if (context->inner != nullptr) {
        mtmd_free(context->inner);
    }
    delete context;
}

bool cogent_mtmd_support_vision(const cogent_mtmd_context * context) {
    return context != nullptr && context->inner != nullptr && mtmd_support_vision(context->inner);
}

cogent_mtmd_bitmap * cogent_mtmd_bitmap_init_from_buf(
    cogent_mtmd_context * context,
    const uint8_t * data,
    size_t len) {
    if (context == nullptr || context->inner == nullptr || data == nullptr || len == 0) {
        return nullptr;
    }

    mtmd_bitmap * inner = mtmd_helper_bitmap_init_from_buf(context->inner, data, len);
    if (inner == nullptr) {
        return nullptr;
    }
    auto * bitmap = new cogent_mtmd_bitmap();
    bitmap->inner = inner;
    return bitmap;
}

void cogent_mtmd_bitmap_free(cogent_mtmd_bitmap * bitmap) {
    if (bitmap == nullptr) {
        return;
    }
    if (bitmap->inner != nullptr) {
        mtmd_bitmap_free(bitmap->inner);
    }
    delete bitmap;
}

cogent_mtmd_input_chunks * cogent_mtmd_input_chunks_init(void) {
    mtmd_input_chunks * inner = mtmd_input_chunks_init();
    if (inner == nullptr) {
        return nullptr;
    }
    auto * chunks = new cogent_mtmd_input_chunks();
    chunks->inner = inner;
    return chunks;
}

void cogent_mtmd_input_chunks_free(cogent_mtmd_input_chunks * chunks) {
    if (chunks == nullptr) {
        return;
    }
    if (chunks->inner != nullptr) {
        mtmd_input_chunks_free(chunks->inner);
    }
    delete chunks;
}

bool cogent_mtmd_tokenize(
    cogent_mtmd_context * context,
    cogent_mtmd_input_chunks * chunks,
    const char * text,
    bool add_special,
    bool parse_special,
    const cogent_mtmd_bitmap * const * bitmaps,
    size_t bitmap_count) {
    if (context == nullptr || context->inner == nullptr || chunks == nullptr ||
        chunks->inner == nullptr || text == nullptr) {
        return false;
    }

    std::vector<const mtmd_bitmap *> inner_bitmaps;
    inner_bitmaps.reserve(bitmap_count);
    for (size_t i = 0; i < bitmap_count; ++i) {
        if (bitmaps == nullptr || bitmaps[i] == nullptr || bitmaps[i]->inner == nullptr) {
            return false;
        }
        inner_bitmaps.push_back(bitmaps[i]->inner);
    }

    mtmd_input_text text_input{};
    text_input.text = text;
    text_input.add_special = add_special;
    text_input.parse_special = parse_special;

    return mtmd_tokenize(
               context->inner,
               chunks->inner,
               &text_input,
               inner_bitmaps.empty() ? nullptr : inner_bitmaps.data(),
               inner_bitmaps.size()) == 0;
}

int32_t cogent_mtmd_eval_chunks(
    cogent_mtmd_context * context,
    llama_context * llama_context,
    const cogent_mtmd_input_chunks * chunks,
    int32_t n_past,
    int32_t seq_id,
    int32_t n_batch,
    bool logits_last,
    int32_t * new_n_past) {
    if (context == nullptr || context->inner == nullptr || llama_context == nullptr ||
        chunks == nullptr || chunks->inner == nullptr || new_n_past == nullptr) {
        return -1;
    }

    llama_pos updated_n_past = 0;
    const int32_t status = mtmd_helper_eval_chunks(
        context->inner,
        llama_context,
        chunks->inner,
        n_past,
        seq_id,
        n_batch,
        logits_last,
        &updated_n_past);
    *new_n_past = static_cast<int32_t>(updated_n_past);
    return status;
}
