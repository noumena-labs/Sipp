#ifndef COGENTLM_SHIM_H
#define COGENTLM_SHIM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_model;
struct llama_context;
struct llama_batch;
struct llama_sampler;
struct llama_vocab;

typedef struct cogent_chat_templates cogent_chat_templates;
typedef struct cogent_common_params cogent_common_params;
typedef struct cogent_common_init cogent_common_init;
typedef struct cogent_common_sampler cogent_common_sampler;
typedef struct cogent_common_checkpoint cogent_common_checkpoint;
typedef struct cogent_mtmd_context cogent_mtmd_context;
typedef struct cogent_mtmd_bitmap cogent_mtmd_bitmap;
typedef struct cogent_mtmd_input_chunks cogent_mtmd_input_chunks;

cogent_common_params * cogent_common_params_parse_server(
    const char * model_path,
    int32_t argc,
    const char * const * argv,
    char ** error_out);

void cogent_common_params_free(cogent_common_params * params);

cogent_common_init * cogent_common_init_from_params(
    const cogent_common_params * params,
    char ** error_out);

void cogent_common_init_free(cogent_common_init * init);

struct llama_model * cogent_common_init_model(cogent_common_init * init);

struct llama_context * cogent_common_init_context(cogent_common_init * init);

const struct llama_vocab * cogent_common_init_vocab(cogent_common_init * init);

int32_t cogent_common_init_n_parallel(const cogent_common_init * init);

int32_t cogent_common_init_n_batch(const cogent_common_init * init);

int32_t cogent_common_init_n_ubatch(const cogent_common_init * init);

int32_t cogent_common_init_n_ctx(const cogent_common_init * init);

int32_t cogent_common_init_n_embd_out(const cogent_common_init * init);

int32_t cogent_common_init_n_cls_out(const cogent_common_init * init);

int32_t cogent_common_init_pooling_type(const cogent_common_init * init);

int32_t cogent_common_init_decoder_start_token(const cogent_common_init * init);

bool cogent_common_init_model_has_encoder(const cogent_common_init * init);

bool cogent_common_init_model_has_decoder(const cogent_common_init * init);

bool cogent_common_init_model_has_chat_template(const cogent_common_init * init);

bool cogent_common_init_kv_unified(const cogent_common_init * init);

char * cogent_common_init_flash_attention(const cogent_common_init * init);

char * cogent_common_init_cache_type_k(const cogent_common_init * init);

char * cogent_common_init_cache_type_v(const cogent_common_init * init);

cogent_common_sampler * cogent_common_sampler_init_from_json(
    cogent_common_init * init,
    const char * sampling_json,
    const char * grammar,
    const char * json_schema,
    char ** error_out);

void cogent_common_sampler_free(cogent_common_sampler * sampler);

struct llama_sampler * cogent_common_sampler_raw(cogent_common_sampler * sampler);

bool cogent_common_sampler_backend_sampling(const cogent_common_sampler * sampler);

char * cogent_common_sampler_print(const cogent_common_sampler * sampler);

void cogent_common_sampler_reset(cogent_common_sampler * sampler);

int32_t cogent_common_sampler_sample(
    cogent_common_sampler * sampler,
    struct llama_context * context,
    int32_t idx);

bool cogent_common_sampler_accept(
    cogent_common_sampler * sampler,
    int32_t token,
    bool is_generated);

bool cogent_llama_state_seq_get_data_ext_alloc(
    struct llama_context * context,
    int32_t seq_id,
    uint32_t flags,
    uint8_t ** data_out,
    size_t * size_out);

bool cogent_llama_state_seq_set_data_ext(
    struct llama_context * context,
    int32_t seq_id,
    uint32_t flags,
    const uint8_t * data,
    size_t size);

cogent_common_checkpoint * cogent_common_checkpoint_capture(
    struct llama_context * context,
    int32_t seq_id,
    uint32_t flags,
    char ** error_out);

bool cogent_common_checkpoint_restore(
    const cogent_common_checkpoint * checkpoint,
    struct llama_context * context,
    int32_t seq_id,
    uint32_t flags,
    char ** error_out);

size_t cogent_common_checkpoint_size(const cogent_common_checkpoint * checkpoint);

void cogent_common_checkpoint_free(cogent_common_checkpoint * checkpoint);

cogent_chat_templates * cogent_chat_templates_init(
    const struct llama_model * model,
    const char * chat_template_override);

void cogent_chat_templates_free(cogent_chat_templates * templates);

char * cogent_chat_templates_source(const cogent_chat_templates * templates);

char * cogent_apply_chat_template(
    const cogent_chat_templates * templates,
    const char * messages_json,
    bool add_assistant);

void cogent_set_llama_log_quiet(bool quiet);

void cogent_backend_load_all(void);

char * cogent_backend_observability_json(bool include_details);

bool cogent_llama_set_sampler(
    struct llama_context * context,
    int32_t seq_id,
    struct llama_sampler * sampler);

int32_t cogent_llama_decode(
    struct llama_context * context,
    const struct llama_batch * batch);

/// Encoder prompt ingest. No KV cache use; deposits cross-attention state
/// into the context for the subsequent decoder pass (encoder-decoder), or
/// produces sequence embeddings (encoder-only). Mirrors cogent_llama_decode's
/// error contract: 0 = ok, < 0 = error.
int32_t cogent_llama_encode(
    struct llama_context * context,
    const struct llama_batch * batch);

/// Read the pooled embedding for `seq_id`. Returned pointer is owned by the
/// context; caller copies before the next encode/decode. Returns NULL when
/// pooling is NONE or the context has no pooled output for the sequence.
const float * cogent_llama_embeddings_seq(
    struct llama_context * context,
    int32_t seq_id);

/// Read the per-token embedding at logical batch index `i`. Returned pointer
/// is owned by the context; caller copies before the next encode/decode.
/// Returns NULL when index is out of range.
const float * cogent_llama_embeddings_ith(
    struct llama_context * context,
    int32_t i);

bool cogent_llama_synchronize(struct llama_context * context);

int32_t cogent_llama_sampler_sample(
    struct llama_sampler * sampler,
    struct llama_context * context,
    int32_t idx);

bool cogent_llama_sampler_accept(struct llama_sampler * sampler, int32_t token);

const char * cogent_mtmd_default_marker(void);

cogent_mtmd_context * cogent_mtmd_init_from_file(
    const char * mmproj_path,
    const struct llama_model * text_model,
    bool use_gpu,
    int n_threads);

void cogent_mtmd_free(cogent_mtmd_context * context);

bool cogent_mtmd_support_vision(const cogent_mtmd_context * context);

cogent_mtmd_bitmap * cogent_mtmd_bitmap_init_from_buf(
    cogent_mtmd_context * context,
    const uint8_t * data,
    size_t len);

void cogent_mtmd_bitmap_free(cogent_mtmd_bitmap * bitmap);

cogent_mtmd_input_chunks * cogent_mtmd_input_chunks_init(void);

void cogent_mtmd_input_chunks_free(cogent_mtmd_input_chunks * chunks);

bool cogent_mtmd_tokenize(
    cogent_mtmd_context * context,
    cogent_mtmd_input_chunks * chunks,
    const char * text,
    bool add_special,
    bool parse_special,
    const cogent_mtmd_bitmap * const * bitmaps,
    size_t bitmap_count);

int32_t cogent_mtmd_eval_chunks(
    cogent_mtmd_context * context,
    struct llama_context * llama_context,
    const cogent_mtmd_input_chunks * chunks,
    int32_t n_past,
    int32_t seq_id,
    int32_t n_batch,
    bool logits_last,
    int32_t * new_n_past);

#ifdef __cplusplus
}
#endif

#endif
