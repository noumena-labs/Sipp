#ifndef SIPP_SHIM_H
#define SIPP_SHIM_H

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

typedef struct sipp_chat_templates sipp_chat_templates;
typedef struct sipp_common_params sipp_common_params;
typedef struct sipp_common_init sipp_common_init;
typedef struct sipp_common_sampler sipp_common_sampler;
typedef struct sipp_common_checkpoint sipp_common_checkpoint;
typedef struct sipp_mtmd_context sipp_mtmd_context;
typedef struct sipp_mtmd_bitmap sipp_mtmd_bitmap;
typedef struct sipp_mtmd_input_chunks sipp_mtmd_input_chunks;

sipp_common_params * sipp_common_params_parse_server(
    const char * model_path,
    int32_t argc,
    const char * const * argv,
    char ** error_out);

void sipp_common_params_free(sipp_common_params * params);

sipp_common_init * sipp_common_init_from_params(
    const sipp_common_params * params,
    char ** error_out);

void sipp_common_init_free(sipp_common_init * init);

struct llama_model * sipp_common_init_model(sipp_common_init * init);

struct llama_context * sipp_common_init_context(sipp_common_init * init);

const struct llama_vocab * sipp_common_init_vocab(sipp_common_init * init);

int32_t sipp_common_init_n_parallel(const sipp_common_init * init);

int32_t sipp_common_init_n_batch(const sipp_common_init * init);

int32_t sipp_common_init_n_ubatch(const sipp_common_init * init);

int32_t sipp_common_init_n_ctx(const sipp_common_init * init);

int32_t sipp_common_init_n_embd_out(const sipp_common_init * init);

int32_t sipp_common_init_n_cls_out(const sipp_common_init * init);

int32_t sipp_common_init_pooling_type(const sipp_common_init * init);

int32_t sipp_common_init_decoder_start_token(const sipp_common_init * init);

bool sipp_common_init_model_has_encoder(const sipp_common_init * init);

bool sipp_common_init_model_has_decoder(const sipp_common_init * init);

bool sipp_common_init_model_has_chat_template(const sipp_common_init * init);

bool sipp_common_init_kv_unified(const sipp_common_init * init);

char * sipp_common_init_flash_attention(const sipp_common_init * init);

char * sipp_common_init_cache_type_k(const sipp_common_init * init);

char * sipp_common_init_cache_type_v(const sipp_common_init * init);

sipp_common_sampler * sipp_common_sampler_init_from_json(
    sipp_common_init * init,
    const char * sampling_json,
    const char * grammar,
    const char * json_schema,
    char ** error_out);

void sipp_common_sampler_free(sipp_common_sampler * sampler);

struct llama_sampler * sipp_common_sampler_raw(sipp_common_sampler * sampler);

bool sipp_common_sampler_backend_sampling(const sipp_common_sampler * sampler);

char * sipp_common_sampler_print(const sipp_common_sampler * sampler);

void sipp_common_sampler_reset(sipp_common_sampler * sampler);

int32_t sipp_common_sampler_sample(
    sipp_common_sampler * sampler,
    struct llama_context * context,
    int32_t idx);

bool sipp_common_sampler_accept(
    sipp_common_sampler * sampler,
    int32_t token,
    bool is_generated);

bool sipp_llama_state_seq_get_data_ext_alloc(
    struct llama_context * context,
    int32_t seq_id,
    uint32_t flags,
    uint8_t ** data_out,
    size_t * size_out);

bool sipp_llama_state_seq_set_data_ext(
    struct llama_context * context,
    int32_t seq_id,
    uint32_t flags,
    const uint8_t * data,
    size_t size);

sipp_common_checkpoint * sipp_common_checkpoint_capture(
    struct llama_context * context,
    int32_t seq_id,
    uint32_t flags,
    char ** error_out);

bool sipp_common_checkpoint_restore(
    const sipp_common_checkpoint * checkpoint,
    struct llama_context * context,
    int32_t seq_id,
    uint32_t flags,
    char ** error_out);

size_t sipp_common_checkpoint_size(const sipp_common_checkpoint * checkpoint);

void sipp_common_checkpoint_free(sipp_common_checkpoint * checkpoint);

sipp_chat_templates * sipp_chat_templates_init(
    const struct llama_model * model,
    const char * chat_template_override);

void sipp_chat_templates_free(sipp_chat_templates * templates);

char * sipp_chat_templates_source(const sipp_chat_templates * templates);

char * sipp_apply_chat_template(
    const sipp_chat_templates * templates,
    const char * messages_json,
    bool add_assistant);

void sipp_set_llama_log_quiet(bool quiet);

void sipp_backend_load_all(void);

char * sipp_backend_observability_json(bool include_details);

bool sipp_llama_set_sampler(
    struct llama_context * context,
    int32_t seq_id,
    struct llama_sampler * sampler);

int32_t sipp_llama_decode(
    struct llama_context * context,
    const struct llama_batch * batch);

/// Encoder prompt ingest. No KV cache use; deposits cross-attention state
/// into the context for the subsequent decoder pass (encoder-decoder), or
/// produces sequence embeddings (encoder-only). Mirrors sipp_llama_decode's
/// error contract: 0 = ok, < 0 = error.
int32_t sipp_llama_encode(
    struct llama_context * context,
    const struct llama_batch * batch);

/// Read the pooled embedding for `seq_id`. Returned pointer is owned by the
/// context; caller copies before the next encode/decode. Returns NULL when
/// pooling is NONE or the context has no pooled output for the sequence.
const float * sipp_llama_embeddings_seq(
    struct llama_context * context,
    int32_t seq_id);

/// Read the per-token embedding at logical batch index `i`. Returned pointer
/// is owned by the context; caller copies before the next encode/decode.
/// Returns NULL when index is out of range.
const float * sipp_llama_embeddings_ith(
    struct llama_context * context,
    int32_t i);

bool sipp_llama_synchronize(struct llama_context * context);

int32_t sipp_llama_sampler_sample(
    struct llama_sampler * sampler,
    struct llama_context * context,
    int32_t idx);

bool sipp_llama_sampler_accept(struct llama_sampler * sampler, int32_t token);

const char * sipp_mtmd_default_marker(void);

sipp_mtmd_context * sipp_mtmd_init_from_file(
    const char * mmproj_path,
    const struct llama_model * text_model,
    bool use_gpu,
    int n_threads);

void sipp_mtmd_free(sipp_mtmd_context * context);

bool sipp_mtmd_support_vision(const sipp_mtmd_context * context);

sipp_mtmd_bitmap * sipp_mtmd_bitmap_init_from_buf(
    sipp_mtmd_context * context,
    const uint8_t * data,
    size_t len);

void sipp_mtmd_bitmap_free(sipp_mtmd_bitmap * bitmap);

sipp_mtmd_input_chunks * sipp_mtmd_input_chunks_init(void);

void sipp_mtmd_input_chunks_free(sipp_mtmd_input_chunks * chunks);

bool sipp_mtmd_tokenize(
    sipp_mtmd_context * context,
    sipp_mtmd_input_chunks * chunks,
    const char * text,
    bool add_special,
    bool parse_special,
    const sipp_mtmd_bitmap * const * bitmaps,
    size_t bitmap_count);

int32_t sipp_mtmd_eval_chunks(
    sipp_mtmd_context * context,
    struct llama_context * llama_context,
    const sipp_mtmd_input_chunks * chunks,
    int32_t n_past,
    int32_t seq_id,
    int32_t n_batch,
    bool logits_last,
    int32_t * new_n_past);

#ifdef __cplusplus
}
#endif

#endif
