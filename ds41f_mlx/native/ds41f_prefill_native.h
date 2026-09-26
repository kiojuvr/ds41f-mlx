#ifndef DS41F_PREFILL_NATIVE_H
#define DS41F_PREFILL_NATIVE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS41F_PREFILL_MAX_LAYERS 128u
#define DS41F_PREFILL_MAX_CHUNKS 256u
#define DS41F_PREFILL_MAX_SWEEP_COMMANDS 2048u
#define DS41F_PREFILL_MAX_ALLOCATIONS 32u

#define DS41F_DS4_AUTHORITY_SHA "0aaea5a238fb41a35106a551e73c8409dfb751ac"
#define DS41F_DS4_AUTHORITY_REMOTE "https://github.com/antirez/ds4.git"

#define DS41F_FRONTIER_ENGRAM     0x01u
#define DS41F_FRONTIER_KV         0x02u
#define DS41F_FRONTIER_INDEX_K    0x04u
#define DS41F_FRONTIER_IDX        0x08u
#define DS41F_FRONTIER_CANDIDATES 0x10u

typedef struct ds41f_prefill_native_config {
    uint32_t n_layers;
    uint32_t n_chunks;
    uint32_t dim;
    uint32_t hc_mult;
    uint32_t vocab_size;
    uint32_t bytes_per_hidden_scalar;
    uint32_t chunk_lengths[DS41F_PREFILL_MAX_CHUNKS];
    uint32_t frontier_masks[DS41F_PREFILL_MAX_LAYERS];
} ds41f_prefill_native_config;

typedef struct ds41f_prefill_native_plan {
    uint32_t n_tokens;
    uint32_t max_chunk_tokens;
    uint32_t layer_steps;
    uint32_t command_batches;
    uint32_t publication_frontiers;
    uint32_t output_rows_retained;
    uint64_t carry_buffer_bytes;
    uint64_t full_prompt_hidden_bytes;
    uint64_t last_chunk_logits_bytes;
    uint64_t full_prompt_logits_bytes;
} ds41f_prefill_native_plan;

typedef struct ds41f_prefill_native_arena_info {
    uint64_t carry_buffer_bytes;
    uint64_t token_buffer_bytes;
    uint64_t total_owned_bytes;
    uint64_t current_carry_checksum;
    uint64_t next_carry_checksum;
    uint32_t n_tokens_uploaded;
    uint32_t current_carry_index;
    uint32_t executed_encode_chunks;
    uint32_t last_encoded_layer;
    uint32_t last_encoded_chunk;
} ds41f_prefill_native_arena_info;

typedef enum ds41f_prefill_native_command_kind {
    DS41F_CMD_UPLOAD_TOKENS = 1,
    DS41F_CMD_UPLOAD_EMBEDDINGS_HC = 2,
    DS41F_CMD_BEGIN_LAYER = 3,
    DS41F_CMD_ENCODE_LAYER_CHUNK = 4,
    DS41F_CMD_PUBLISH_FRONTIER = 5,
    DS41F_CMD_SWAP_CARRY = 6,
    DS41F_CMD_END_LAYER = 7,
    DS41F_CMD_ENCODE_OUTPUT_HEAD = 8,
    DS41F_CMD_READ_LOGITS = 9
} ds41f_prefill_native_command_kind;

typedef struct ds41f_prefill_native_command {
    uint32_t kind;
    uint32_t layer;
    uint32_t chunk;
    uint32_t token_offset;
    uint32_t token_count;
    uint32_t frontier_mask;
} ds41f_prefill_native_command;

typedef enum ds41f_prefill_persistence_class {
    DS41F_PERSIST_SWEEP = 1,
    DS41F_PERSIST_LAYER = 2,
    DS41F_PERSIST_DEFERRED_DECODER = 3,
    DS41F_PERSIST_STAGE_LOCAL = 4
} ds41f_prefill_persistence_class;

typedef enum ds41f_prefill_sweep_command_kind {
    DS41F_SWEEP_BEGIN_INVALIDATE = 101,
    DS41F_SWEEP_PREFETCH_ENGRAM0 = 102,
    DS41F_SWEEP_PREFETCH_ENGRAM1 = 103,
    DS41F_SWEEP_SSD_READ_AHEAD = 104,
    DS41F_SWEEP_BEGIN_LAYER = 105,
    DS41F_SWEEP_ENCODE_ROWS = 106,
    DS41F_SWEEP_DECODER_PREPARE_SUFFIX = 107,
    DS41F_SWEEP_SWAP_HC_LAYER = 108,
    DS41F_SWEEP_PUBLISH_FRONTIER = 109,
    DS41F_SWEEP_ENCODER_ONLY_COMPLETE_INVALID = 110,
    DS41F_SWEEP_DECODER_PENDING_INVALID = 111,
    DS41F_SWEEP_ENCODE_OUTPUT_HEAD = 112,
    DS41F_SWEEP_READ_LOGITS = 113,
    DS41F_SWEEP_CHECKPOINT_MAY_COMMIT = 114,
    DS41F_SWEEP_END_LAYER = 115
} ds41f_prefill_sweep_command_kind;

typedef struct ds41f_prefill_sweep_config {
    uint32_t ctx;
    uint32_t remaining;
    uint32_t dim;
    uint32_t hc_mult;
    uint32_t vocab_size;
    uint32_t bytes_per_hidden_scalar;
    uint32_t encoder_resident;
    uint32_t encoder_only;
    uint32_t resume_encoder;
    uint32_t frontier_masks[DS41F_PREFILL_MAX_LAYERS];
    uint64_t memory_budget_bytes;
} ds41f_prefill_sweep_config;

typedef struct ds41f_prefill_sweep_allocation {
    char semantic_role[48];
    uint64_t size_bytes;
    uint32_t alignment;
    char representation[24];
    uint32_t first_use_step;
    uint32_t last_use_step;
    char owner[32];
    char alias_reuse_class[32];
    uint32_t persistence_class;
    uint32_t survives_encoder_only;
    uint32_t survives_deferred_decoder;
} ds41f_prefill_sweep_allocation;

typedef struct ds41f_prefill_sweep_command {
    uint32_t kind;
    uint32_t layer;
    uint32_t offset;
    uint32_t rows;
    uint32_t phase;
    uint32_t checkpoint_valid;
} ds41f_prefill_sweep_command;

typedef struct ds41f_prefill_buffer_slot {
    char semantic_role[48];
    uint64_t size_bytes;
    uint64_t checksum;
    uint32_t allocated;
    uint32_t touched;
    uint32_t persistence_class;
} ds41f_prefill_buffer_slot;

typedef struct ds41f_official_linear_result {
    uint64_t output_checksum;
    uint32_t rows;
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t metal_enabled;
    uint32_t metal_command_buffers;
    uint32_t metal_compute_encoders;
    uint32_t metal_completion_waits;
} ds41f_official_linear_result;

typedef struct ds41f_official_embedding_result {
    uint64_t output_checksum;
    uint32_t n_tokens;
    uint32_t dim;
    uint32_t vocab_rows;
    uint32_t metal_enabled;
    uint32_t metal_command_buffers;
    uint32_t metal_compute_encoders;
    uint32_t metal_completion_waits;
} ds41f_official_embedding_result;

typedef struct ds41f_official_rmsnorm_result {
    uint64_t output_checksum;
    uint32_t rows;
    uint32_t dim;
    uint32_t metal_enabled;
    uint32_t metal_command_buffers;
    uint32_t metal_compute_encoders;
    uint32_t metal_completion_waits;
} ds41f_official_rmsnorm_result;

typedef struct ds41f_official_rotary_result {
    uint64_t freqs_checksum;
    uint64_t rotated_checksum;
    uint64_t inverse_checksum;
    uint32_t batch;
    uint32_t seqlen;
    uint32_t heads;
    uint32_t dim;
    uint32_t original_seq_len;
    uint32_t metal_enabled;
    uint32_t metal_command_buffers;
    uint32_t metal_compute_encoders;
    uint32_t metal_completion_waits;
} ds41f_official_rotary_result;

typedef struct ds41f_prefill_submission_info {
    uint64_t total_owned_bytes;
    uint64_t memory_budget_bytes;
    uint32_t n_buffers;
    uint32_t submitted_commands;
    uint32_t submitted_command_buffers;
    uint32_t encoded_buffer_ops;
    uint32_t checkpoint_valid;
    uint32_t current_hc_slot;
    uint32_t failed_command_index;
    uint32_t metal_enabled;
    uint32_t metal_buffers_allocated;
    uint32_t metal_command_buffers_committed;
    uint32_t metal_blit_encoders_committed;
    uint32_t metal_completion_waits;
    uint32_t command_buffers_encode_rows;
    uint32_t command_buffers_decoder_prepare;
    uint32_t command_buffers_swap_hc;
    uint32_t command_buffers_output;
    uint32_t command_buffers_by_phase[8];
    uint32_t command_buffers_by_layer[DS41F_PREFILL_MAX_LAYERS];
    ds41f_prefill_buffer_slot buffers[DS41F_PREFILL_MAX_ALLOCATIONS];
} ds41f_prefill_submission_info;

typedef struct ds41f_prefill_sweep_plan {
    uint32_t count;
    uint32_t prefill_cap;
    uint32_t encoder_chunk;
    uint32_t wide;
    uint32_t decoder_suffix;
    uint32_t encoder_only;
    uint32_t resume_encoder;
    uint32_t defer_decoder_candidate;
    uint32_t checkpoint_valid_during_sweep;
    uint32_t checkpoint_valid_after_sweep;
    uint64_t encoder_row_layer_work;
    uint64_t decoder_suffix_row_layer_work;
    uint64_t total_row_layer_work;
    uint32_t n_allocations;
    uint32_t n_commands;
    uint32_t n_prefetch_events;
    uint32_t n_checkpoint_transitions;
    ds41f_prefill_sweep_allocation allocations[DS41F_PREFILL_MAX_ALLOCATIONS];
    ds41f_prefill_sweep_command commands[DS41F_PREFILL_MAX_SWEEP_COMMANDS];
} ds41f_prefill_sweep_plan;

typedef struct ds41f_prefill_native_context ds41f_prefill_native_context;
typedef struct ds41f_prefill_submission_context ds41f_prefill_submission_context;

int ds41f_prefill_native_build_plan(const ds41f_prefill_native_config *cfg,
                                    ds41f_prefill_native_plan *out);

int ds41f_prefill_native_context_create(const ds41f_prefill_native_config *cfg,
                                        ds41f_prefill_native_context **out);
void ds41f_prefill_native_context_destroy(ds41f_prefill_native_context *ctx);
int ds41f_prefill_native_context_info(const ds41f_prefill_native_context *ctx,
                                      ds41f_prefill_native_arena_info *out);
int ds41f_prefill_native_upload_tokens(ds41f_prefill_native_context *ctx,
                                       uint32_t offset,
                                       const int32_t *tokens,
                                       uint32_t count);
int ds41f_prefill_native_swap_carry(ds41f_prefill_native_context *ctx);
void *ds41f_prefill_native_current_carry(ds41f_prefill_native_context *ctx);
void *ds41f_prefill_native_next_carry(ds41f_prefill_native_context *ctx);
int ds41f_prefill_native_build_commands(ds41f_prefill_native_context *ctx);
uint32_t ds41f_prefill_native_command_count(const ds41f_prefill_native_context *ctx);
int ds41f_prefill_native_command_at(const ds41f_prefill_native_context *ctx,
                                    uint32_t index,
                                    ds41f_prefill_native_command *out);
int ds41f_prefill_native_execute_noop_graph(ds41f_prefill_native_context *ctx,
                                            uint32_t *submitted_commands);

int ds41f_prefill_native_build_sweep_plan(const ds41f_prefill_sweep_config *cfg,
                                          ds41f_prefill_sweep_plan *out);

int ds41f_prefill_submission_context_create(const ds41f_prefill_sweep_config *cfg,
                                            ds41f_prefill_submission_context **out);
void ds41f_prefill_submission_context_destroy(ds41f_prefill_submission_context *ctx);
int ds41f_prefill_submission_context_info(const ds41f_prefill_submission_context *ctx,
                                          ds41f_prefill_submission_info *out);
int ds41f_prefill_submission_submit(ds41f_prefill_submission_context *ctx,
                                    uint32_t *submitted_commands);

int ds41f_official_embedding_gather_bf16(const uint16_t *embedding_bf16,
                                         uint32_t vocab_rows,
                                         uint32_t dim,
                                         const int32_t *tokens,
                                         uint32_t n_tokens,
                                         uint16_t *out_bf16,
                                         ds41f_official_embedding_result *result);

int ds41f_official_bf16_linear_f32(const uint16_t *input_bf16,
                                   const uint16_t *weight_bf16,
                                   uint32_t rows,
                                   uint32_t in_dim,
                                   uint32_t out_dim,
                                   float *out_f32,
                                   ds41f_official_linear_result *result);

int ds41f_official_f32_linear_f32(const float *input_f32,
                                  const float *weight_f32,
                                  uint32_t rows,
                                  uint32_t in_dim,
                                  uint32_t out_dim,
                                  float *out_f32,
                                  ds41f_official_linear_result *result);

int ds41f_official_fp8_linear_bf16(const uint16_t *input_bf16,
                                   const uint8_t *weight_fp8,
                                   const uint8_t *weight_scale_e8m0,
                                   uint32_t rows,
                                   uint32_t in_dim,
                                   uint32_t out_dim,
                                   uint32_t block_size,
                                   uint16_t *out_bf16,
                                   ds41f_official_linear_result *result);

int ds41f_official_act_quant_bf16(const uint16_t *input_bf16,
                                  uint32_t rows,
                                  uint32_t dim,
                                  uint32_t block_size,
                                  uint8_t *out_fp8,
                                  uint8_t *out_scale_e8m0,
                                  uint16_t *out_bf16,
                                  ds41f_official_linear_result *result);

int ds41f_official_sparse_attn_bf16(const uint16_t *q_bf16,
                                    const uint16_t *kv_bf16,
                                    const float *attn_sink_f32,
                                    const int32_t *topk_idxs,
                                    uint32_t batch,
                                    uint32_t seqlen,
                                    uint32_t kv_len,
                                    uint32_t heads,
                                    uint32_t dim,
                                    uint32_t topk,
                                    float softmax_scale,
                                    uint16_t *out_bf16,
                                    ds41f_official_linear_result *result);

int ds41f_official_rmsnorm_bf16(const uint16_t *input_bf16,
                                const uint16_t *weight_bf16,
                                uint32_t rows,
                                uint32_t dim,
                                float eps,
                                uint16_t *out_bf16,
                                ds41f_official_rmsnorm_result *result);

int ds41f_official_rotary_f32(uint32_t batch,
                              uint32_t seqlen,
                              uint32_t heads,
                              uint32_t dim,
                              uint32_t original_seq_len,
                              float base,
                              float factor,
                              uint32_t beta_fast,
                              uint32_t beta_slow,
                              const float *input_f32,
                              float *freqs_real_f32,
                              float *freqs_imag_f32,
                              float *rotated_f32,
                              float *inverse_f32,
                              ds41f_official_rotary_result *result);

const char *ds41f_prefill_native_version(void);

#ifdef __cplusplus
}
#endif

#endif
