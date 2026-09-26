#include "ds41f_prefill_native.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(DS41F_ENABLE_METAL) && defined(__APPLE__) && defined(__OBJC__)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#define DS41F_HAVE_METAL 1
#else
#define DS41F_HAVE_METAL 0
#endif

struct ds41f_prefill_native_context {
    ds41f_prefill_native_config cfg;
    ds41f_prefill_native_plan plan;
    int32_t *tokens;
    unsigned char *carry[2];
    ds41f_prefill_native_command *commands;
    uint32_t n_commands;
    uint32_t uploaded;
    uint32_t current;
    uint32_t executed_encode_chunks;
    uint32_t last_encoded_layer;
    uint32_t last_encoded_chunk;
};

struct ds41f_prefill_submission_context {
    ds41f_prefill_sweep_config cfg;
    ds41f_prefill_sweep_plan plan;
    unsigned char *buffers[DS41F_PREFILL_MAX_ALLOCATIONS];
    uint64_t buffer_sizes[DS41F_PREFILL_MAX_ALLOCATIONS];
    uint32_t buffer_touched[DS41F_PREFILL_MAX_ALLOCATIONS];
    uint64_t total_owned_bytes;
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
#if DS41F_HAVE_METAL
    id<MTLDevice> metal_device;
    id<MTLCommandQueue> metal_queue;
    id<MTLBuffer> metal_buffers[DS41F_PREFILL_MAX_ALLOCATIONS];
    id<MTLCommandBuffer> active_command_buffer;
    id<MTLBlitCommandEncoder> active_blit_encoder;
#endif
};

const char *ds41f_prefill_native_version(void) {
#if DS41F_HAVE_METAL
    return "ds41f-prefill-native-official-sparse-attn-v11";
#else
    return "ds41f-prefill-native-buffer-official-rotary-v7";
#endif
}

static uint32_t ds41f_min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }
static uint64_t ds41f_prefill_checksum64(const unsigned char *p, uint64_t n);

static uint32_t ds41f_prefill_limit(uint32_t ctx) {
    uint32_t limit = 8192u;
    if (ctx < 8192u) limit = 2048u;
    else if (ctx < 16384u) limit = 4096u;
    return ds41f_min_u32(ctx, limit);
}

static uint32_t ds41f_encoder_chunk_cap(uint32_t prefill_cap, uint32_t total_count) {
    if (total_count < 8192u && prefill_cap > 2048u) return 2048u;
    if (total_count < 16384u && prefill_cap > 4096u) return 4096u;
    return prefill_cap;
}

static uint32_t ds41f_prefill_count(uint32_t ctx, uint32_t remaining) {
    const uint32_t prefill_cap = ds41f_prefill_limit(ctx);
    if (remaining >= 4096u) {
        uint32_t count = ds41f_min_u32(remaining, prefill_cap);
        return count - (count % 2048u);
    }
    return ds41f_min_u32(remaining, ds41f_min_u32(prefill_cap, 2048u));
}

static void ds41f_copy_str(char *dst, size_t n, const char *src) {
    if (!dst || n == 0) return;
    if (!src) src = "";
    strncpy(dst, src, n - 1u);
    dst[n - 1u] = '\0';
}

static int ds41f_add_alloc(ds41f_prefill_sweep_plan *p,
                           const char *role,
                           uint64_t size_bytes,
                           uint32_t alignment,
                           const char *repr,
                           uint32_t first,
                           uint32_t last,
                           const char *owner,
                           const char *alias,
                           uint32_t persistence,
                           uint32_t survives_encoder,
                           uint32_t survives_deferred) {
    if (p->n_allocations >= DS41F_PREFILL_MAX_ALLOCATIONS) return -1;
    ds41f_prefill_sweep_allocation *a = &p->allocations[p->n_allocations++];
    ds41f_copy_str(a->semantic_role, sizeof(a->semantic_role), role);
    a->size_bytes = size_bytes;
    a->alignment = alignment;
    ds41f_copy_str(a->representation, sizeof(a->representation), repr);
    a->first_use_step = first;
    a->last_use_step = last;
    ds41f_copy_str(a->owner, sizeof(a->owner), owner);
    ds41f_copy_str(a->alias_reuse_class, sizeof(a->alias_reuse_class), alias);
    a->persistence_class = persistence;
    a->survives_encoder_only = survives_encoder;
    a->survives_deferred_decoder = survives_deferred;
    return 0;
}

static int ds41f_add_sweep_cmd(ds41f_prefill_sweep_plan *p,
                               uint32_t kind,
                               uint32_t layer,
                               uint32_t offset,
                               uint32_t rows,
                               uint32_t phase,
                               uint32_t checkpoint_valid) {
    if (p->n_commands >= DS41F_PREFILL_MAX_SWEEP_COMMANDS) return -1;
    ds41f_prefill_sweep_command *c = &p->commands[p->n_commands++];
    c->kind = kind;
    c->layer = layer;
    c->offset = offset;
    c->rows = rows;
    c->phase = phase;
    c->checkpoint_valid = checkpoint_valid;
    return 0;
}

int ds41f_prefill_native_build_sweep_plan(const ds41f_prefill_sweep_config *cfg,
                                          ds41f_prefill_sweep_plan *out) {
    if (!cfg || !out) return -1;
    if (cfg->ctx == 0 || cfg->remaining == 0) return -2;
    if (cfg->dim == 0 || cfg->hc_mult == 0 || cfg->vocab_size == 0) return -3;
    if ((cfg->encoder_only || cfg->resume_encoder) && cfg->remaining < 8192u) return -4;
    memset(out, 0, sizeof(*out));
    const uint32_t bps = cfg->bytes_per_hidden_scalar ? cfg->bytes_per_hidden_scalar : 2u;
    const uint32_t count = ds41f_prefill_count(cfg->ctx, cfg->remaining);
    const uint32_t prefill_cap = ds41f_prefill_limit(cfg->ctx);
    const uint32_t enc_chunk = ds41f_encoder_chunk_cap(prefill_cap, count);
    const uint32_t wide = count > enc_chunk;
    const uint32_t decoder_suffix = wide && count >= 8192u;
    if ((cfg->encoder_only || cfg->resume_encoder) && !decoder_suffix) return -5;
    out->count = count;
    out->prefill_cap = prefill_cap;
    out->encoder_chunk = enc_chunk;
    out->wide = wide;
    out->decoder_suffix = decoder_suffix;
    out->encoder_only = cfg->encoder_only ? 1u : 0u;
    out->resume_encoder = cfg->resume_encoder ? 1u : 0u;
    out->defer_decoder_candidate = (!cfg->encoder_resident && count >= 8192u && cfg->remaining - count >= 8192u) ? 1u : 0u;
    out->checkpoint_valid_during_sweep = 0u;
    out->checkpoint_valid_after_sweep = cfg->encoder_only ? 0u : 1u;

    const uint64_t hc_dim = (uint64_t)cfg->dim * (uint64_t)cfg->hc_mult;
    const uint64_t full_hidden = (uint64_t)count * hc_dim * (uint64_t)bps;
    const uint32_t final_step_hint = 500u;
    if (ds41f_add_alloc(out, "batch_cur_hc", full_hidden, 256u, "BF16 full-prompt HC", 1u, final_step_hint, "native sweep arena", "hc_pingpong", DS41F_PERSIST_SWEEP, 1u, 1u) != 0) return -10;
    if (ds41f_add_alloc(out, "batch_next_hc", full_hidden, 256u, "BF16 full-prompt HC", 2u, final_step_hint, "native sweep arena", "hc_pingpong", DS41F_PERSIST_LAYER, 0u, 0u) != 0) return -10;
    if (ds41f_add_alloc(out, "carry.residual", full_hidden, 256u, "BF16 compactable", 1u, final_step_hint, "DS41_CARRY_ROWS", "structured_carry", DS41F_PERSIST_SWEEP, 1u, 1u) != 0) return -10;
    if (ds41f_add_alloc(out, "carry.pre", (uint64_t)count * cfg->hc_mult * 4ull, 64u, "F32", 1u, final_step_hint, "DS41_CARRY_ROWS", "structured_carry", DS41F_PERSIST_SWEEP, 1u, 1u) != 0) return -10;
    if (ds41f_add_alloc(out, "carry.ffn_split", (uint64_t)count * 24ull * 4ull, 64u, "F32", 1u, final_step_hint, "DS41_CARRY_ROWS", "structured_carry", DS41F_PERSIST_LAYER, 0u, 0u) != 0) return -10;
    if (ds41f_add_alloc(out, "carry.selected_comp", (uint64_t)count * 8ull * 4ull, 64u, "F32 index_topk placeholder", 1u, final_step_hint, "DS41_CARRY_ROWS", "structured_carry", DS41F_PERSIST_LAYER, 0u, 0u) != 0) return -10;
    if (ds41f_add_alloc(out, "carry.block_mask", (uint64_t)count * (((uint64_t)cfg->ctx + 7ull) / 8ull), 64u, "MASK compact", 1u, final_step_hint, "DS41_CARRY_ROWS", "structured_carry", DS41F_PERSIST_DEFERRED_DECODER, 1u, 1u) != 0) return -10;
    if (decoder_suffix) {
        uint32_t max_needed = 1u + (39u - 20u) * 127u;
        if (ds41f_add_alloc(out, "decoder_suffix_rows", (uint64_t)max_needed * hc_dim * (uint64_t)bps, 256u, "BF16 suffix reconstruction", 1u, final_step_hint, "deferred decoder", "decoder_suffix", DS41F_PERSIST_DEFERRED_DECODER, 1u, 1u) != 0) return -10;
    }
    if (ds41f_add_alloc(out, "stage_scratch", (uint64_t)enc_chunk * hc_dim * (uint64_t)bps, 256u, "reusable BF16 scratch", 1u, final_step_hint, "command encoder", "stage_local_reuse", DS41F_PERSIST_STAGE_LOCAL, 0u, 0u) != 0) return -10;
    if (ds41f_add_alloc(out, "final_logits", (uint64_t)enc_chunk * cfg->vocab_size * 4ull, 256u, "F32 logits", 1u, final_step_hint, "output head", "final_output", DS41F_PERSIST_STAGE_LOCAL, 0u, 0u) != 0) return -10;

    if (ds41f_add_sweep_cmd(out, DS41F_SWEEP_BEGIN_INVALIDATE, UINT32_MAX, 0u, count, 0u, 0u) != 0) return -20;
    out->n_checkpoint_transitions++;
    if (ds41f_add_sweep_cmd(out, DS41F_SWEEP_PREFETCH_ENGRAM0, UINT32_MAX, 0u, count, 0u, 0u) != 0) return -20;
    out->n_prefetch_events++;
    for (uint32_t layer = 0; layer < 40u; layer++) {
        if (cfg->encoder_only && layer == 20u) {
            if (ds41f_add_sweep_cmd(out, DS41F_SWEEP_ENCODER_ONLY_COMPLETE_INVALID, layer, 0u, count, 1u, 0u) != 0) return -20;
            out->n_checkpoint_transitions++;
            break;
        }
        uint32_t first = 0u;
        uint32_t rows = count;
        uint32_t phase = layer < 20u ? 1u : 2u;
        uint32_t chunk = enc_chunk;
        if (decoder_suffix && layer >= 20u) {
            if (layer == 20u) {
                if (ds41f_add_sweep_cmd(out, DS41F_SWEEP_DECODER_PREPARE_SUFFIX, layer, 0u, count, 2u, 0u) != 0) return -20;
            }
            uint32_t needed = 1u + (39u - layer) * 127u;
            first = count - needed;
            rows = needed;
            phase = 3u;
            chunk = (wide && enc_chunk > 2048u) ? 2048u : enc_chunk;
            if (first >= 127u) {
                if (ds41f_add_sweep_cmd(out, DS41F_SWEEP_DECODER_PREPARE_SUFFIX, layer, first - 127u, 127u, 3u, 0u) != 0) return -20;
            }
            out->decoder_suffix_row_layer_work += rows;
        } else if (layer < 20u) {
            out->encoder_row_layer_work += rows;
        }
        if (layer == 2u) {
            if (ds41f_add_sweep_cmd(out, DS41F_SWEEP_PREFETCH_ENGRAM1, UINT32_MAX, 0u, count, phase, 0u) != 0) return -20;
            out->n_prefetch_events++;
        }
        if (layer + 1u < 40u) {
            if (ds41f_add_sweep_cmd(out, DS41F_SWEEP_SSD_READ_AHEAD, layer + 1u, 0u, count, phase, 0u) != 0) return -20;
            out->n_prefetch_events++;
        }
        out->total_row_layer_work += rows;
        if (ds41f_add_sweep_cmd(out, DS41F_SWEEP_BEGIN_LAYER, layer, first, rows, phase, 0u) != 0) return -20;
        uint32_t off = first;
        const uint32_t end = first + rows;
        while (off < end) {
            uint32_t n = ds41f_min_u32(chunk, end - off);
            if (ds41f_add_sweep_cmd(out, DS41F_SWEEP_ENCODE_ROWS, layer, off, n, phase, 0u) != 0) return -20;
            off += n;
        }
        if (ds41f_add_sweep_cmd(out, DS41F_SWEEP_SWAP_HC_LAYER, layer, first, rows, phase, 0u) != 0) return -20;
        if (ds41f_add_sweep_cmd(out, DS41F_SWEEP_PUBLISH_FRONTIER, layer, first, rows, phase, 0u) != 0) return -20;
        if (ds41f_add_sweep_cmd(out, DS41F_SWEEP_END_LAYER, layer, first, rows, phase, 0u) != 0) return -20;
    }
    if (out->defer_decoder_candidate) {
        if (ds41f_add_sweep_cmd(out, DS41F_SWEEP_DECODER_PENDING_INVALID, UINT32_MAX, 0u, count, 4u, 0u) != 0) return -20;
        out->n_checkpoint_transitions++;
    }
    if (!cfg->encoder_only) {
        if (ds41f_add_sweep_cmd(out, DS41F_SWEEP_ENCODE_OUTPUT_HEAD, UINT32_MAX, count - 1u, 1u, 5u, 0u) != 0) return -20;
        if (ds41f_add_sweep_cmd(out, DS41F_SWEEP_READ_LOGITS, UINT32_MAX, count - 1u, 1u, 5u, 0u) != 0) return -20;
        if (ds41f_add_sweep_cmd(out, DS41F_SWEEP_CHECKPOINT_MAY_COMMIT, UINT32_MAX, count - 1u, 1u, 5u, 1u) != 0) return -20;
        out->n_checkpoint_transitions++;
    }
    const uint32_t last_step = out->n_commands ? out->n_commands - 1u : 0u;
    for (uint32_t i = 0; i < out->n_allocations; i++) {
        if (out->allocations[i].last_use_step > last_step) out->allocations[i].last_use_step = last_step;
    }
    return 0;
}

static int ds41f_submission_find_role(const ds41f_prefill_submission_context *ctx, const char *role) {
    if (!ctx || !role) return -1;
    for (uint32_t i = 0; i < ctx->plan.n_allocations; i++) {
        if (strncmp(ctx->plan.allocations[i].semantic_role, role, sizeof(ctx->plan.allocations[i].semantic_role)) == 0) return (int)i;
    }
    return -1;
}

#if DS41F_HAVE_METAL
static int ds41f_submission_begin_metal_batch(ds41f_prefill_submission_context *ctx) {
    if (!ctx || !ctx->metal_enabled) return 0;
    if (ctx->active_command_buffer || ctx->active_blit_encoder) return -1;
    ctx->active_command_buffer = [ctx->metal_queue commandBuffer];
    if (!ctx->active_command_buffer) return -2;
    ctx->active_blit_encoder = [ctx->active_command_buffer blitCommandEncoder];
    if (!ctx->active_blit_encoder) return -3;
    return 0;
}

static int ds41f_submission_end_metal_batch(ds41f_prefill_submission_context *ctx,
                                            const ds41f_prefill_sweep_command *cmd,
                                            uint32_t category) {
    if (!ctx || !ctx->metal_enabled) return 0;
    if (!ctx->active_command_buffer || !ctx->active_blit_encoder) return -1;
    [ctx->active_blit_encoder endEncoding];
    [ctx->active_command_buffer commit];
    [ctx->active_command_buffer waitUntilCompleted];
    ctx->active_blit_encoder = nil;
    ctx->active_command_buffer = nil;
    ctx->metal_command_buffers_committed++;
    ctx->metal_blit_encoders_committed++;
    ctx->metal_completion_waits++;
    if (cmd) {
        if (cmd->phase < 8u) ctx->command_buffers_by_phase[cmd->phase]++;
        if (cmd->layer < DS41F_PREFILL_MAX_LAYERS) ctx->command_buffers_by_layer[cmd->layer]++;
    }
    if (category == DS41F_SWEEP_ENCODE_ROWS) ctx->command_buffers_encode_rows++;
    else if (category == DS41F_SWEEP_DECODER_PREPARE_SUFFIX) ctx->command_buffers_decoder_prepare++;
    else if (category == DS41F_SWEEP_SWAP_HC_LAYER) ctx->command_buffers_swap_hc++;
    else if (category == DS41F_SWEEP_ENCODE_OUTPUT_HEAD) ctx->command_buffers_output++;
    return 0;
}
#endif

static void ds41f_submission_record_cpu_batch(ds41f_prefill_submission_context *ctx,
                                              const ds41f_prefill_sweep_command *cmd,
                                              uint32_t category) {
    if (!ctx || ctx->metal_enabled) return;
    ctx->submitted_command_buffers++;
    if (cmd) {
        if (cmd->phase < 8u) ctx->command_buffers_by_phase[cmd->phase]++;
        if (cmd->layer < DS41F_PREFILL_MAX_LAYERS) ctx->command_buffers_by_layer[cmd->layer]++;
    }
    if (category == DS41F_SWEEP_ENCODE_ROWS) ctx->command_buffers_encode_rows++;
    else if (category == DS41F_SWEEP_DECODER_PREPARE_SUFFIX) ctx->command_buffers_decoder_prepare++;
    else if (category == DS41F_SWEEP_SWAP_HC_LAYER) ctx->command_buffers_swap_hc++;
    else if (category == DS41F_SWEEP_ENCODE_OUTPUT_HEAD) ctx->command_buffers_output++;
}

static void ds41f_submission_touch(ds41f_prefill_submission_context *ctx,
                                   int slot,
                                   uint32_t command_index,
                                   uint32_t salt) {
    if (!ctx || slot < 0 || (uint32_t)slot >= ctx->plan.n_allocations) return;
    unsigned char *p = ctx->buffers[slot];
    const uint64_t n = ctx->buffer_sizes[slot];
    if (!p || n == 0) return;
    uint64_t seed = 0x4d544c425546ull; /* MTLBUF */
    seed ^= ((uint64_t)command_index + 1ull) * 0x9e3779b185ebca87ull;
    seed ^= ((uint64_t)salt + 1ull) * 0xc2b2ae3d27d4eb4full;
    const uint64_t limit = n < 64ull ? n : 64ull;
#if DS41F_HAVE_METAL
    if (ctx->metal_enabled && ctx->active_blit_encoder && ctx->metal_buffers[slot]) {
        uint8_t value = (uint8_t)(((seed >> 7) ^ (seed >> 19) ^ seed) & 0xffu);
        [ctx->active_blit_encoder fillBuffer:ctx->metal_buffers[slot] range:NSMakeRange(0, (NSUInteger)limit) value:value];
        ctx->buffer_touched[slot] = 1u;
        ctx->encoded_buffer_ops++;
        return;
    }
#endif
    for (uint64_t i = 0; i < limit; i++) {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        p[i] = (unsigned char)((seed * 2685821657736338717ull) >> 56);
    }
    ctx->buffer_touched[slot] = 1u;
    ctx->encoded_buffer_ops++;
}

int ds41f_prefill_submission_context_create(const ds41f_prefill_sweep_config *cfg,
                                            ds41f_prefill_submission_context **out) {
    if (!cfg || !out) return -1;
    *out = NULL;
    ds41f_prefill_submission_context *ctx = (ds41f_prefill_submission_context *)calloc(1, sizeof(*ctx));
    if (!ctx) return -10;
    ctx->cfg = *cfg;
    int rc = ds41f_prefill_native_build_sweep_plan(cfg, &ctx->plan);
    if (rc != 0) {
        free(ctx);
        return rc;
    }
    uint64_t total = 0;
    for (uint32_t i = 0; i < ctx->plan.n_allocations; i++) {
        const uint64_t n = ctx->plan.allocations[i].size_bytes;
        if (UINT64_MAX - total < n) {
            ds41f_prefill_submission_context_destroy(ctx);
            return -11;
        }
        total += n;
    }
    if (cfg->memory_budget_bytes && total > cfg->memory_budget_bytes) {
        ds41f_prefill_submission_context_destroy(ctx);
        return -12;
    }
    ctx->total_owned_bytes = total;
#if DS41F_HAVE_METAL
    @autoreleasepool {
        ctx->metal_device = MTLCreateSystemDefaultDevice();
        if (!ctx->metal_device) {
            ds41f_prefill_submission_context_destroy(ctx);
            return -14;
        }
        ctx->metal_queue = [ctx->metal_device newCommandQueue];
        if (!ctx->metal_queue) {
            ds41f_prefill_submission_context_destroy(ctx);
            return -15;
        }
        ctx->metal_enabled = 1u;
    }
#endif
    for (uint32_t i = 0; i < ctx->plan.n_allocations; i++) {
        const uint64_t n = ctx->plan.allocations[i].size_bytes;
        ctx->buffer_sizes[i] = n;
#if DS41F_HAVE_METAL
        if (ctx->metal_enabled) {
            @autoreleasepool {
                ctx->metal_buffers[i] = [ctx->metal_device newBufferWithLength:(NSUInteger)(n ? n : 1ull)
                                                                       options:MTLResourceStorageModeShared];
            }
            if (!ctx->metal_buffers[i]) {
                ds41f_prefill_submission_context_destroy(ctx);
                return -16;
            }
            ctx->buffers[i] = (unsigned char *)[ctx->metal_buffers[i] contents];
            ctx->metal_buffers_allocated++;
        } else
#endif
        {
            ctx->buffers[i] = (unsigned char *)calloc((size_t)n ? (size_t)n : 1u, 1u);
            if (!ctx->buffers[i]) {
                ds41f_prefill_submission_context_destroy(ctx);
                return -13;
            }
        }
    }
    ctx->failed_command_index = UINT32_MAX;
    *out = ctx;
    return 0;
}

void ds41f_prefill_submission_context_destroy(ds41f_prefill_submission_context *ctx) {
    if (!ctx) return;
#if DS41F_HAVE_METAL
    if (ctx->metal_enabled) {
        for (uint32_t i = 0; i < ctx->plan.n_allocations; i++) {
            [ctx->metal_buffers[i] release];
        }
        [ctx->metal_queue release];
        [ctx->metal_device release];
    } else
#endif
    {
        for (uint32_t i = 0; i < ctx->plan.n_allocations; i++) free(ctx->buffers[i]);
    }
    free(ctx);
}

int ds41f_prefill_submission_context_info(const ds41f_prefill_submission_context *ctx,
                                          ds41f_prefill_submission_info *out) {
    if (!ctx || !out) return -1;
    memset(out, 0, sizeof(*out));
    out->total_owned_bytes = ctx->total_owned_bytes;
    out->memory_budget_bytes = ctx->cfg.memory_budget_bytes;
    out->n_buffers = ctx->plan.n_allocations;
    out->submitted_commands = ctx->submitted_commands;
    out->submitted_command_buffers = ctx->submitted_command_buffers;
    out->encoded_buffer_ops = ctx->encoded_buffer_ops;
    out->checkpoint_valid = ctx->checkpoint_valid;
    out->current_hc_slot = ctx->current_hc_slot;
    out->failed_command_index = ctx->failed_command_index;
    out->metal_enabled = ctx->metal_enabled;
    out->metal_buffers_allocated = ctx->metal_buffers_allocated;
    out->metal_command_buffers_committed = ctx->metal_command_buffers_committed;
    out->metal_blit_encoders_committed = ctx->metal_blit_encoders_committed;
    out->metal_completion_waits = ctx->metal_completion_waits;
    out->command_buffers_encode_rows = ctx->command_buffers_encode_rows;
    out->command_buffers_decoder_prepare = ctx->command_buffers_decoder_prepare;
    out->command_buffers_swap_hc = ctx->command_buffers_swap_hc;
    out->command_buffers_output = ctx->command_buffers_output;
    memcpy(out->command_buffers_by_phase, ctx->command_buffers_by_phase, sizeof(out->command_buffers_by_phase));
    memcpy(out->command_buffers_by_layer, ctx->command_buffers_by_layer, sizeof(out->command_buffers_by_layer));
    for (uint32_t i = 0; i < ctx->plan.n_allocations; i++) {
        ds41f_copy_str(out->buffers[i].semantic_role, sizeof(out->buffers[i].semantic_role), ctx->plan.allocations[i].semantic_role);
        out->buffers[i].size_bytes = ctx->buffer_sizes[i];
        out->buffers[i].checksum = ds41f_prefill_checksum64(ctx->buffers[i], ctx->buffer_sizes[i]);
        out->buffers[i].allocated = ctx->buffers[i] != NULL;
        out->buffers[i].touched = ctx->buffer_touched[i];
        out->buffers[i].persistence_class = ctx->plan.allocations[i].persistence_class;
    }
    return 0;
}

int ds41f_prefill_submission_submit(ds41f_prefill_submission_context *ctx,
                                    uint32_t *submitted_commands) {
    if (!ctx) return -1;
    const int slot_cur = ds41f_submission_find_role(ctx, "batch_cur_hc");
    const int slot_next = ds41f_submission_find_role(ctx, "batch_next_hc");
    const int slot_residual = ds41f_submission_find_role(ctx, "carry.residual");
    const int slot_pre = ds41f_submission_find_role(ctx, "carry.pre");
    const int slot_ffn = ds41f_submission_find_role(ctx, "carry.ffn_split");
    const int slot_comp = ds41f_submission_find_role(ctx, "carry.selected_comp");
    const int slot_mask = ds41f_submission_find_role(ctx, "carry.block_mask");
    const int slot_suffix = ds41f_submission_find_role(ctx, "decoder_suffix_rows");
    const int slot_scratch = ds41f_submission_find_role(ctx, "stage_scratch");
    const int slot_logits = ds41f_submission_find_role(ctx, "final_logits");
    ctx->submitted_commands = 0;
    ctx->submitted_command_buffers = 0;
    ctx->encoded_buffer_ops = 0;
    ctx->metal_command_buffers_committed = 0;
    ctx->metal_blit_encoders_committed = 0;
    ctx->metal_completion_waits = 0;
    ctx->command_buffers_encode_rows = 0;
    ctx->command_buffers_decoder_prepare = 0;
    ctx->command_buffers_swap_hc = 0;
    ctx->command_buffers_output = 0;
    memset(ctx->command_buffers_by_phase, 0, sizeof(ctx->command_buffers_by_phase));
    memset(ctx->command_buffers_by_layer, 0, sizeof(ctx->command_buffers_by_layer));
    ctx->checkpoint_valid = 0;
    ctx->current_hc_slot = 0;
    ctx->failed_command_index = UINT32_MAX;
    memset(ctx->buffer_touched, 0, sizeof(ctx->buffer_touched));
    for (uint32_t i = 0; i < ctx->plan.n_commands; i++) {
        const ds41f_prefill_sweep_command *cmd = &ctx->plan.commands[i];
        switch (cmd->kind) {
            case DS41F_SWEEP_BEGIN_INVALIDATE:
                ctx->checkpoint_valid = 0;
                break;
            case DS41F_SWEEP_BEGIN_LAYER:
                break;
            case DS41F_SWEEP_ENCODE_ROWS:
#if DS41F_HAVE_METAL
                if (ctx->metal_enabled && ds41f_submission_begin_metal_batch(ctx) != 0) { ctx->failed_command_index = i; return -2; }
#endif
                ds41f_submission_touch(ctx, slot_next, i, cmd->layer ^ cmd->offset ^ cmd->rows);
                ds41f_submission_touch(ctx, slot_residual, i, cmd->layer);
                ds41f_submission_touch(ctx, slot_pre, i, cmd->rows);
                ds41f_submission_touch(ctx, slot_ffn, i, cmd->offset);
                ds41f_submission_touch(ctx, slot_comp, i, cmd->phase);
                ds41f_submission_touch(ctx, slot_mask, i, cmd->layer + cmd->rows);
                ds41f_submission_touch(ctx, slot_scratch, i, cmd->offset + cmd->rows);
#if DS41F_HAVE_METAL
                if (ctx->metal_enabled && ds41f_submission_end_metal_batch(ctx, cmd, DS41F_SWEEP_ENCODE_ROWS) != 0) { ctx->failed_command_index = i; return -3; }
#endif
                ds41f_submission_record_cpu_batch(ctx, cmd, DS41F_SWEEP_ENCODE_ROWS);
                break;
            case DS41F_SWEEP_DECODER_PREPARE_SUFFIX:
#if DS41F_HAVE_METAL
                if (ctx->metal_enabled && ds41f_submission_begin_metal_batch(ctx) != 0) { ctx->failed_command_index = i; return -2; }
#endif
                ds41f_submission_touch(ctx, slot_suffix, i, cmd->offset ^ cmd->rows);
#if DS41F_HAVE_METAL
                if (ctx->metal_enabled && ds41f_submission_end_metal_batch(ctx, cmd, DS41F_SWEEP_DECODER_PREPARE_SUFFIX) != 0) { ctx->failed_command_index = i; return -3; }
#endif
                ds41f_submission_record_cpu_batch(ctx, cmd, DS41F_SWEEP_DECODER_PREPARE_SUFFIX);
                break;
            case DS41F_SWEEP_SWAP_HC_LAYER:
                ctx->current_hc_slot = 1u - ctx->current_hc_slot;
                if (slot_cur >= 0) ctx->buffer_touched[slot_cur] = 1u;
                if (slot_next >= 0) ctx->buffer_touched[slot_next] = 1u;
                break;
            case DS41F_SWEEP_ENCODE_OUTPUT_HEAD:
#if DS41F_HAVE_METAL
                if (ctx->metal_enabled && ds41f_submission_begin_metal_batch(ctx) != 0) { ctx->failed_command_index = i; return -2; }
#endif
                ds41f_submission_touch(ctx, slot_logits, i, cmd->rows);
                break;
            case DS41F_SWEEP_READ_LOGITS:
                ds41f_submission_touch(ctx, slot_logits, i, cmd->offset);
#if DS41F_HAVE_METAL
                if (ctx->metal_enabled && ds41f_submission_end_metal_batch(ctx, cmd, DS41F_SWEEP_ENCODE_OUTPUT_HEAD) != 0) { ctx->failed_command_index = i; return -3; }
#endif
                ds41f_submission_record_cpu_batch(ctx, cmd, DS41F_SWEEP_ENCODE_OUTPUT_HEAD);
                break;
            case DS41F_SWEEP_CHECKPOINT_MAY_COMMIT:
                ctx->checkpoint_valid = 1u;
                break;
            case DS41F_SWEEP_ENCODER_ONLY_COMPLETE_INVALID:
            case DS41F_SWEEP_DECODER_PENDING_INVALID:
                ctx->checkpoint_valid = 0;
                break;
            default:
                break;
        }
        ctx->submitted_commands++;
    }
    if (ctx->metal_enabled) ctx->submitted_command_buffers = ctx->metal_command_buffers_committed;
    if (submitted_commands) *submitted_commands = ctx->submitted_commands;
    return 0;
}

int ds41f_official_embedding_gather_bf16(const uint16_t *embedding_bf16,
                                         uint32_t vocab_rows,
                                         uint32_t dim,
                                         const int32_t *tokens,
                                         uint32_t n_tokens,
                                         uint16_t *out_bf16,
                                         ds41f_official_embedding_result *result) {
    if (!embedding_bf16 || !tokens || !out_bf16 || !result) return -1;
    if (vocab_rows == 0 || dim == 0 || n_tokens == 0) return -2;
    memset(result, 0, sizeof(*result));
    result->n_tokens = n_tokens;
    result->dim = dim;
    result->vocab_rows = vocab_rows;
    const uint64_t weight_elems = (uint64_t)vocab_rows * (uint64_t)dim;
    const uint64_t out_elems = (uint64_t)n_tokens * (uint64_t)dim;
    if (weight_elems > SIZE_MAX / sizeof(uint16_t) || out_elems > SIZE_MAX / sizeof(uint16_t)) return -3;
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= vocab_rows) return -4;
    }
#if DS41F_HAVE_METAL
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return -10;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) { [device release]; return -11; }
        NSString *src = @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "kernel void ds41f_embed_bf16_gather(device const ushort *w [[buffer(0)]],\n"
            "                                    device const int *tok [[buffer(1)]],\n"
            "                                    device ushort *out [[buffer(2)]],\n"
            "                                    constant uint &dim [[buffer(3)]],\n"
            "                                    uint gid [[thread_position_in_grid]]) {\n"
            "  uint t = gid / dim; uint c = gid - t * dim; int id = tok[t];\n"
            "  out[gid] = w[(uint)id * dim + c];\n"
            "}\n";
        NSError *err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
        if (!lib) { [queue release]; [device release]; return -12; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"ds41f_embed_bf16_gather"];
        if (!fn) { [lib release]; [queue release]; [device release]; return -13; }
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { [fn release]; [lib release]; [queue release]; [device release]; return -14; }
        id<MTLBuffer> wb = [device newBufferWithBytes:embedding_bf16 length:(NSUInteger)(weight_elems * sizeof(uint16_t)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> tb = [device newBufferWithBytes:tokens length:(NSUInteger)n_tokens * sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> ob = [device newBufferWithLength:(NSUInteger)(out_elems * sizeof(uint16_t)) options:MTLResourceStorageModeShared];
        if (!wb || !tb || !ob) {
            [ob release]; [tb release]; [wb release]; [pso release]; [fn release]; [lib release]; [queue release]; [device release];
            return -15;
        }
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:wb offset:0 atIndex:0];
        [enc setBuffer:tb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBytes:&dim length:sizeof(dim) atIndex:3];
        const NSUInteger w = pso.threadExecutionWidth ? pso.threadExecutionWidth : 32u;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)out_elems, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            [ob release]; [tb release]; [wb release]; [pso release]; [fn release]; [lib release]; [queue release]; [device release];
            return -16;
        }
        memcpy(out_bf16, [ob contents], (size_t)(out_elems * sizeof(uint16_t)));
        result->metal_enabled = 1u;
        result->metal_command_buffers = 1u;
        result->metal_compute_encoders = 1u;
        result->metal_completion_waits = 1u;
        [ob release]; [tb release]; [wb release]; [pso release]; [fn release]; [lib release]; [queue release]; [device release];
    }
#else
    for (uint32_t t = 0; t < n_tokens; t++) {
        memcpy(out_bf16 + (uint64_t)t * dim,
               embedding_bf16 + (uint64_t)(uint32_t)tokens[t] * dim,
               (size_t)dim * sizeof(uint16_t));
    }
#endif
    result->output_checksum = ds41f_prefill_checksum64((const unsigned char *)out_bf16, out_elems * sizeof(uint16_t));
    return 0;
}

int ds41f_official_bf16_linear_f32(const uint16_t *input_bf16,
                                   const uint16_t *weight_bf16,
                                   uint32_t rows,
                                   uint32_t in_dim,
                                   uint32_t out_dim,
                                   float *out_f32,
                                   ds41f_official_linear_result *result) {
    if (!input_bf16 || !weight_bf16 || !out_f32 || !result) return -1;
    if (rows == 0 || in_dim == 0 || out_dim == 0) return -2;
    memset(result, 0, sizeof(*result));
    result->rows = rows;
    result->in_dim = in_dim;
    result->out_dim = out_dim;
    const uint64_t input_elems = (uint64_t)rows * (uint64_t)in_dim;
    const uint64_t weight_elems = (uint64_t)out_dim * (uint64_t)in_dim;
    const uint64_t out_elems = (uint64_t)rows * (uint64_t)out_dim;
    if (input_elems > SIZE_MAX / sizeof(uint16_t) || weight_elems > SIZE_MAX / sizeof(uint16_t) || out_elems > SIZE_MAX / sizeof(float)) return -3;
#if DS41F_HAVE_METAL
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return -10;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) { [device release]; return -11; }
        NSString *src = @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "static inline float ds41f_bf16_to_f32(ushort x) { uint u = ((uint)x) << 16; return as_type<float>(u); }\n"
            "kernel void ds41f_bf16_linear_f32(device const ushort *inp [[buffer(0)]],\n"
            "                                  device const ushort *w [[buffer(1)]],\n"
            "                                  device float *out [[buffer(2)]],\n"
            "                                  constant uint &in_dim [[buffer(3)]],\n"
            "                                  constant uint &out_dim [[buffer(4)]],\n"
            "                                  uint gid [[thread_position_in_grid]]) {\n"
            "  uint r = gid / out_dim; uint o = gid - r * out_dim; float acc = 0.0f;\n"
            "  for (uint k = 0; k < in_dim; k++) acc += ds41f_bf16_to_f32(inp[r * in_dim + k]) * ds41f_bf16_to_f32(w[o * in_dim + k]);\n"
            "  out[gid] = acc;\n"
            "}\n";
        NSError *err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
        if (!lib) { [queue release]; [device release]; return -12; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"ds41f_bf16_linear_f32"];
        if (!fn) { [lib release]; [queue release]; [device release]; return -13; }
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { [fn release]; [lib release]; [queue release]; [device release]; return -14; }
        id<MTLBuffer> ib = [device newBufferWithBytes:input_bf16 length:(NSUInteger)(input_elems * sizeof(uint16_t)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> wb = [device newBufferWithBytes:weight_bf16 length:(NSUInteger)(weight_elems * sizeof(uint16_t)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> ob = [device newBufferWithLength:(NSUInteger)(out_elems * sizeof(float)) options:MTLResourceStorageModeShared];
        if (!ib || !wb || !ob) {
            [ob release]; [wb release]; [ib release]; [pso release]; [fn release]; [lib release]; [queue release]; [device release];
            return -15;
        }
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:ib offset:0 atIndex:0];
        [enc setBuffer:wb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBytes:&in_dim length:sizeof(in_dim) atIndex:3];
        [enc setBytes:&out_dim length:sizeof(out_dim) atIndex:4];
        const NSUInteger tw = pso.threadExecutionWidth ? pso.threadExecutionWidth : 32u;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)out_elems, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            [ob release]; [wb release]; [ib release]; [pso release]; [fn release]; [lib release]; [queue release]; [device release];
            return -16;
        }
        memcpy(out_f32, [ob contents], (size_t)(out_elems * sizeof(float)));
        result->metal_enabled = 1u;
        result->metal_command_buffers = 1u;
        result->metal_compute_encoders = 1u;
        result->metal_completion_waits = 1u;
        [ob release]; [wb release]; [ib release]; [pso release]; [fn release]; [lib release]; [queue release]; [device release];
    }
#else
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < in_dim; k++) {
                uint32_t ix = ((uint32_t)input_bf16[(uint64_t)r * in_dim + k]) << 16;
                uint32_t wx = ((uint32_t)weight_bf16[(uint64_t)o * in_dim + k]) << 16;
                float a, b;
                memcpy(&a, &ix, sizeof(float));
                memcpy(&b, &wx, sizeof(float));
                acc += a * b;
            }
            out_f32[(uint64_t)r * out_dim + o] = acc;
        }
    }
#endif
    result->output_checksum = ds41f_prefill_checksum64((const unsigned char *)out_f32, out_elems * sizeof(float));
    return 0;
}

int ds41f_official_f32_linear_f32(const float *input_f32,
                                  const float *weight_f32,
                                  uint32_t rows,
                                  uint32_t in_dim,
                                  uint32_t out_dim,
                                  float *out_f32,
                                  ds41f_official_linear_result *result) {
    if (!input_f32 || !weight_f32 || !out_f32 || !result) return -1;
    if (rows == 0 || in_dim == 0 || out_dim == 0) return -2;
    memset(result, 0, sizeof(*result));
    result->rows = rows;
    result->in_dim = in_dim;
    result->out_dim = out_dim;
    const uint64_t input_elems = (uint64_t)rows * (uint64_t)in_dim;
    const uint64_t weight_elems = (uint64_t)out_dim * (uint64_t)in_dim;
    const uint64_t out_elems = (uint64_t)rows * (uint64_t)out_dim;
    if (input_elems > SIZE_MAX / sizeof(float) || weight_elems > SIZE_MAX / sizeof(float) || out_elems > SIZE_MAX / sizeof(float)) return -3;
#if DS41F_HAVE_METAL
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return -10;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) { [device release]; return -11; }
        NSString *src = @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "kernel void ds41f_f32_linear_f32(device const float *inp [[buffer(0)]],\n"
            "                                 device const float *w [[buffer(1)]],\n"
            "                                 device float *out [[buffer(2)]],\n"
            "                                 constant uint &in_dim [[buffer(3)]],\n"
            "                                 constant uint &out_dim [[buffer(4)]],\n"
            "                                 uint gid [[thread_position_in_grid]]) {\n"
            "  uint r = gid / out_dim; uint o = gid - r * out_dim; float acc = 0.0f;\n"
            "  for (uint k = 0; k < in_dim; k++) acc += inp[r * in_dim + k] * w[o * in_dim + k];\n"
            "  out[gid] = acc;\n"
            "}\n";
        NSError *err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
        if (!lib) { [queue release]; [device release]; return -12; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"ds41f_f32_linear_f32"];
        if (!fn) { [lib release]; [queue release]; [device release]; return -13; }
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { [fn release]; [lib release]; [queue release]; [device release]; return -14; }
        id<MTLBuffer> ib = [device newBufferWithBytes:input_f32 length:(NSUInteger)(input_elems * sizeof(float)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> wb = [device newBufferWithBytes:weight_f32 length:(NSUInteger)(weight_elems * sizeof(float)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> ob = [device newBufferWithLength:(NSUInteger)(out_elems * sizeof(float)) options:MTLResourceStorageModeShared];
        if (!ib || !wb || !ob) {
            [ob release]; [wb release]; [ib release]; [pso release]; [fn release]; [lib release]; [queue release]; [device release];
            return -15;
        }
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:ib offset:0 atIndex:0];
        [enc setBuffer:wb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBytes:&in_dim length:sizeof(in_dim) atIndex:3];
        [enc setBytes:&out_dim length:sizeof(out_dim) atIndex:4];
        const NSUInteger tw = pso.threadExecutionWidth ? pso.threadExecutionWidth : 32u;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)out_elems, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            [ob release]; [wb release]; [ib release]; [pso release]; [fn release]; [lib release]; [queue release]; [device release];
            return -16;
        }
        memcpy(out_f32, [ob contents], (size_t)(out_elems * sizeof(float)));
        result->metal_enabled = 1u;
        result->metal_command_buffers = 1u;
        result->metal_compute_encoders = 1u;
        result->metal_completion_waits = 1u;
        [ob release]; [wb release]; [ib release]; [pso release]; [fn release]; [lib release]; [queue release]; [device release];
    }
#else
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < in_dim; k++) acc += input_f32[(uint64_t)r * in_dim + k] * weight_f32[(uint64_t)o * in_dim + k];
            out_f32[(uint64_t)r * out_dim + o] = acc;
        }
    }
#endif
    result->output_checksum = ds41f_prefill_checksum64((const unsigned char *)out_f32, out_elems * sizeof(float));
    return 0;
}

static float ds41f_fp8_e4m3fn_to_f32(uint8_t x) {
    if ((x & 0x7fu) == 0u) return (x & 0x80u) ? -0.0f : 0.0f;
    uint32_t sign = x & 0x80u;
    uint32_t exp = (x >> 3) & 0x0fu;
    uint32_t man = x & 0x07u;
    float v;
    if (exp == 0u) v = ldexpf((float)man / 8.0f, -6);
    else v = ldexpf(1.0f + (float)man / 8.0f, (int)exp - 7);
    return sign ? -v : v;
}

static float ds41f_fp8_e8m0fnu_to_f32(uint8_t x) {
    return ldexpf(1.0f, (int)x - 127);
}

static uint8_t ds41f_f32_to_fp8_e4m3fn_nearest(float v) {
    if (isnan(v)) return 0x7fu;
    if (v == 0.0f) return signbit(v) ? 0x80u : 0x00u;
    if (v > 448.0f) v = 448.0f;
    if (v < -448.0f) v = -448.0f;
    uint8_t best = 0;
    float bestd = INFINITY;
    for (uint32_t c = 0; c < 256u; c++) {
        if ((c & 0x7fu) == 0x7fu) continue; /* NaN code in finite-only E4M3FN. */
        float dv = ds41f_fp8_e4m3fn_to_f32((uint8_t)c);
        float d = fabsf(v - dv);
        if (d < bestd || (d == bestd && ((c & 1u) == 0u) && ((best & 1u) != 0u))) {
            bestd = d;
            best = (uint8_t)c;
        }
    }
    return best;
}

static uint16_t ds41f_f32_to_bf16_rne(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}

int ds41f_official_fp8_linear_bf16(const uint16_t *input_bf16,
                                   const uint8_t *weight_fp8,
                                   const uint8_t *weight_scale_e8m0,
                                   uint32_t rows,
                                   uint32_t in_dim,
                                   uint32_t out_dim,
                                   uint32_t block_size,
                                   uint16_t *out_bf16,
                                   ds41f_official_linear_result *result) {
    if (!input_bf16 || !weight_fp8 || !weight_scale_e8m0 || !out_bf16 || !result) return -1;
    if (rows == 0 || in_dim == 0 || out_dim == 0 || block_size == 0) return -2;
    if (in_dim % block_size != 0) return -3;
    memset(result, 0, sizeof(*result));
    result->rows = rows;
    result->in_dim = in_dim;
    result->out_dim = out_dim;
    const uint64_t input_elems = (uint64_t)rows * (uint64_t)in_dim;
    const uint64_t weight_elems = (uint64_t)out_dim * (uint64_t)in_dim;
    const uint64_t out_elems = (uint64_t)rows * (uint64_t)out_dim;
    const uint32_t k_blocks = in_dim / block_size;
    const uint32_t n_blocks = (out_dim + block_size - 1u) / block_size;
    const uint64_t scale_elems = (uint64_t)n_blocks * (uint64_t)k_blocks;
    if (input_elems > SIZE_MAX / sizeof(uint16_t) || weight_elems > SIZE_MAX || out_elems > SIZE_MAX / sizeof(uint16_t) || scale_elems > SIZE_MAX) return -4;
#if DS41F_HAVE_METAL
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return -10;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) { [device release]; return -11; }
        NSString *src = @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "static inline float bf16_to_f32(ushort x) { uint u = ((uint)x) << 16; return as_type<float>(u); }\n"
            "static inline float e8m0_to_f32(uchar x) { return exp2((float)((int)x - 127)); }\n"
            "static inline float e4m3_to_f32(uchar x) { uint ax = ((uint)x) & 0x7fu; if (ax == 0u) return (((uint)x) & 0x80u) ? -0.0f : 0.0f; uint e = (((uint)x) >> 3) & 15u; uint m = ((uint)x) & 7u; float v = (e == 0u) ? ldexp((float)m / 8.0f, -6) : ldexp(1.0f + (float)m / 8.0f, (int)e - 7); return (((uint)x) & 0x80u) ? -v : v; }\n"
            "static inline uchar f32_to_e4m3(float v) { if (v > 448.0f) v = 448.0f; if (v < -448.0f) v = -448.0f; uchar best = 0; float bestd = INFINITY; for (uint c = 0; c < 256u; c++) { if ((c & 0x7fu) == 0x7fu) continue; float d = fabs(v - e4m3_to_f32((uchar)c)); if (d < bestd || (d == bestd && ((c & 1u) == 0u) && (((uint)best & 1u) != 0u))) { bestd = d; best = (uchar)c; } } return best; }\n"
            "static inline ushort f32_to_bf16(float x) { uint u = as_type<uint>(x); uint l = (u >> 16) & 1u; u += 0x7fffu + l; return (ushort)(u >> 16); }\n"
            "kernel void ds41f_fp8_linear_bf16(device const ushort *inp [[buffer(0)]], device const uchar *w [[buffer(1)]], device const uchar *ws [[buffer(2)]], device ushort *out [[buffer(3)]], constant uint &in_dim [[buffer(4)]], constant uint &out_dim [[buffer(5)]], constant uint &block_size [[buffer(6)]], uint gid [[thread_position_in_grid]]) {\n"
            "  uint r = gid / out_dim; uint o = gid - r * out_dim; uint k_blocks = in_dim / block_size; uint n_blocks = (out_dim + block_size - 1u) / block_size; float acc = 0.0f;\n"
            "  for (uint kb = 0; kb < k_blocks; kb++) { float amax = 0.0f; for (uint t = 0; t < block_size; t++) { float av = fabs(bf16_to_f32(inp[r * in_dim + kb * block_size + t])); amax = max(amax, av); } amax = max(amax, 1.0e-4f); float as = exp2(ceil(log2(amax / 448.0f))); float bs = e8m0_to_f32(ws[(o / block_size) * k_blocks + kb]); for (uint t = 0; t < block_size; t++) { float x = bf16_to_f32(inp[r * in_dim + kb * block_size + t]); uchar aq = f32_to_e4m3(clamp(x / as, -448.0f, 448.0f)); float ad = e4m3_to_f32(aq) * as; float bd = e4m3_to_f32(w[o * in_dim + kb * block_size + t]) * bs; acc += ad * bd; } }\n"
            "  out[gid] = f32_to_bf16(acc);\n"
            "}\n";
        NSError *err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
        if (!lib) { [queue release]; [device release]; return -12; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"ds41f_fp8_linear_bf16"];
        if (!fn) { [lib release]; [queue release]; [device release]; return -13; }
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { [fn release]; [lib release]; [queue release]; [device release]; return -14; }
        id<MTLBuffer> ib = [device newBufferWithBytes:input_bf16 length:(NSUInteger)(input_elems * sizeof(uint16_t)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> wb = [device newBufferWithBytes:weight_fp8 length:(NSUInteger)weight_elems options:MTLResourceStorageModeShared];
        id<MTLBuffer> sb = [device newBufferWithBytes:weight_scale_e8m0 length:(NSUInteger)scale_elems options:MTLResourceStorageModeShared];
        id<MTLBuffer> ob = [device newBufferWithLength:(NSUInteger)(out_elems * sizeof(uint16_t)) options:MTLResourceStorageModeShared];
        if (!ib || !wb || !sb || !ob) { [ob release]; [sb release]; [wb release]; [ib release]; [pso release]; [fn release]; [lib release]; [queue release]; [device release]; return -15; }
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:ib offset:0 atIndex:0]; [enc setBuffer:wb offset:0 atIndex:1]; [enc setBuffer:sb offset:0 atIndex:2]; [enc setBuffer:ob offset:0 atIndex:3];
        [enc setBytes:&in_dim length:sizeof(in_dim) atIndex:4]; [enc setBytes:&out_dim length:sizeof(out_dim) atIndex:5]; [enc setBytes:&block_size length:sizeof(block_size) atIndex:6];
        const NSUInteger tw = pso.threadExecutionWidth ? pso.threadExecutionWidth : 32u;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)out_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) { [ob release]; [sb release]; [wb release]; [ib release]; [pso release]; [fn release]; [lib release]; [queue release]; [device release]; return -16; }
        memcpy(out_bf16, [ob contents], (size_t)(out_elems * sizeof(uint16_t)));
        result->metal_enabled = 1u; result->metal_command_buffers = 1u; result->metal_compute_encoders = 1u; result->metal_completion_waits = 1u;
        [ob release]; [sb release]; [wb release]; [ib release]; [pso release]; [fn release]; [lib release]; [queue release]; [device release];
    }
#else
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            float acc = 0.0f;
            for (uint32_t kb = 0; kb < k_blocks; kb++) {
                float amax = 0.0f;
                for (uint32_t t = 0; t < block_size; t++) {
                    uint32_t u = ((uint32_t)input_bf16[(uint64_t)r * in_dim + kb * block_size + t]) << 16; float xv; memcpy(&xv, &u, sizeof(xv));
                    amax = fmaxf(amax, fabsf(xv));
                }
                amax = fmaxf(amax, 1.0e-4f);
                float as = exp2f(ceilf(log2f(amax / 448.0f)));
                float bs = ds41f_fp8_e8m0fnu_to_f32(weight_scale_e8m0[(uint64_t)(o / block_size) * k_blocks + kb]);
                for (uint32_t t = 0; t < block_size; t++) {
                    uint32_t u = ((uint32_t)input_bf16[(uint64_t)r * in_dim + kb * block_size + t]) << 16; float xv; memcpy(&xv, &u, sizeof(xv));
                    uint8_t aq = ds41f_f32_to_fp8_e4m3fn_nearest(fminf(fmaxf(xv / as, -448.0f), 448.0f));
                    acc += ds41f_fp8_e4m3fn_to_f32(aq) * as * ds41f_fp8_e4m3fn_to_f32(weight_fp8[(uint64_t)o * in_dim + kb * block_size + t]) * bs;
                }
            }
            out_bf16[(uint64_t)r * out_dim + o] = ds41f_f32_to_bf16_rne(acc);
        }
    }
#endif
    result->output_checksum = ds41f_prefill_checksum64((const unsigned char *)out_bf16, out_elems * sizeof(uint16_t));
    return 0;
}

int ds41f_official_act_quant_bf16(const uint16_t *input_bf16,
                                  uint32_t rows,
                                  uint32_t dim,
                                  uint32_t block_size,
                                  uint8_t *out_fp8,
                                  uint8_t *out_scale_e8m0,
                                  uint16_t *out_bf16,
                                  ds41f_official_linear_result *result) {
    if (!input_bf16 || !out_fp8 || !out_scale_e8m0 || !out_bf16 || !result) return -1;
    if (rows == 0 || dim == 0 || block_size == 0 || dim % block_size != 0) return -2;
    memset(result, 0, sizeof(*result));
    result->rows = rows; result->in_dim = dim; result->out_dim = dim;
    const uint64_t elems = (uint64_t)rows * (uint64_t)dim;
    const uint64_t scale_elems = (uint64_t)rows * (uint64_t)(dim / block_size);
    if (elems > SIZE_MAX / sizeof(uint16_t) || scale_elems > SIZE_MAX) return -3;
#if DS41F_HAVE_METAL
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice(); if (!device) return -10;
        id<MTLCommandQueue> queue = [device newCommandQueue]; if (!queue) { [device release]; return -11; }
        NSString *src = @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "static inline float bf16_to_f32(ushort x) { uint u = ((uint)x) << 16; return as_type<float>(u); }\n"
            "static inline float e4m3_to_f32(uchar x) { uint ax = ((uint)x) & 0x7fu; if (ax == 0u) return (((uint)x) & 0x80u) ? -0.0f : 0.0f; uint e = (((uint)x) >> 3) & 15u; uint m = ((uint)x) & 7u; float v = (e == 0u) ? ldexp((float)m / 8.0f, -6) : ldexp(1.0f + (float)m / 8.0f, (int)e - 7); return (((uint)x) & 0x80u) ? -v : v; }\n"
            "static inline uchar f32_to_e4m3(float v) { if (v > 448.0f) v = 448.0f; if (v < -448.0f) v = -448.0f; uchar best = 0; float bestd = INFINITY; for (uint c = 0; c < 256u; c++) { if ((c & 0x7fu) == 0x7fu) continue; float d = fabs(v - e4m3_to_f32((uchar)c)); if (d < bestd || (d == bestd && ((c & 1u) == 0u) && (((uint)best & 1u) != 0u))) { bestd = d; best = (uchar)c; } } return best; }\n"
            "static inline ushort f32_to_bf16(float x) { uint u = as_type<uint>(x); uint l = (u >> 16) & 1u; u += 0x7fffu + l; return (ushort)(u >> 16); }\n"
            "kernel void ds41f_act_quant_bf16(device const ushort *inp [[buffer(0)]], device uchar *q [[buffer(1)]], device uchar *s [[buffer(2)]], device ushort *out [[buffer(3)]], constant uint &dim [[buffer(4)]], constant uint &block_size [[buffer(5)]], uint gid [[thread_position_in_grid]]) {\n"
            "  uint blocks = dim / block_size; uint r = gid / blocks; uint b = gid - r * blocks; float amax = 0.0f; uint base = r * dim + b * block_size;\n"
            "  for (uint t = 0; t < block_size; t++) amax = max(amax, fabs(bf16_to_f32(inp[base + t])));\n"
            "  amax = max(amax, 1.0e-4f); int e = (int)ceil(log2(amax / 448.0f)); float scale = exp2((float)e); s[gid] = (uchar)(e + 127);\n"
            "  for (uint t = 0; t < block_size; t++) { float x = bf16_to_f32(inp[base + t]); uchar c = f32_to_e4m3(clamp(x / scale, -448.0f, 448.0f)); q[base + t] = c; out[base + t] = f32_to_bf16(e4m3_to_f32(c) * scale); }\n"
            "}\n";
        NSError *err = nil; id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err]; if (!lib) { [queue release]; [device release]; return -12; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"ds41f_act_quant_bf16"]; if (!fn) { [lib release]; [queue release]; [device release]; return -13; }
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err]; if (!pso) { [fn release]; [lib release]; [queue release]; [device release]; return -14; }
        id<MTLBuffer> ib = [device newBufferWithBytes:input_bf16 length:(NSUInteger)(elems * sizeof(uint16_t)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> qb = [device newBufferWithLength:(NSUInteger)elems options:MTLResourceStorageModeShared];
        id<MTLBuffer> sb = [device newBufferWithLength:(NSUInteger)scale_elems options:MTLResourceStorageModeShared];
        id<MTLBuffer> ob = [device newBufferWithLength:(NSUInteger)(elems * sizeof(uint16_t)) options:MTLResourceStorageModeShared];
        if (!ib || !qb || !sb || !ob) { [ob release]; [sb release]; [qb release]; [ib release]; [pso release]; [fn release]; [lib release]; [queue release]; [device release]; return -15; }
        id<MTLCommandBuffer> cb = [queue commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso]; [enc setBuffer:ib offset:0 atIndex:0]; [enc setBuffer:qb offset:0 atIndex:1]; [enc setBuffer:sb offset:0 atIndex:2]; [enc setBuffer:ob offset:0 atIndex:3];
        [enc setBytes:&dim length:sizeof(dim) atIndex:4]; [enc setBytes:&block_size length:sizeof(block_size) atIndex:5];
        const NSUInteger tw = pso.threadExecutionWidth ? pso.threadExecutionWidth : 32u;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)scale_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)]; [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) { [ob release]; [sb release]; [qb release]; [ib release]; [pso release]; [fn release]; [lib release]; [queue release]; [device release]; return -16; }
        memcpy(out_fp8, [qb contents], (size_t)elems); memcpy(out_scale_e8m0, [sb contents], (size_t)scale_elems); memcpy(out_bf16, [ob contents], (size_t)(elems * sizeof(uint16_t)));
        result->metal_enabled = 1u; result->metal_command_buffers = 1u; result->metal_compute_encoders = 1u; result->metal_completion_waits = 1u;
        [ob release]; [sb release]; [qb release]; [ib release]; [pso release]; [fn release]; [lib release]; [queue release]; [device release];
    }
#else
    const uint32_t blocks = dim / block_size;
    for (uint32_t r = 0; r < rows; r++) for (uint32_t b = 0; b < blocks; b++) {
        float amax = 0.0f; uint64_t base = (uint64_t)r * dim + (uint64_t)b * block_size;
        for (uint32_t t = 0; t < block_size; t++) { uint32_t u = ((uint32_t)input_bf16[base+t]) << 16; float x; memcpy(&x,&u,4); amax = fmaxf(amax, fabsf(x)); }
        amax = fmaxf(amax, 1.0e-4f); int e = (int)ceilf(log2f(amax / 448.0f)); float scale = exp2f((float)e); out_scale_e8m0[(uint64_t)r*blocks+b] = (uint8_t)(e+127);
        for (uint32_t t = 0; t < block_size; t++) { uint32_t u = ((uint32_t)input_bf16[base+t]) << 16; float x; memcpy(&x,&u,4); uint8_t c=ds41f_f32_to_fp8_e4m3fn_nearest(fminf(fmaxf(x/scale,-448.0f),448.0f)); out_fp8[base+t]=c; out_bf16[base+t]=ds41f_f32_to_bf16_rne(ds41f_fp8_e4m3fn_to_f32(c)*scale); }
    }
#endif
    result->output_checksum = ds41f_prefill_checksum64((const unsigned char *)out_bf16, elems * sizeof(uint16_t));
    return 0;
}

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
                                    ds41f_official_linear_result *result) {
    if (!q_bf16 || !kv_bf16 || !attn_sink_f32 || !topk_idxs || !out_bf16 || !result) return -1;
    if (batch == 0 || seqlen == 0 || kv_len == 0 || heads == 0 || dim == 0 || topk == 0) return -2;
    memset(result, 0, sizeof(*result));
    result->rows = batch * seqlen * heads;
    result->in_dim = dim;
    result->out_dim = dim;
    const uint64_t q_elems = (uint64_t)batch * seqlen * heads * dim;
    const uint64_t kv_elems = (uint64_t)batch * kv_len * dim;
    const uint64_t idx_elems = (uint64_t)batch * seqlen * topk;
    if (q_elems > SIZE_MAX / sizeof(uint16_t) || kv_elems > SIZE_MAX / sizeof(uint16_t) || idx_elems > SIZE_MAX / sizeof(int32_t)) return -3;
#if DS41F_HAVE_METAL
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice(); if (!device) return -10;
        id<MTLCommandQueue> queue = [device newCommandQueue]; if (!queue) { [device release]; return -11; }
        NSString *src = @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "static inline float bf16_to_f32(ushort x) { uint u = ((uint)x) << 16; return as_type<float>(u); }\n"
            "static inline ushort f32_to_bf16(float x) { uint u = as_type<uint>(x); uint l = (u >> 16) & 1u; u += 0x7fffu + l; return (ushort)(u >> 16); }\n"
            "kernel void ds41f_sparse_attn_bf16(device const ushort *q [[buffer(0)]], device const ushort *kv [[buffer(1)]], device const float *sink [[buffer(2)]], device const int *idxs [[buffer(3)]], device ushort *out [[buffer(4)]], constant uint &seqlen [[buffer(5)]], constant uint &kv_len [[buffer(6)]], constant uint &heads [[buffer(7)]], constant uint &dim [[buffer(8)]], constant uint &topk [[buffer(9)]], constant float &scale [[buffer(10)]], uint gid [[thread_position_in_grid]]) {\n"
            "  uint d = gid % dim; uint tmp = gid / dim; uint h = tmp % heads; tmp /= heads; uint m = tmp % seqlen; uint b = tmp / seqlen;\n"
            "  float scores[8]; float rowmax = -1.0e30f;\n"
            "  for (uint t = 0; t < topk; t++) { int ix = idxs[(b * seqlen + m) * topk + t]; float sc = -INFINITY; if (ix >= 0 && (uint)ix < kv_len) { sc = 0.0f; for (uint k = 0; k < dim; k++) sc += bf16_to_f32(q[((b * seqlen + m) * heads + h) * dim + k]) * bf16_to_f32(kv[(b * kv_len + (uint)ix) * dim + k]); sc *= scale; rowmax = max(rowmax, sc); } scores[t] = sc; }\n"
            "  float denom = 0.0f; float acc = 0.0f;\n"
            "  for (uint t = 0; t < topk; t++) { int ix = idxs[(b * seqlen + m) * topk + t]; if (ix >= 0 && (uint)ix < kv_len) { float p = exp(scores[t] - rowmax); denom += p; float pb = bf16_to_f32(f32_to_bf16(p)); acc += pb * bf16_to_f32(kv[(b * kv_len + (uint)ix) * dim + d]); } }\n"
            "  denom += exp(sink[h] - rowmax);\n"
            "  out[gid] = f32_to_bf16(acc / denom);\n"
            "}\n";
        NSError *err = nil; id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err]; if (!lib) { [queue release]; [device release]; return -12; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"ds41f_sparse_attn_bf16"]; if (!fn) { [lib release]; [queue release]; [device release]; return -13; }
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err]; if (!pso) { [fn release]; [lib release]; [queue release]; [device release]; return -14; }
        id<MTLBuffer> qb = [device newBufferWithBytes:q_bf16 length:(NSUInteger)(q_elems * sizeof(uint16_t)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> kb = [device newBufferWithBytes:kv_bf16 length:(NSUInteger)(kv_elems * sizeof(uint16_t)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> sb = [device newBufferWithBytes:attn_sink_f32 length:(NSUInteger)(heads * sizeof(float)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> ib = [device newBufferWithBytes:topk_idxs length:(NSUInteger)(idx_elems * sizeof(int32_t)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> ob = [device newBufferWithLength:(NSUInteger)(q_elems * sizeof(uint16_t)) options:MTLResourceStorageModeShared];
        if (!qb || !kb || !sb || !ib || !ob) { [ob release]; [ib release]; [sb release]; [kb release]; [qb release]; [pso release]; [fn release]; [lib release]; [queue release]; [device release]; return -15; }
        id<MTLCommandBuffer> cb = [queue commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso]; [enc setBuffer:qb offset:0 atIndex:0]; [enc setBuffer:kb offset:0 atIndex:1]; [enc setBuffer:sb offset:0 atIndex:2]; [enc setBuffer:ib offset:0 atIndex:3]; [enc setBuffer:ob offset:0 atIndex:4];
        [enc setBytes:&seqlen length:sizeof(seqlen) atIndex:5]; [enc setBytes:&kv_len length:sizeof(kv_len) atIndex:6]; [enc setBytes:&heads length:sizeof(heads) atIndex:7]; [enc setBytes:&dim length:sizeof(dim) atIndex:8]; [enc setBytes:&topk length:sizeof(topk) atIndex:9]; [enc setBytes:&softmax_scale length:sizeof(softmax_scale) atIndex:10];
        const NSUInteger tw = pso.threadExecutionWidth ? pso.threadExecutionWidth : 32u;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)q_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)]; [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) { [ob release]; [ib release]; [sb release]; [kb release]; [qb release]; [pso release]; [fn release]; [lib release]; [queue release]; [device release]; return -16; }
        memcpy(out_bf16, [ob contents], (size_t)(q_elems * sizeof(uint16_t)));
        result->metal_enabled = 1u; result->metal_command_buffers = 1u; result->metal_compute_encoders = 1u; result->metal_completion_waits = 1u;
        [ob release]; [ib release]; [sb release]; [kb release]; [qb release]; [pso release]; [fn release]; [lib release]; [queue release]; [device release];
    }
#else
    for (uint32_t b = 0; b < batch; b++) for (uint32_t m = 0; m < seqlen; m++) for (uint32_t h = 0; h < heads; h++) {
        float *scores = (float *)alloca((size_t)topk * sizeof(float));
        float rowmax = -1.0e30f;
        for (uint32_t t = 0; t < topk; t++) {
            int32_t ix = topk_idxs[((uint64_t)b * seqlen + m) * topk + t]; float sc = -INFINITY;
            if (ix >= 0 && (uint32_t)ix < kv_len) { sc = 0.0f; for (uint32_t k = 0; k < dim; k++) { uint32_t qu=((uint32_t)q_bf16[(((uint64_t)b*seqlen+m)*heads+h)*dim+k])<<16; uint32_t ku=((uint32_t)kv_bf16[((uint64_t)b*kv_len+(uint32_t)ix)*dim+k])<<16; float qf,kf; memcpy(&qf,&qu,4); memcpy(&kf,&ku,4); sc += qf*kf; } sc *= softmax_scale; rowmax = fmaxf(rowmax, sc); }
            scores[t] = sc;
        }
        float denom = 0.0f; for (uint32_t t = 0; t < topk; t++) if (isfinite(scores[t])) denom += expf(scores[t] - rowmax); denom += expf(attn_sink_f32[h] - rowmax);
        for (uint32_t d = 0; d < dim; d++) { float acc=0.0f; for (uint32_t t=0;t<topk;t++){ int32_t ix=topk_idxs[((uint64_t)b*seqlen+m)*topk+t]; if(ix>=0&&(uint32_t)ix<kv_len){ float p=expf(scores[t]-rowmax); uint16_t pb=ds41f_f32_to_bf16_rne(p); uint32_t pu=((uint32_t)pb)<<16; float pf; memcpy(&pf,&pu,4); uint32_t vu=((uint32_t)kv_bf16[((uint64_t)b*kv_len+(uint32_t)ix)*dim+d])<<16; float vf; memcpy(&vf,&vu,4); acc += pf*vf; }} out_bf16[(((uint64_t)b*seqlen+m)*heads+h)*dim+d] = ds41f_f32_to_bf16_rne(acc/denom); }
    }
#endif
    result->output_checksum = ds41f_prefill_checksum64((const unsigned char *)out_bf16, q_elems * sizeof(uint16_t));
    return 0;
}

int ds41f_official_rmsnorm_bf16(const uint16_t *input_bf16,
                                const uint16_t *weight_bf16,
                                uint32_t rows,
                                uint32_t dim,
                                float eps,
                                uint16_t *out_bf16,
                                ds41f_official_rmsnorm_result *result) {
    if (!input_bf16 || !weight_bf16 || !out_bf16 || !result) return -1;
    if (rows == 0 || dim == 0) return -2;
    memset(result, 0, sizeof(*result));
    result->rows = rows;
    result->dim = dim;
    const uint64_t elems = (uint64_t)rows * (uint64_t)dim;
    if (elems > SIZE_MAX / sizeof(uint16_t)) return -3;
#if DS41F_HAVE_METAL
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return -10;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) { [device release]; return -11; }
        NSString *src = @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "static inline float ds41f_bf16_to_f32(ushort x) { uint u = ((uint)x) << 16; return as_type<float>(u); }\n"
            "static inline ushort ds41f_f32_to_bf16_rne(float x) { uint u = as_type<uint>(x); uint lsb = (u >> 16) & 1u; u += 0x7fffu + lsb; return (ushort)(u >> 16); }\n"
            "kernel void ds41f_rmsnorm_bf16(device const ushort *inp [[buffer(0)]],\n"
            "                                device const ushort *w [[buffer(1)]],\n"
            "                                device ushort *out [[buffer(2)]],\n"
            "                                constant uint &dim [[buffer(3)]],\n"
            "                                constant float &eps [[buffer(4)]],\n"
            "                                uint gid [[thread_position_in_grid]]) {\n"
            "  uint r = gid / dim; uint c = gid - r * dim; float sum = 0.0f;\n"
            "  for (uint k = 0; k < dim; k++) { float v = ds41f_bf16_to_f32(inp[r * dim + k]); sum += v * v; }\n"
            "  float inv = 1.0f / precise::sqrt(sum / (float)dim + eps);\n"
            "  float y = ds41f_bf16_to_f32(inp[gid]) * inv * ds41f_bf16_to_f32(w[c]);\n"
            "  out[gid] = ds41f_f32_to_bf16_rne(y);\n"
            "}\n";
        NSError *err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
        if (!lib) { [queue release]; [device release]; return -12; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"ds41f_rmsnorm_bf16"];
        if (!fn) { [lib release]; [queue release]; [device release]; return -13; }
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { [fn release]; [lib release]; [queue release]; [device release]; return -14; }
        id<MTLBuffer> ib = [device newBufferWithBytes:input_bf16 length:(NSUInteger)(elems * sizeof(uint16_t)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> wb = [device newBufferWithBytes:weight_bf16 length:(NSUInteger)dim * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> ob = [device newBufferWithLength:(NSUInteger)(elems * sizeof(uint16_t)) options:MTLResourceStorageModeShared];
        if (!ib || !wb || !ob) {
            [ob release]; [wb release]; [ib release]; [pso release]; [fn release]; [lib release]; [queue release]; [device release];
            return -15;
        }
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:ib offset:0 atIndex:0];
        [enc setBuffer:wb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBytes:&dim length:sizeof(dim) atIndex:3];
        [enc setBytes:&eps length:sizeof(eps) atIndex:4];
        const NSUInteger tw = pso.threadExecutionWidth ? pso.threadExecutionWidth : 32u;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)elems, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            [ob release]; [wb release]; [ib release]; [pso release]; [fn release]; [lib release]; [queue release]; [device release];
            return -16;
        }
        memcpy(out_bf16, [ob contents], (size_t)(elems * sizeof(uint16_t)));
        result->metal_enabled = 1u;
        result->metal_command_buffers = 1u;
        result->metal_compute_encoders = 1u;
        result->metal_completion_waits = 1u;
        [ob release]; [wb release]; [ib release]; [pso release]; [fn release]; [lib release]; [queue release]; [device release];
    }
#else
    for (uint32_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < dim; k++) {
            uint32_t ix = ((uint32_t)input_bf16[(uint64_t)r * dim + k]) << 16;
            float v;
            memcpy(&v, &ix, sizeof(float));
            sum += v * v;
        }
        float inv = 1.0f / __builtin_sqrtf(sum / (float)dim + eps);
        for (uint32_t c = 0; c < dim; c++) {
            uint32_t ix = ((uint32_t)input_bf16[(uint64_t)r * dim + c]) << 16;
            uint32_t wx = ((uint32_t)weight_bf16[c]) << 16;
            float x, w;
            memcpy(&x, &ix, sizeof(float));
            memcpy(&w, &wx, sizeof(float));
            float y = x * inv * w;
            uint32_t uy;
            memcpy(&uy, &y, sizeof(uint32_t));
            uint32_t lsb = (uy >> 16) & 1u;
            uy += 0x7fffu + lsb;
            out_bf16[(uint64_t)r * dim + c] = (uint16_t)(uy >> 16);
        }
    }
#endif
    result->output_checksum = ds41f_prefill_checksum64((const unsigned char *)out_bf16, elems * sizeof(uint16_t));
    return 0;
}

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
                              ds41f_official_rotary_result *result) {
    if (!input_f32 || !freqs_real_f32 || !freqs_imag_f32 || !rotated_f32 || !inverse_f32 || !result) return -1;
    if (batch == 0 || seqlen == 0 || heads == 0 || dim == 0 || (dim & 1u) != 0) return -2;
    memset(result, 0, sizeof(*result));
    result->batch = batch;
    result->seqlen = seqlen;
    result->heads = heads;
    result->dim = dim;
    result->original_seq_len = original_seq_len;
    const uint64_t pairs_per_pos = (uint64_t)dim / 2ull;
    const uint64_t freq_elems = (uint64_t)seqlen * pairs_per_pos;
    const uint64_t complex_elems = (uint64_t)batch * (uint64_t)seqlen * (uint64_t)heads * pairs_per_pos;
    const uint64_t scalar_elems = complex_elems * 2ull;
    if (freq_elems > SIZE_MAX / sizeof(float) || scalar_elems > SIZE_MAX / sizeof(float)) return -3;
#if DS41F_HAVE_METAL
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return -10;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) { [device release]; return -11; }
        NSString *src = @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "kernel void ds41f_rotary_precompute(device float *fr [[buffer(0)]], device float *fi [[buffer(1)]],\n"
            "                                     constant uint &dim [[buffer(2)]], constant uint &seqlen [[buffer(3)]],\n"
            "                                     constant uint &original_seq_len [[buffer(4)]], constant float &base [[buffer(5)]],\n"
            "                                     constant float &factor [[buffer(6)]], constant uint &beta_fast [[buffer(7)]],\n"
            "                                     constant uint &beta_slow [[buffer(8)]], uint gid [[thread_position_in_grid]]) {\n"
            "  uint hdim = dim >> 1; uint pos = gid / hdim; uint j = gid - pos * hdim;\n"
            "  float freq = 1.0f / pow(base, ((float)(j * 2u)) / (float)dim);\n"
            "  if (original_seq_len > 0u) {\n"
            "    float cd_fast = ((float)dim) * log(((float)original_seq_len) / (((float)beta_fast) * 6.2831853071795864769f)) / (2.0f * log(base));\n"
            "    float cd_slow = ((float)dim) * log(((float)original_seq_len) / (((float)beta_slow) * 6.2831853071795864769f)) / (2.0f * log(base));\n"
            "    float low = max(floor(cd_fast), 0.0f); float high = min(ceil(cd_slow), (float)(dim - 1u));\n"
            "    float ramp = clamp(((float)j - low) / max(high - low, 1.0e-3f), 0.0f, 1.0f); float smooth = 1.0f - ramp;\n"
            "    freq = freq / factor * (1.0f - smooth) + freq * smooth;\n"
            "  }\n"
            "  float angle = ((float)pos) * freq; fr[gid] = cos(angle); fi[gid] = sin(angle);\n"
            "}\n"
            "kernel void ds41f_rotary_apply(device const float *inp [[buffer(0)]], device const float *fr [[buffer(1)]],\n"
            "                                device const float *fi [[buffer(2)]], device float *out [[buffer(3)]],\n"
            "                                constant uint &seqlen [[buffer(4)]], constant uint &heads [[buffer(5)]],\n"
            "                                constant uint &dim [[buffer(6)]], constant uint &inverse [[buffer(7)]],\n"
            "                                uint gid [[thread_position_in_grid]]) {\n"
            "  uint hdim = dim >> 1; uint pair = gid % hdim; uint tmp = gid / hdim; uint h = tmp % heads; tmp /= heads; uint s = tmp % seqlen; uint b = tmp / seqlen;\n"
            "  uint scalar = (((b * seqlen + s) * heads + h) * dim) + pair * 2u; uint fidx = s * hdim + pair;\n"
            "  float xr = inp[scalar]; float xi = inp[scalar + 1u]; float cr = fr[fidx]; float ci = inverse ? -fi[fidx] : fi[fidx];\n"
            "  out[scalar] = xr * cr - xi * ci; out[scalar + 1u] = xr * ci + xi * cr;\n"
            "}\n";
        NSError *err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
        if (!lib) { [queue release]; [device release]; return -12; }
        id<MTLFunction> fn_pre = [lib newFunctionWithName:@"ds41f_rotary_precompute"];
        id<MTLFunction> fn_apply = [lib newFunctionWithName:@"ds41f_rotary_apply"];
        if (!fn_pre || !fn_apply) { [fn_apply release]; [fn_pre release]; [lib release]; [queue release]; [device release]; return -13; }
        id<MTLComputePipelineState> pso_pre = [device newComputePipelineStateWithFunction:fn_pre error:&err];
        id<MTLComputePipelineState> pso_apply = [device newComputePipelineStateWithFunction:fn_apply error:&err];
        if (!pso_pre || !pso_apply) { [pso_apply release]; [pso_pre release]; [fn_apply release]; [fn_pre release]; [lib release]; [queue release]; [device release]; return -14; }
        id<MTLBuffer> ib = [device newBufferWithBytes:input_f32 length:(NSUInteger)(scalar_elems * sizeof(float)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> frb = [device newBufferWithLength:(NSUInteger)(freq_elems * sizeof(float)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> fib = [device newBufferWithLength:(NSUInteger)(freq_elems * sizeof(float)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> rb = [device newBufferWithLength:(NSUInteger)(scalar_elems * sizeof(float)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> invb = [device newBufferWithLength:(NSUInteger)(scalar_elems * sizeof(float)) options:MTLResourceStorageModeShared];
        if (!ib || !frb || !fib || !rb || !invb) {
            [invb release]; [rb release]; [fib release]; [frb release]; [ib release]; [pso_apply release]; [pso_pre release]; [fn_apply release]; [fn_pre release]; [lib release]; [queue release]; [device release];
            return -15;
        }
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso_pre];
        [enc setBuffer:frb offset:0 atIndex:0]; [enc setBuffer:fib offset:0 atIndex:1];
        [enc setBytes:&dim length:sizeof(dim) atIndex:2]; [enc setBytes:&seqlen length:sizeof(seqlen) atIndex:3];
        [enc setBytes:&original_seq_len length:sizeof(original_seq_len) atIndex:4]; [enc setBytes:&base length:sizeof(base) atIndex:5];
        [enc setBytes:&factor length:sizeof(factor) atIndex:6]; [enc setBytes:&beta_fast length:sizeof(beta_fast) atIndex:7]; [enc setBytes:&beta_slow length:sizeof(beta_slow) atIndex:8];
        NSUInteger tw_pre = pso_pre.threadExecutionWidth ? pso_pre.threadExecutionWidth : 32u;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)freq_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tw_pre, 1, 1)];
        [enc endEncoding];
        uint32_t inv0 = 0u, inv1 = 1u;
        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso_apply];
        [enc setBuffer:ib offset:0 atIndex:0]; [enc setBuffer:frb offset:0 atIndex:1]; [enc setBuffer:fib offset:0 atIndex:2]; [enc setBuffer:rb offset:0 atIndex:3];
        [enc setBytes:&seqlen length:sizeof(seqlen) atIndex:4]; [enc setBytes:&heads length:sizeof(heads) atIndex:5]; [enc setBytes:&dim length:sizeof(dim) atIndex:6]; [enc setBytes:&inv0 length:sizeof(inv0) atIndex:7];
        NSUInteger tw_apply = pso_apply.threadExecutionWidth ? pso_apply.threadExecutionWidth : 32u;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)complex_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tw_apply, 1, 1)];
        [enc endEncoding];
        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso_apply];
        [enc setBuffer:rb offset:0 atIndex:0]; [enc setBuffer:frb offset:0 atIndex:1]; [enc setBuffer:fib offset:0 atIndex:2]; [enc setBuffer:invb offset:0 atIndex:3];
        [enc setBytes:&seqlen length:sizeof(seqlen) atIndex:4]; [enc setBytes:&heads length:sizeof(heads) atIndex:5]; [enc setBytes:&dim length:sizeof(dim) atIndex:6]; [enc setBytes:&inv1 length:sizeof(inv1) atIndex:7];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)complex_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tw_apply, 1, 1)];
        [enc endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            [invb release]; [rb release]; [fib release]; [frb release]; [ib release]; [pso_apply release]; [pso_pre release]; [fn_apply release]; [fn_pre release]; [lib release]; [queue release]; [device release];
            return -16;
        }
        memcpy(freqs_real_f32, [frb contents], (size_t)(freq_elems * sizeof(float)));
        memcpy(freqs_imag_f32, [fib contents], (size_t)(freq_elems * sizeof(float)));
        memcpy(rotated_f32, [rb contents], (size_t)(scalar_elems * sizeof(float)));
        memcpy(inverse_f32, [invb contents], (size_t)(scalar_elems * sizeof(float)));
        result->metal_enabled = 1u; result->metal_command_buffers = 1u; result->metal_compute_encoders = 3u; result->metal_completion_waits = 1u;
        [invb release]; [rb release]; [fib release]; [frb release]; [ib release]; [pso_apply release]; [pso_pre release]; [fn_apply release]; [fn_pre release]; [lib release]; [queue release]; [device release];
    }
#else
    for (uint32_t pos = 0; pos < seqlen; pos++) {
        for (uint32_t j = 0; j < dim / 2u; j++) {
            float freq = 1.0f / powf(base, (float)(j * 2u) / (float)dim);
            if (original_seq_len > 0u) {
                float cd_fast = (float)dim * logf((float)original_seq_len / ((float)beta_fast * 6.2831853071795864769f)) / (2.0f * logf(base));
                float cd_slow = (float)dim * logf((float)original_seq_len / ((float)beta_slow * 6.2831853071795864769f)) / (2.0f * logf(base));
                float low = fmaxf(floorf(cd_fast), 0.0f), high = fminf(ceilf(cd_slow), (float)(dim - 1u));
                float ramp = fminf(fmaxf(((float)j - low) / fmaxf(high - low, 1.0e-3f), 0.0f), 1.0f);
                float smooth = 1.0f - ramp;
                freq = freq / factor * (1.0f - smooth) + freq * smooth;
            }
            float angle = (float)pos * freq;
            freqs_real_f32[(uint64_t)pos * (dim / 2u) + j] = cosf(angle);
            freqs_imag_f32[(uint64_t)pos * (dim / 2u) + j] = sinf(angle);
        }
    }
    for (uint64_t gid = 0; gid < complex_elems; gid++) {
        uint32_t hdim = dim / 2u;
        uint32_t pair = (uint32_t)(gid % hdim);
        uint64_t tmp = gid / hdim;
        uint32_t h = (uint32_t)(tmp % heads); tmp /= heads;
        uint32_t s = (uint32_t)(tmp % seqlen); tmp /= seqlen;
        uint32_t b = (uint32_t)tmp;
        uint64_t scalar = (((uint64_t)b * seqlen + s) * heads + h) * dim + (uint64_t)pair * 2ull;
        uint64_t fidx = (uint64_t)s * hdim + pair;
        float xr = input_f32[scalar], xi = input_f32[scalar + 1ull], cr = freqs_real_f32[fidx], ci = freqs_imag_f32[fidx];
        rotated_f32[scalar] = xr * cr - xi * ci;
        rotated_f32[scalar + 1ull] = xr * ci + xi * cr;
        xr = rotated_f32[scalar]; xi = rotated_f32[scalar + 1ull]; ci = -ci;
        inverse_f32[scalar] = xr * cr - xi * ci;
        inverse_f32[scalar + 1ull] = xr * ci + xi * cr;
    }
#endif
    result->freqs_checksum = ds41f_prefill_checksum64((const unsigned char *)freqs_real_f32, freq_elems * sizeof(float)) ^ ds41f_prefill_checksum64((const unsigned char *)freqs_imag_f32, freq_elems * sizeof(float));
    result->rotated_checksum = ds41f_prefill_checksum64((const unsigned char *)rotated_f32, scalar_elems * sizeof(float));
    result->inverse_checksum = ds41f_prefill_checksum64((const unsigned char *)inverse_f32, scalar_elems * sizeof(float));
    return 0;
}

int ds41f_prefill_native_build_plan(const ds41f_prefill_native_config *cfg,
                                    ds41f_prefill_native_plan *out) {
    if (!cfg || !out) return -1;
    if (cfg->n_layers == 0 || cfg->n_layers > DS41F_PREFILL_MAX_LAYERS) return -2;
    if (cfg->n_chunks == 0 || cfg->n_chunks > DS41F_PREFILL_MAX_CHUNKS) return -3;
    if (cfg->dim == 0 || cfg->hc_mult == 0 || cfg->vocab_size == 0) return -4;
    const uint32_t bps = cfg->bytes_per_hidden_scalar ? cfg->bytes_per_hidden_scalar : 2u;

    uint32_t total = 0;
    uint32_t max_chunk = 0;
    for (uint32_t i = 0; i < cfg->n_chunks; i++) {
        const uint32_t n = cfg->chunk_lengths[i];
        if (n == 0) return -5;
        if (UINT32_MAX - total < n) return -6;
        total += n;
        if (n > max_chunk) max_chunk = n;
    }

    uint32_t frontiers = 0;
    for (uint32_t il = 0; il < cfg->n_layers; il++) {
        if (cfg->frontier_masks[il] != 0) frontiers++;
    }

    const uint64_t hc_dim = (uint64_t)cfg->dim * (uint64_t)cfg->hc_mult;
    const uint64_t max_chunk_hidden = (uint64_t)max_chunk * hc_dim * (uint64_t)bps;
    const uint64_t full_hidden = (uint64_t)total * hc_dim * (uint64_t)bps;
    const uint64_t last_logits = (uint64_t)max_chunk * (uint64_t)cfg->vocab_size * 4ull;
    const uint64_t full_logits = (uint64_t)total * (uint64_t)cfg->vocab_size * 4ull;

    out->n_tokens = total;
    out->max_chunk_tokens = max_chunk;
    out->layer_steps = cfg->n_layers;
    /* DwarfStar-style split command batches: upload + one per layer + head/read. */
    out->command_batches = cfg->n_layers + 2u;
    out->publication_frontiers = frontiers;
    out->output_rows_retained = max_chunk;
    /* DwarfStar batch_cur_hc/batch_next_hc are full prefill-call buffers.
     * metal_graph_encode_layer_batch writes all rows for the layer, then swaps
     * cur/next once per layer. */
    (void)max_chunk_hidden;
    out->carry_buffer_bytes = 2ull * full_hidden;
    out->full_prompt_hidden_bytes = full_hidden;
    out->last_chunk_logits_bytes = last_logits;
    out->full_prompt_logits_bytes = full_logits;
    return 0;
}

int ds41f_prefill_native_context_create(const ds41f_prefill_native_config *cfg,
                                        ds41f_prefill_native_context **out) {
    if (!cfg || !out) return -1;
    *out = NULL;
    ds41f_prefill_native_plan plan;
    int rc = ds41f_prefill_native_build_plan(cfg, &plan);
    if (rc != 0) return rc;
    ds41f_prefill_native_context *ctx = (ds41f_prefill_native_context *)calloc(1, sizeof(*ctx));
    if (!ctx) return -10;
    ctx->cfg = *cfg;
    ctx->plan = plan;
    ctx->last_encoded_layer = UINT32_MAX;
    ctx->last_encoded_chunk = UINT32_MAX;
    ctx->tokens = (int32_t *)calloc((size_t)plan.n_tokens, sizeof(int32_t));
    const uint64_t one_carry = plan.carry_buffer_bytes / 2ull;
    ctx->carry[0] = (unsigned char *)calloc((size_t)one_carry, 1u);
    ctx->carry[1] = (unsigned char *)calloc((size_t)one_carry, 1u);
    if (!ctx->tokens || !ctx->carry[0] || !ctx->carry[1]) {
        ds41f_prefill_native_context_destroy(ctx);
        return -11;
    }
    *out = ctx;
    return 0;
}

void ds41f_prefill_native_context_destroy(ds41f_prefill_native_context *ctx) {
    if (!ctx) return;
    free(ctx->tokens);
    free(ctx->carry[0]);
    free(ctx->carry[1]);
    free(ctx->commands);
    free(ctx);
}

static uint64_t ds41f_prefill_checksum64(const unsigned char *p, uint64_t n) {
    if (!p || n == 0) return 0;
    uint64_t h = 1469598103934665603ull;
    const uint64_t limit = n < 4096ull ? n : 4096ull;
    for (uint64_t i = 0; i < limit; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

int ds41f_prefill_native_context_info(const ds41f_prefill_native_context *ctx,
                                      ds41f_prefill_native_arena_info *out) {
    if (!ctx || !out) return -1;
    const uint64_t one_carry = ctx->plan.carry_buffer_bytes / 2ull;
    out->carry_buffer_bytes = ctx->plan.carry_buffer_bytes;
    out->token_buffer_bytes = (uint64_t)ctx->plan.n_tokens * sizeof(int32_t);
    out->total_owned_bytes = out->carry_buffer_bytes + out->token_buffer_bytes;
    out->current_carry_checksum = ds41f_prefill_checksum64(ctx->carry[ctx->current], one_carry);
    out->next_carry_checksum = ds41f_prefill_checksum64(ctx->carry[1u - ctx->current], one_carry);
    out->n_tokens_uploaded = ctx->uploaded;
    out->current_carry_index = ctx->current;
    out->executed_encode_chunks = ctx->executed_encode_chunks;
    out->last_encoded_layer = ctx->last_encoded_layer;
    out->last_encoded_chunk = ctx->last_encoded_chunk;
    return 0;
}

int ds41f_prefill_native_upload_tokens(ds41f_prefill_native_context *ctx,
                                       uint32_t offset,
                                       const int32_t *tokens,
                                       uint32_t count) {
    if (!ctx || !tokens) return -1;
    if (offset > ctx->plan.n_tokens || count > ctx->plan.n_tokens - offset) return -2;
    memcpy(ctx->tokens + offset, tokens, (size_t)count * sizeof(int32_t));
    if (ctx->uploaded < offset + count) ctx->uploaded = offset + count;
    return 0;
}

int ds41f_prefill_native_swap_carry(ds41f_prefill_native_context *ctx) {
    if (!ctx) return -1;
    ctx->current = 1u - ctx->current;
    return 0;
}

void *ds41f_prefill_native_current_carry(ds41f_prefill_native_context *ctx) {
    if (!ctx) return NULL;
    return ctx->carry[ctx->current];
}

void *ds41f_prefill_native_next_carry(ds41f_prefill_native_context *ctx) {
    if (!ctx) return NULL;
    return ctx->carry[1u - ctx->current];
}

static void ds41f_cmd_set(ds41f_prefill_native_command *cmd,
                          uint32_t kind,
                          uint32_t layer,
                          uint32_t chunk,
                          uint32_t token_offset,
                          uint32_t token_count,
                          uint32_t frontier_mask) {
    cmd->kind = kind;
    cmd->layer = layer;
    cmd->chunk = chunk;
    cmd->token_offset = token_offset;
    cmd->token_count = token_count;
    cmd->frontier_mask = frontier_mask;
}

int ds41f_prefill_native_build_commands(ds41f_prefill_native_context *ctx) {
    if (!ctx) return -1;
    const uint32_t n_layers = ctx->cfg.n_layers;
    const uint32_t n_chunks = ctx->cfg.n_chunks;
    uint64_t count64 = 2ull + (uint64_t)n_layers * (4ull + (uint64_t)n_chunks) + 2ull;
    if (count64 > UINT32_MAX) return -2;
    ds41f_prefill_native_command *cmds = (ds41f_prefill_native_command *)calloc((size_t)count64, sizeof(*cmds));
    if (!cmds) return -3;
    uint32_t k = 0;
    ds41f_cmd_set(&cmds[k++], DS41F_CMD_UPLOAD_TOKENS, UINT32_MAX, UINT32_MAX, 0, ctx->plan.n_tokens, 0);
    ds41f_cmd_set(&cmds[k++], DS41F_CMD_UPLOAD_EMBEDDINGS_HC, UINT32_MAX, UINT32_MAX, 0, ctx->plan.n_tokens, 0);
    for (uint32_t layer = 0; layer < n_layers; layer++) {
        ds41f_cmd_set(&cmds[k++], DS41F_CMD_BEGIN_LAYER, layer, UINT32_MAX, 0, ctx->plan.n_tokens, 0);
        uint32_t off = 0;
        for (uint32_t chunk = 0; chunk < n_chunks; chunk++) {
            const uint32_t len = ctx->cfg.chunk_lengths[chunk];
            ds41f_cmd_set(&cmds[k++], DS41F_CMD_ENCODE_LAYER_CHUNK, layer, chunk, off, len, 0);
            off += len;
        }
        ds41f_cmd_set(&cmds[k++], DS41F_CMD_SWAP_CARRY, layer, UINT32_MAX, 0, ctx->plan.n_tokens, 0);
        ds41f_cmd_set(&cmds[k++], DS41F_CMD_PUBLISH_FRONTIER, layer, UINT32_MAX, 0, ctx->plan.n_tokens, ctx->cfg.frontier_masks[layer]);
        ds41f_cmd_set(&cmds[k++], DS41F_CMD_END_LAYER, layer, UINT32_MAX, 0, ctx->plan.n_tokens, 0);
    }
    ds41f_cmd_set(&cmds[k++], DS41F_CMD_ENCODE_OUTPUT_HEAD, UINT32_MAX, UINT32_MAX, ctx->plan.n_tokens - ctx->plan.max_chunk_tokens, ctx->plan.max_chunk_tokens, 0);
    ds41f_cmd_set(&cmds[k++], DS41F_CMD_READ_LOGITS, UINT32_MAX, UINT32_MAX, ctx->plan.n_tokens - ctx->plan.max_chunk_tokens, ctx->plan.max_chunk_tokens, 0);
    free(ctx->commands);
    ctx->commands = cmds;
    ctx->n_commands = k;
    return 0;
}

uint32_t ds41f_prefill_native_command_count(const ds41f_prefill_native_context *ctx) {
    return ctx ? ctx->n_commands : 0;
}

int ds41f_prefill_native_command_at(const ds41f_prefill_native_context *ctx,
                                    uint32_t index,
                                    ds41f_prefill_native_command *out) {
    if (!ctx || !out) return -1;
    if (index >= ctx->n_commands) return -2;
    *out = ctx->commands[index];
    return 0;
}

static int ds41f_prefill_native_execute_encode_chunk(ds41f_prefill_native_context *ctx,
                                                     const ds41f_prefill_native_command *cmd) {
    if (!ctx || !cmd) return -1;
    const uint64_t one_carry = ctx->plan.carry_buffer_bytes / 2ull;
    if (one_carry < 64ull) return -2;
    const uint64_t hc_dim = (uint64_t)ctx->cfg.dim * (uint64_t)ctx->cfg.hc_mult;
    const uint64_t bps = ctx->cfg.bytes_per_hidden_scalar ? ctx->cfg.bytes_per_hidden_scalar : 2u;
    const uint64_t offset = (uint64_t)cmd->token_offset * hc_dim * bps;
    if (offset >= one_carry) return -3;
    unsigned char *dst = ctx->carry[1u - ctx->current] + offset;
    const uint64_t writable = one_carry - offset;
    const uint32_t stamp = writable < 64ull ? (uint32_t)writable : 64u;
    uint64_t seed = 0x44533431504FULL; /* DS41PO */
    seed ^= ((uint64_t)cmd->layer + 1ull) * 0x9e3779b185ebca87ull;
    seed ^= ((uint64_t)cmd->chunk + 1ull) * 0xc2b2ae3d27d4eb4full;
    seed ^= ((uint64_t)cmd->token_offset << 17);
    seed ^= ((uint64_t)cmd->token_count << 33);
    for (uint32_t i = 0; i < stamp; i++) {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        dst[i] = (unsigned char)((seed * 2685821657736338717ull) >> 56);
    }
    ctx->executed_encode_chunks++;
    ctx->last_encoded_layer = cmd->layer;
    ctx->last_encoded_chunk = cmd->chunk;
    return 0;
}

int ds41f_prefill_native_execute_noop_graph(ds41f_prefill_native_context *ctx,
                                            uint32_t *submitted_commands) {
    if (!ctx) return -1;
    if (!ctx->commands || ctx->n_commands == 0) {
        int rc = ds41f_prefill_native_build_commands(ctx);
        if (rc != 0) return rc;
    }
    uint32_t submitted = 0;
    for (uint32_t i = 0; i < ctx->n_commands; i++) {
        const ds41f_prefill_native_command *cmd = &ctx->commands[i];
        if (cmd->kind == DS41F_CMD_ENCODE_LAYER_CHUNK) {
            int rc = ds41f_prefill_native_execute_encode_chunk(ctx, cmd);
            if (rc != 0) return rc;
        } else if (cmd->kind == DS41F_CMD_SWAP_CARRY) {
            int rc = ds41f_prefill_native_swap_carry(ctx);
            if (rc != 0) return rc;
        }
        submitted++;
    }
    if (submitted_commands) *submitted_commands = submitted;
    return 0;
}
