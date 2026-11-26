#include "llama-context.h"
#include "llamango.h"

#include <algorithm>

void llama_context::llamango_opt_set_optimizer_state_type(enum ggml_type type) {
    GGML_ASSERT(!opt_ctx);
    llamango_optimizer_state_type = type;
}

size_t llama_context::llamango_opt_state_size() const {
    return opt_ctx ? ggml_opt_state_size(opt_ctx) : 0;
}

bool llama_context::llamango_opt_state_export(void * data, size_t size) const {
    return opt_ctx && ggml_opt_state_export(opt_ctx, data, size);
}

bool llama_context::llamango_opt_state_import(const void * data, size_t size) {
    return opt_ctx && ggml_opt_state_import(opt_ctx, data, size);
}

void llama_context::llamango_opt_set_seed(uint32_t seed) {
    GGML_ASSERT(opt_ctx);
    ggml_opt_set_seed(opt_ctx, seed);
}

void llama_context::llamango_opt_set_opt_period(int32_t opt_period) {
    GGML_ASSERT(opt_ctx);
    GGML_ASSERT(opt_period >= 1);
    const uint32_t n_batch = std::min(this->n_batch(), this->n_ctx());
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(n_ubatch == this->n_ctx() && "accumulation period counts full-context windows");
    ggml_opt_set_opt_period(opt_ctx, opt_period);
    opt_accumulation_period = opt_period;
}

void llama_context::llamango_opt_set_activation_checkpointing(bool enabled) {
    GGML_ASSERT(opt_ctx);
    ggml_opt_set_activation_checkpointing(opt_ctx, enabled);
}

void llama_context::llamango_opt_dataset_shuffle(ggml_opt_dataset_t dataset, int64_t idata) {
    GGML_ASSERT(opt_ctx);
    ggml_opt_dataset_shuffle(opt_ctx, dataset, idata);
}

void llamango_opt_set_optimizer_state_type(struct llama_context * ctx, enum ggml_type type) {
    ctx->llamango_opt_set_optimizer_state_type(type);
}

void llamango_opt_set_seed(struct llama_context * ctx, uint32_t seed) {
    ctx->llamango_opt_set_seed(seed);
}

void llamango_opt_set_opt_period(struct llama_context * ctx, int32_t opt_period) {
    ctx->llamango_opt_set_opt_period(opt_period);
}

void llamango_opt_set_activation_checkpointing(struct llama_context * ctx, bool enabled) {
    ctx->llamango_opt_set_activation_checkpointing(enabled);
}

size_t llamango_opt_state_size(const struct llama_context * ctx) {
    return ctx ? ctx->llamango_opt_state_size() : 0;
}

bool llamango_opt_state_export(const struct llama_context * ctx, void * data, size_t size) {
    return ctx && ctx->llamango_opt_state_export(data, size);
}

bool llamango_opt_state_import(struct llama_context * ctx, const void * data, size_t size) {
    return ctx && ctx->llamango_opt_state_import(data, size);
}

void llamango_opt_dataset_shuffle(
        struct llama_context * ctx,
        ggml_opt_dataset_t     dataset,
        int64_t                idata) {
    ctx->llamango_opt_dataset_shuffle(dataset, idata);
}

void llamango_opt_epoch_range(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_start,
        int64_t                   idata_split,
        int64_t                   idata_eval_end,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_start,
        idata_split,
        idata_eval_end,
        callback_train,
        callback_eval,
        nullptr,
        nullptr);
}

bool llamango_opt_epoch_range_ex(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_start,
        int64_t                   idata_split,
        int64_t                   idata_eval_end,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval,
        ggml_abort_callback       abort_callback,
        void                    * abort_callback_data) {
    return ctx && ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_start,
        idata_split,
        idata_eval_end,
        callback_train,
        callback_eval,
        abort_callback,
        abort_callback_data);
}
