#ifndef LLAMANGO_H
#define LLAMANGO_H

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// QLoRA-oriented extensions to the upstream optimizer API.
LLAMA_API void llamango_opt_set_optimizer_state_type(struct llama_context * lctx, enum ggml_type type);
LLAMA_API void llamango_opt_set_seed(struct llama_context * lctx, uint32_t seed);
LLAMA_API void llamango_opt_set_opt_period(struct llama_context * lctx, int32_t opt_period);
LLAMA_API void llamango_opt_set_activation_checkpointing(struct llama_context * lctx, bool enabled);
LLAMA_API size_t llamango_opt_state_size(const struct llama_context * lctx);
LLAMA_API bool llamango_opt_state_export(const struct llama_context * lctx, void * data, size_t size);
LLAMA_API bool llamango_opt_state_import(struct llama_context * lctx, const void * data, size_t size);
LLAMA_API void llamango_opt_dataset_shuffle(
        struct llama_context * lctx,
        ggml_opt_dataset_t     dataset,
        int64_t                idata);
LLAMA_API void llamango_opt_epoch_range(
        struct llama_context    * lctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_start,
        int64_t                   idata_split,
        int64_t                   idata_eval_end,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval);
LLAMA_API bool llamango_opt_epoch_range_ex(
        struct llama_context    * lctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_start,
        int64_t                   idata_split,
        int64_t                   idata_eval_end,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval,
        ggml_abort_callback       abort_callback,
        void                    * abort_callback_data);

#ifdef __cplusplus
}
#endif

#endif // LLAMANGO_H
