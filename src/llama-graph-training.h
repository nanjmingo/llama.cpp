#pragma once

#include "llama-graph.h"

// Training-only input preparation lives outside llama-graph.cpp so the
// upstream graph implementation only needs small, explicit call sites.
void llm_graph_set_training_attn_mask(
        const llama_kv_cache_context * cache,
        const llm_attn_mask & mask,
        const llama_ubatch * ubatch,
        const llama_cparams & cparams,
        const llama_hparams & hparams,
        int n_swa,
        llama_swa_type swa_type,
        bool causal_training);

void llm_graph_set_training_relative_pos(
        const llama_kv_cache_context * cache,
        ggml_tensor * dst,
        const llama_ubatch * ubatch,
        uint32_t extent,
        bool training);
