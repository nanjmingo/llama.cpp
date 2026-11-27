#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Llamango-specific FlashAttention backward precision controls.
GGML_API void ggml_flash_attn_ext_set_back_prec(
        struct ggml_tensor * a,
        enum ggml_prec       prec);

GGML_API enum ggml_prec ggml_flash_attn_ext_get_back_prec(
        const struct ggml_tensor * a);

#ifdef __cplusplus
}
#endif
