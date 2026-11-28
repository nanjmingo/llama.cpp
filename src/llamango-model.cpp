#include "llamango-model.h"

ggml_prec llamango_prec_policy_resolve(ggml_prec requested, ggml_type weight_type) {
    if (requested == GGML_PREC_Q8 && !ggml_is_quantized(weight_type)) {
        return GGML_PREC_F16;
    }

    return requested;
}
