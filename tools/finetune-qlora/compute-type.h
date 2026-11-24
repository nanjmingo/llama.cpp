#pragma once

#include "ggml.h"

#include <stdexcept>
#include <string>

inline ggml_prec qlora_compute_type_from_name(const std::string & name) {
    if (name == "f32") {
        return GGML_PREC_F32;
    }
    if (name == "f16") {
        return GGML_PREC_F16;
    }
    if (name == "bf16") {
        return GGML_PREC_BF16;
    }
    if (name == "q8") {
        return GGML_PREC_Q8;
    }
    throw std::invalid_argument("invalid --compute-type: " + name);
}
