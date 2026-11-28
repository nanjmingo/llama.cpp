#pragma once

#include "ggml.h"

ggml_prec llamango_prec_policy_resolve(ggml_prec requested, ggml_type weight_type);
