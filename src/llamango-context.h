#pragma once

#include "llamango.h"

void llamango_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata);
