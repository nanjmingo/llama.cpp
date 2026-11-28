#include "llamango-context.h"

#include <cstring>

void llamango_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (std::strcmp(tensor->name, "token_embd.weight") == 0) {
        return;
    }
    if (std::strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return;
    }
    ggml_set_param(tensor);
}
