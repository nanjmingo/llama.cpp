#pragma once

#include "adapter.h"

#include <string>
#include <vector>

namespace qlora_adapter_detail {

bool lora_type_is_supported(enum ggml_type type, const std::string & name);
bool tensor_read_f32_values(const void * data, enum ggml_type type, int64_t count, std::vector<float> & values);

}
