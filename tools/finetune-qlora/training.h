#pragma once

#include "adapter.h"
#include "args.h"
#include "checkpoint.h"
#include "common.h"
#include "llama.h"

#include <string>

enum ggml_type qlora_type_from_name(const std::string & name);

bool train_adapter(const qlora_params & qparams,
                   common_params & params,
                   llama_model * model,
                   llama_context * ctx,
                   const lora_tensor_refs & output_refs,
                   const std::string & arch,
                   float lora_alpha,
                   qlora_checkpoint * resume);
