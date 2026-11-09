#pragma once

#include "dataset-types.h"

#include <string>

#include "llama.h"

bool tokenize_rendered_example(const llama_vocab * vocab,
                               const rendered_example & example,
                               training_sample & output,
                               std::string & error);
