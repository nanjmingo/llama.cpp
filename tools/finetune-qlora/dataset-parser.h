#pragma once

#include "dataset-types.h"

#include <string>

#include "nlohmann/json.hpp"

bool parse_dataset_record(const nlohmann::json & input,
                          dataset_example & output,
                          std::string & error);
