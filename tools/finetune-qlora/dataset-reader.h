#pragma once

#include "dataset-types.h"

#include <cstddef>
#include <functional>

struct dataset_read_result {
    std::vector<sourced_dataset_example> examples;
    dataset_stats stats;
    std::vector<dataset_diagnostic> diagnostics;
    bool success = false;
};

struct dataset_read_summary {
    dataset_stats stats;
    bool success = false;
};

inline constexpr size_t dataset_batch_size = 128;

class dataset_thread_pool;

using dataset_batch_callback = std::function<void(dataset_read_result &&)>;

dataset_read_result read_dataset_records(const dataset_options & options);
dataset_read_result read_jsonl_dataset(const std::string & path, const dataset_options & options);
dataset_read_summary read_dataset_records_batched(const dataset_options & options,
                                                    const dataset_batch_callback & consume_batch,
                                                    dataset_thread_pool * pool = nullptr);
dataset_read_summary read_jsonl_dataset_batched(const std::string & path,
                                                 const dataset_options & options,
                                                 const dataset_batch_callback & consume_batch,
                                                 dataset_thread_pool * pool = nullptr);
