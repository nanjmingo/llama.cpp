#include "dataset.h"

#include "dataset-template.h"
#include "dataset-tokenize.h"
#include "dataset-thread-pool.h"
#include "log.h"

#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

struct rendered_record {
    std::vector<std::string> diagnostics;
};

struct pending_example {
    rendered_example example;
};

struct tokenized_example {
    training_sample sample;
    std::string error;
    bool success = false;
};

void log_diagnostic(const dataset_diagnostic & diagnostic, bool warning) {
    const char * path = diagnostic.source_path.empty() ? "<dataset>" : diagnostic.source_path.c_str();
    if (diagnostic.record_number > 0) {
        if (warning) {
            LOG_WRN("%s:%zu: %s\n", path, diagnostic.record_number, diagnostic.message.c_str());
        } else {
            LOG_ERR("%s:%zu: %s\n", path, diagnostic.record_number, diagnostic.message.c_str());
        }
    } else if (warning) {
        LOG_WRN("%s: %s\n", path, diagnostic.message.c_str());
    } else {
        LOG_ERR("%s: %s\n", path, diagnostic.message.c_str());
    }
}

void append_tokenized_batch(const std::vector<sourced_dataset_example> & examples,
                            const llama_vocab * vocab,
                            common_chat_templates * templates,
                            dataset_thread_pool * pool,
                            dataset_load_result & result,
                            bool & preview_logged) {
    if (examples.empty()) {
        return;
    }

    std::vector<rendered_record> records(examples.size());
    std::vector<pending_example> pending;
    std::vector<size_t> pending_record_indices;
    for (size_t record_index = 0; record_index < examples.size(); ++record_index) {
        try {
            dataset_template_result rendered = render_dataset_example(examples[record_index].example, templates);
            records[record_index].diagnostics = std::move(rendered.diagnostics);
            for (rendered_example & example : rendered.examples) {
                pending.push_back({std::move(example)});
                pending_record_indices.push_back(record_index);
            }
        } catch (const std::exception & exception) {
            records[record_index].diagnostics.push_back(
                std::string("failed to render dataset record: ") + exception.what());
        }
    }

    std::vector<tokenized_example> tokenized(pending.size());
    const auto tokenize = [&](size_t index) {
        tokenized_example & output = tokenized[index];
        output.success = tokenize_rendered_example(vocab, pending[index].example, output.sample, output.error);
        if (!output.success && output.error.empty()) {
            output.error = "rendered example produced no tokens";
        }
    };
    if (pool) {
        pool->parallel_for(pending.size(), tokenize);
    } else {
        for (size_t index = 0; index < pending.size(); ++index) {
            tokenize(index);
        }
    }

    pending.clear();
    for (size_t index = 0; index < tokenized.size(); ++index) {
        if (!tokenized[index].success) {
            records[pending_record_indices[index]].diagnostics.push_back(std::move(tokenized[index].error));
        }
    }

    size_t tokenized_index = 0;
    for (size_t record_index = 0; record_index < examples.size(); ++record_index) {
        const sourced_dataset_example & sourced = examples[record_index];
        for (const std::string & message : records[record_index].diagnostics) {
            log_diagnostic({sourced.source_path, sourced.record_number, message}, true);
        }
        while (tokenized_index < tokenized.size() && pending_record_indices[tokenized_index] == record_index) {
            tokenized_example & output = tokenized[tokenized_index++];
            if (!output.success) {
                continue;
            }
            ++result.stats.rendered_samples;
            if (!preview_logged) {
                LOG_INF("%s: first training sample contains %zu tokens\n", __func__, output.sample.tokens.size());
                preview_logged = true;
            }
            result.samples.push_back(std::move(output.sample));
        }
    }
}

}

dataset_load_result load_training_samples(const dataset_options & options,
                                          const llama_vocab * vocab,
                                          common_chat_templates * templates) {
    dataset_load_result result;
    bool preview_logged = false;
    bool batch_processing_success = true;
    std::unique_ptr<dataset_thread_pool> pool;
    if (!options.input_paths.empty()) {
        pool = std::make_unique<dataset_thread_pool>(dataset_thread_count(options.threads, dataset_batch_size));
    }

    const dataset_read_summary summary = read_dataset_records_batched(options, [&](dataset_read_result && batch) {
        for (const dataset_diagnostic & diagnostic : batch.diagnostics) {
            log_diagnostic(diagnostic, options.skip_invalid && batch.success);
        }
        if (!batch.success) {
            batch_processing_success = false;
            return;
        }
        append_tokenized_batch(batch.examples, vocab, templates, pool.get(), result, preview_logged);
    }, pool.get());

    result.stats.records_read = summary.stats.records_read;
    result.stats.records_accepted = summary.stats.records_accepted;
    result.stats.invalid_records = summary.stats.invalid_records;
    result.success = summary.success && batch_processing_success;
    if (!result.success) {
        return result;
    }

    LOG_INF("%s: read %zu records, accepted %zu, invalid %zu, created %zu training samples from %zu input files\n",
            __func__, result.stats.records_read, result.stats.records_accepted,
            result.stats.invalid_records, result.samples.size(), options.input_paths.size());
    return result;
}
