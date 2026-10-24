#include "dataset-reader.h"

#include <iterator>
#include <string>
#include <utility>

namespace {

void append_stats(dataset_stats & target, const dataset_stats & source) {
    target.records_read += source.records_read;
    target.records_accepted += source.records_accepted;
    target.invalid_records += source.invalid_records;
    target.rendered_samples += source.rendered_samples;
    target.training_windows += source.training_windows;
}

void append_result(dataset_read_result & target, dataset_read_result && source) {
    append_stats(target.stats, source.stats);
    target.examples.insert(target.examples.end(),
                           std::make_move_iterator(source.examples.begin()),
                           std::make_move_iterator(source.examples.end()));
    target.diagnostics.insert(target.diagnostics.end(),
                              std::make_move_iterator(source.diagnostics.begin()),
                              std::make_move_iterator(source.diagnostics.end()));
    target.success = target.success && source.success;
}

}

dataset_read_summary read_dataset_records_batched(const dataset_options & options,
                                                   const dataset_batch_callback & consume_batch,
                                                   dataset_thread_pool * pool) {
    dataset_read_summary summary;
    summary.success = !options.input_paths.empty();
    if (options.input_paths.empty()) {
        dataset_read_result batch;
        batch.diagnostics.push_back({{}, 0, "no dataset input files were specified"});
        consume_batch(std::move(batch));
        summary.success = false;
        return summary;
    }

    for (const std::string & path : options.input_paths) {

        dataset_read_summary current = read_jsonl_dataset_batched(path, options, consume_batch, pool);
        append_stats(summary.stats, current.stats);
        if (!current.success) {
            summary.success = false;
            break;
        }
    }
    return summary;
}

dataset_read_result read_dataset_records(const dataset_options & options) {
    dataset_read_result result;
    result.success = true;
    const dataset_read_summary summary = read_dataset_records_batched(options, [&](dataset_read_result && batch) {
        append_result(result, std::move(batch));
    });
    result.success = summary.success;
    result.stats = summary.stats;
    return result;
}
