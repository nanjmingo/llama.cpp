#include "dataset-reader.h"

#include "dataset-format.h"
#include "dataset-thread-pool.h"

#include <exception>
#include <fstream>
#include <memory>
#include <optional>
#include <utility>

namespace {

struct parsed_line {
    std::optional<dataset_example> example;
    std::string error;
    size_t record_number = 0;
};

parsed_line parse_line(const std::pair<size_t, std::string> & line) {
    parsed_line result;
    result.record_number = line.first;
    try {
        const nlohmann::json data = nlohmann::json::parse(line.second);
        dataset_example example;
        if (parse_dataset_record(data, example, result.error)) {
            result.example = std::move(example);
        }
    } catch (const std::exception & exception) {
        result.error = std::string("invalid JSON record: ") + exception.what();
    }
    return result;
}

std::vector<parsed_line> parse_lines_serial(
        const std::vector<std::pair<size_t, std::string>> & lines) {
    std::vector<parsed_line> result;
    result.reserve(lines.size());
    for (const auto & line : lines) {
        result.push_back(parse_line(line));
    }
    return result;
}

bool collect_batch(const std::string & path,
                   std::vector<parsed_line> & parsed,
                   const dataset_options & options,
                   dataset_read_result & result) {
    for (parsed_line & item : parsed) {
        if (!item.example) {
            ++result.stats.invalid_records;
            result.diagnostics.push_back({path, item.record_number,
                item.error.empty() ? "invalid dataset record" : std::move(item.error)});
            if (!options.skip_invalid) {
                return false;
            }
            continue;
        }
        result.examples.push_back({std::move(*item.example), path, item.record_number});
        ++result.stats.records_accepted;
    }
    return true;
}

void append_stats(dataset_stats & target, const dataset_stats & source) {
    target.records_read += source.records_read;
    target.records_accepted += source.records_accepted;
    target.invalid_records += source.invalid_records;
}

}

dataset_read_summary read_jsonl_dataset_batched(const std::string & path,
                                                const dataset_options & options,
                                                const dataset_batch_callback & consume_batch,
                                                dataset_thread_pool * shared_pool) {
    dataset_read_summary summary;
    std::ifstream input(path);
    if (!input) {
        dataset_read_result batch;
        batch.diagnostics.push_back({path, 0, "failed to open JSONL file"});
        consume_batch(std::move(batch));
        return summary;
    }

    std::vector<std::pair<size_t, std::string>> lines;
    lines.reserve(dataset_batch_size);
    std::unique_ptr<dataset_thread_pool> owned_pool;
    dataset_thread_pool * pool = shared_pool;
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty()) {
            lines.emplace_back(line_number, std::move(line));
        }
        line.clear();
        if (lines.size() >= dataset_batch_size) {
            dataset_read_result batch;
            batch.stats.records_read = lines.size();
            if (!pool) {
                owned_pool = std::make_unique<dataset_thread_pool>(dataset_thread_count(options.threads, lines.size()));
                pool = owned_pool.get();
            }
            {
                std::vector<parsed_line> parsed(lines.size());
                pool->parallel_for(lines.size(), [&](size_t index) {
                    parsed[index] = parse_line(lines[index]);
                });
                batch.success = collect_batch(path, parsed, options, batch);
            }
            const bool batch_success = batch.success;
            append_stats(summary.stats, batch.stats);
            lines.clear();
            consume_batch(std::move(batch));
            if (!batch_success) {
                summary.success = false;
                return summary;
            }
            summary.success = true;
        }
    }
    if (input.bad()) {
        dataset_read_result batch;
        batch.diagnostics.push_back({path, line_number, "failed while reading JSONL file"});
        consume_batch(std::move(batch));
        summary.success = false;
        return summary;
    }
    if (!lines.empty()) {
        dataset_read_result batch;
        batch.stats.records_read = lines.size();
        {
            std::vector<parsed_line> parsed;
            if (pool) {
                parsed.resize(lines.size());
                pool->parallel_for(lines.size(), [&](size_t index) {
                    parsed[index] = parse_line(lines[index]);
                });
            } else {
                parsed = parse_lines_serial(lines);
            }
            batch.success = collect_batch(path, parsed, options, batch);
        }
        append_stats(summary.stats, batch.stats);
        const bool batch_success = batch.success;
        lines.clear();
        consume_batch(std::move(batch));
        if (!batch_success) {
            summary.success = false;
            return summary;
        }
    }
    summary.success = true;
    return summary;
}

dataset_read_result read_jsonl_dataset(const std::string & path, const dataset_options & options) {
    dataset_read_result result;
    result.success = true;
    const dataset_read_summary summary = read_jsonl_dataset_batched(path, options, [&](dataset_read_result && batch) {
        result.stats.records_read += batch.stats.records_read;
        result.stats.records_accepted += batch.stats.records_accepted;
        result.stats.invalid_records += batch.stats.invalid_records;
        result.examples.insert(result.examples.end(),
                               std::make_move_iterator(batch.examples.begin()),
                               std::make_move_iterator(batch.examples.end()));
        result.diagnostics.insert(result.diagnostics.end(),
                                  std::make_move_iterator(batch.diagnostics.begin()),
                                  std::make_move_iterator(batch.diagnostics.end()));
        result.success = result.success && batch.success;
    });
    result.success = summary.success;
    result.stats = summary.stats;
    return result;
}
