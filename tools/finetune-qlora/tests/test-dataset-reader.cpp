#include "dataset-reader.h"
#include "dataset-thread-pool.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <variant>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

}

int main() {
    const std::string path = "test-dataset-reader.jsonl";
    {
        std::ofstream output(path);
        output << R"({"prompt":"q0","completion":"a0"})" << '\n';
        output << R"({"prompt":"q1","completion":"a1"})" << '\n';
        output << "{invalid json" << '\n';
        output << R"({"text":"t3"})" << '\n';
        output << R"({"prompt":"q4","completion":"a4"})" << '\n';
        for (size_t i = 5; i < 2050; ++i) {
            output << R"({"prompt":"q","completion":"a"})" << '\n';
        }
    }

    dataset_options serial_options;
    serial_options.input_paths = {path};
    serial_options.threads = 1;
    const dataset_read_result serial = read_jsonl_dataset(path, serial_options);
    require(serial.success, "serial reader should complete when invalid records are skipped");
    require(serial.stats.records_read == 2050, "all non-empty records should be counted");
    require(serial.stats.records_accepted == 2049, "valid records should be accepted");
    require(serial.stats.invalid_records == 1, "one malformed record should be diagnosed");
    require(serial.examples.size() == 2049, "valid records should be returned");

    dataset_options parallel_options = serial_options;
    parallel_options.threads = 4;
    const dataset_read_result parallel = read_jsonl_dataset(path, parallel_options);
    require(parallel.success, "parallel reader should complete when invalid records are skipped");
    require(parallel.examples.size() == serial.examples.size(), "serial and parallel result counts should match");
    for (size_t i = 0; i < serial.examples.size(); ++i) {
        require(serial.examples[i].record_number == parallel.examples[i].record_number,
                "serial and parallel record order should match");
    }
    require(std::get<prompt_completion_example>(parallel.examples[1].example).completion == "a1",
            "prompt/completion should be preserved in parallel mode");
    require(parallel.diagnostics[0].record_number == 3, "diagnostics should retain source line numbers");

    size_t batch_count = 0;
    size_t streamed_examples = 0;
    size_t last_record_number = 0;
    dataset_thread_pool shared_pool(4);
    const dataset_read_summary streamed = read_jsonl_dataset_batched(path, parallel_options,
        [&](dataset_read_result && batch) {
            ++batch_count;
            require(batch.examples.size() <= dataset_batch_size, "streamed batches must not exceed the configured batch size");
            std::vector<size_t> callback_results(batch.examples.size());
            shared_pool.parallel_for(batch.examples.size(), [&](size_t index) {
                callback_results[index] = batch.examples[index].record_number;
            });
            for (size_t index = 0; index < batch.examples.size(); ++index) {
                const sourced_dataset_example & example = batch.examples[index];
                require(example.record_number == callback_results[index], "shared worker pool should be reusable in callbacks");
                require(example.record_number > last_record_number, "streamed records must preserve input order");
                last_record_number = example.record_number;
                ++streamed_examples;
            }
        }, &shared_pool);
    require(streamed.success, "streamed reader should complete when invalid records are skipped");
    require(streamed.stats.records_read == 2050, "streamed reader should count all non-empty records");
    require(streamed.stats.records_accepted == 2049, "streamed reader should count accepted records");
    require(streamed.stats.invalid_records == 1, "streamed reader should count malformed records");
    require(streamed_examples == 2049, "streamed reader should deliver all valid records");
    require(batch_count == (2050 + dataset_batch_size - 1) / dataset_batch_size,
            "streamed reader should emit one batch per configured batch size and a final partial batch");

    parallel_options.skip_invalid = false;
    const dataset_read_result strict = read_jsonl_dataset(path, parallel_options);
    require(!strict.success, "strict mode should fail on malformed input");


    std::remove(path.c_str());
    return 0;
}
