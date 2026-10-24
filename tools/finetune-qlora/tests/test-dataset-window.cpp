#include "dataset-window.h"

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

training_sample make_sample(size_t size) {
    training_sample sample;
    for (size_t i = 0; i < size; ++i) {
        sample.tokens.push_back(static_cast<llama_token>(i + 10));
        sample.weights.push_back(1.0f);
    }
    return sample;
}

bool packed_matches_windows(const std::vector<training_sample> & samples, const window_options & options) {
    const std::vector<training_window> windows = make_training_windows(samples, options);
    ggml_opt_dataset_t dataset = build_training_dataset(samples, options);
    if (windows.empty()) {
        return dataset == nullptr;
    }
    if (!dataset || ggml_opt_dataset_ndata(dataset) != static_cast<int64_t>(windows.size())) {
        if (dataset) {
            ggml_opt_dataset_free(dataset);
        }
        return false;
    }

    const auto * data = static_cast<const int32_t *>(ggml_opt_dataset_data(dataset)->data);
    const auto * labels = static_cast<const int32_t *>(ggml_opt_dataset_labels(dataset)->data);
    bool matches = true;
    for (size_t i = 0; i < windows.size() && matches; ++i) {
        const size_t offset = i * static_cast<size_t>(options.context_length);
        for (int32_t j = 0; j < options.context_length; ++j) {
            if (data[offset + j] != windows[i].data[j] || labels[offset + j] != windows[i].labels[j]) {
                matches = false;
                break;
            }
        }
    }
    ggml_opt_dataset_free(dataset);
    return matches;
}

}

int main() {
    const std::vector<training_sample> short_samples = {make_sample(3)};
    const auto short_windows = make_training_windows(short_samples, window_options{4, 99, -1});
    require(short_windows.size() == 1, "short samples should produce a padded window");
    require(short_windows[0].data == std::vector<int32_t>({10, 11, 12, 99}), "short window should be padded with BOS");
    require(short_windows[0].labels == std::vector<int32_t>({11, 12, -1, -1}), "short window should mask padding and final token");
    const auto no_bos_windows = make_training_windows(short_samples, window_options{4, -1, -1});
    require(no_bos_windows[0].data[3] == 0, "padding should use a valid fallback token when the vocabulary has no BOS");

    const std::vector<training_sample> long_samples = {make_sample(10)};
    const auto long_windows = make_training_windows(long_samples, window_options{4, 99, -1});
    require(long_windows.size() == 4, "long sample should cover complete and tail windows");
    size_t supervised = 0;
    for (const training_window & window : long_windows) {
        require(window.data.size() == 4 && window.labels.size() == 4, "every window should have context length");
        for (int32_t label : window.labels) {
            supervised += label >= 0;
        }
    }
    require(supervised == 9, "every next-token target should be supervised exactly once");

    training_sample masked_sample;
    masked_sample.tokens = {10, 11, 12};
    masked_sample.weights = {0.0f, 0.0f, 1.0f};
    const auto masked_windows = make_training_windows({masked_sample}, window_options{4, 99, -1});
    require(masked_windows.size() == 1, "weighted sample should produce a window");
    require(masked_windows[0].labels == std::vector<int32_t>({-1, 12, -1, -1}),
            "loss weights should mask prompt tokens and supervise the selected target");

    require(make_training_windows(long_samples, window_options{0, 99, -1}).empty(),
            "invalid context length should return no windows");
    require(packed_matches_windows(short_samples, window_options{4, 99, -1}),
            "direct dataset packing should match materialized short windows");
    require(packed_matches_windows(long_samples, window_options{4, 99, -1}),
            "direct dataset packing should match materialized long windows");
    require(packed_matches_windows({masked_sample}, window_options{4, 99, -1}),
            "direct dataset packing should preserve masked labels");
    require(packed_matches_windows(long_samples, window_options{0, 99, -1}),
            "direct dataset packing should reject invalid context lengths");
    return 0;
}
