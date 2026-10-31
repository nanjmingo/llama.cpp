#include "dataset-window.h"

#include <algorithm>
#include <cmath>
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

bool dataset_matches_windows(ggml_opt_dataset_t dataset,
                             const std::vector<training_window> & windows,
                             const window_options & options) {
    if (!dataset || ggml_opt_dataset_ndata(dataset) != static_cast<int64_t>(windows.size())) {
        return false;
    }

    const auto * data = static_cast<const int32_t *>(ggml_opt_dataset_data(dataset)->data);
    const auto * labels = static_cast<const int32_t *>(ggml_opt_dataset_labels(dataset)->data);
    const auto * weights_tensor = ggml_opt_dataset_weights(dataset);
    const auto * weights = weights_tensor ? static_cast<const float *>(weights_tensor->data) : nullptr;
    if (!weights) {
        return false;
    }
    std::vector<int64_t> permutation(windows.size());
    bool matches = true;
    for (size_t i = 0; i < windows.size(); ++i) {
        permutation[i] = static_cast<int64_t>(windows.size() - i - 1);
        const size_t offset = i * static_cast<size_t>(options.context_length);
        for (int32_t j = 0; j < options.context_length; ++j) {
            const float stored_weight = weights[offset + j];
            if (data[offset + j] != windows[i].data[j] || labels[offset + j] != windows[i].labels[j] ||
                std::fabs(stored_weight - windows[i].weights[j]) > 1e-7f) {
                matches = false;
                break;
            }
        }
    }
    matches = matches && ggml_opt_dataset_set_permutation(dataset, permutation.data(),
            static_cast<int64_t>(permutation.size()));
    for (size_t i = 0; i < windows.size() && matches; ++i) {
        const training_window & expected = windows[static_cast<size_t>(permutation[i])];
        std::vector<int32_t> batch_data(static_cast<size_t>(options.context_length));
        std::vector<int32_t> batch_labels(static_cast<size_t>(options.context_length));
        std::vector<float> batch_weights(static_cast<size_t>(options.context_length));
        ggml_opt_dataset_get_batch_host_with_weights(dataset,
                batch_data.data(), batch_data.size() * sizeof(int32_t),
                batch_labels.data(), batch_weights.data(), batch_weights.size() * sizeof(float), static_cast<int64_t>(i));
        for (int32_t j = 0; j < options.context_length; ++j) {
            if (batch_data[j] != expected.data[j] || batch_labels[j] != expected.labels[j] ||
                std::fabs(batch_weights[j] - expected.weights[j]) > 1e-7f) {
                matches = false;
                break;
            }
        }
    }
    return matches;
}

bool packed_matches_windows(const std::vector<training_sample> & samples, const window_options & options) {
    const std::vector<training_window> windows = make_training_windows(samples, options);
    ggml_opt_dataset_t dataset = build_training_dataset(samples, options);
    if (windows.empty()) {
        if (dataset) {
            ggml_opt_dataset_free(dataset);
        }
        return dataset == nullptr && pack_training_windows(windows, options.context_length) == nullptr;
    }

    bool matches = dataset_matches_windows(dataset, windows, options);
    if (dataset) {
        ggml_opt_dataset_free(dataset);
    }

    dataset = pack_training_windows(windows, options.context_length);
    matches = matches && dataset_matches_windows(dataset, windows, options);
    if (dataset) {
        ggml_opt_dataset_free(dataset);
    }
    return matches;
}
}

int main() {
    const std::vector<training_sample> short_samples = {make_sample(3)};
    const auto short_windows = make_training_windows(short_samples, window_options{4, 99, -1});
    require(short_windows.size() == 1, "short samples should produce a padded window");
    require(short_windows[0].data == std::vector<int32_t>({10, 11, 12, 99}), "short window should be padded with BOS");
    require(short_windows[0].labels == std::vector<int32_t>({11, 12, -1, -1}), "short window should mask padding and final token");
    require(short_windows[0].weights == std::vector<float>({2.0f, 2.0f, 0.0f, 0.0f}),
            "short window weights should normalize over supervised targets");
    const auto no_bos_windows = make_training_windows(short_samples, window_options{4, -1, -1});
    require(no_bos_windows[0].data[3] == 0, "padding should use a valid fallback token when the vocabulary has no BOS");

    const std::vector<training_sample> long_samples = {make_sample(10)};
    const auto long_windows = make_training_windows(long_samples, window_options{4, 99, -1});
    require(long_windows.size() == 4, "long samples should include a padded final window for the remaining targets");
    size_t supervised = 0;
    for (const training_window & window : long_windows) {
        require(window.data.size() == 4 && window.labels.size() == 4, "every window should have context length");
        for (int32_t label : window.labels) {
            supervised += label >= 0;
        }
    }
    require(supervised == 9, "all next-token targets should be covered, including the final sample token");
    require(long_windows.back().data == std::vector<int32_t>({16, 17, 18, 19}),
            "the final window should retain the remaining input tokens");
    require(long_windows.back().labels == std::vector<int32_t>({-1, -1, 19, -1}),
            "the padded final window should train the final sample token without training padding");
    require(long_windows[1].weights == std::vector<float>({0.0f, 0.0f, 2.0f, 2.0f}),
            "overlap windows should normalize only their non-overlapping supervised targets");

    const auto no_overlap_windows = make_training_windows(long_samples, window_options{4, 99, 0});
    require(no_overlap_windows.size() == 3, "non-overlapping windows should include the remaining tail targets");
    const auto padded_tail_windows = make_training_windows({make_sample(9)}, window_options{4, 99, 1});
    require(padded_tail_windows.size() == 3 &&
            padded_tail_windows.back().data == std::vector<int32_t>({16, 17, 18, 99}) &&
            padded_tail_windows.back().labels == std::vector<int32_t>({-1, 18, -1, -1}),
            "a partial final window should be right-padded and retain its final real target");

    training_sample masked_sample;
    masked_sample.tokens = {10, 11, 12, 13};
    masked_sample.weights = {0.0f, 0.0f, 0.25f, 0.8f};
    const auto masked_windows = make_training_windows({masked_sample}, window_options{4, 99, -1});
    require(masked_windows.size() == 1, "weighted sample should produce a window");
    require(masked_windows[0].labels == std::vector<int32_t>({-1, 12, 13, -1}),
            "zero loss weights should mask labels and positive weights should preserve them");
    require(std::fabs(masked_windows[0].weights[1] - (0.25f * 4.0f / 1.05f)) < 1e-6f &&
            std::fabs(masked_windows[0].weights[2] - (0.8f * 4.0f / 1.05f)) < 1e-6f,
            "fractional loss weights should be normalized by their total supervised weight");

    training_sample text_sample = make_sample(10);
    text_sample.text_continuation = true;
    const auto text_windows = make_training_windows({text_sample}, window_options{4, 99, -1});
    require(text_windows.size() == 4, "long text samples should produce all supervised windows");
    require(text_windows[0].labels == std::vector<int32_t>({-1, 12, 13, 14}),
            "text samples should use the first half of the first context window as prompt");
    require(std::fabs(text_windows[0].weights[1] - (4.0f / 3.0f)) < 1e-6f,
            "text continuation loss should normalize over its three supervised target tokens");

    training_sample prompt_response_sample = make_sample(10);
    prompt_response_sample.weights[1] = 0.0f;
    prompt_response_sample.weights[2] = 1.0f;
    const auto prompt_response_windows = make_training_windows(
        {prompt_response_sample}, window_options{4, 99, -1});
    require(prompt_response_windows[0].labels == std::vector<int32_t>({-1, 12, 13, 14}),
            "prompt/completion samples should use the response mask without an extra initial-prefix mask");

    training_sample delayed_response_sample = make_sample(10);
    std::fill(delayed_response_sample.weights.begin(), delayed_response_sample.weights.begin() + 7, 0.0f);
    const auto delayed_response_windows = make_training_windows(
        {delayed_response_sample}, window_options{4, 99, -1});
    require(delayed_response_windows.size() == 2 && delayed_response_windows[0].data[0] == 14 &&
            delayed_response_windows[1].data[0] == 16,
            "windows without supervised targets should be discarded while retaining the response tail");

    training_sample tail_weight_sample = make_sample(10);
    tail_weight_sample.weights[9] = 0.25f;
    const auto tail_weight_windows = make_training_windows({tail_weight_sample}, window_options{4, 99, -1});
    require(tail_weight_windows.size() == 4, "a supervised tail token should produce a final window");
    require(tail_weight_windows.back().labels == std::vector<int32_t>({-1, -1, 19, -1}) &&
            std::fabs(tail_weight_windows.back().weights[2] - 4.0f) < 1e-6f,
            "the final fractional target should be retained and normalized within its window");

    require(make_training_windows(long_samples, window_options{0, 99, -1}).empty(),
            "invalid context length should return no windows");
    require(packed_matches_windows(short_samples, window_options{4, 99, -1}),
            "direct dataset packing should match materialized short windows");
    require(packed_matches_windows(long_samples, window_options{4, 99, -1}),
            "direct dataset packing should match materialized long windows including the tail");
    require(packed_matches_windows({masked_sample}, window_options{4, 99, -1}),
            "direct dataset packing should preserve masked labels");
    require(packed_matches_windows({text_sample}, window_options{4, 99, -1}),
            "direct dataset packing should preserve text continuation masks");
    require(packed_matches_windows({tail_weight_sample}, window_options{4, 99, -1}),
            "direct dataset packing should ignore fractional weights in discarded tails");
    require(packed_matches_windows(long_samples, window_options{0, 99, -1}),
            "direct dataset packing should reject invalid context lengths");
    return 0;
}
