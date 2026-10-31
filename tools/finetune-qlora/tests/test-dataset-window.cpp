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
    const bool weighted = std::any_of(windows.begin(), windows.end(), [](const training_window & window) {
        return std::any_of(window.weights.begin(), window.weights.end(), [](float weight) {
            return weight > 0.0f && weight < 1.0f;
        });
    });
    if (weighted != (weights != nullptr)) {
        return false;
    }
    std::vector<int64_t> permutation(windows.size());
    bool matches = true;
    for (size_t i = 0; i < windows.size(); ++i) {
        permutation[i] = static_cast<int64_t>(windows.size() - i - 1);
        const size_t offset = i * static_cast<size_t>(options.context_length);
        for (int32_t j = 0; j < options.context_length; ++j) {
            const float stored_weight = weights ? weights[offset + j] : (labels[offset + j] >= 0 ? 1.0f : 0.0f);
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
        if (weights) {
            ggml_opt_dataset_get_batch_host_with_weights(dataset,
                    batch_data.data(), batch_data.size() * sizeof(int32_t),
                    batch_labels.data(), batch_weights.data(), batch_weights.size() * sizeof(float), static_cast<int64_t>(i));
        } else {
            ggml_opt_dataset_get_batch_host(dataset, batch_data.data(), batch_data.size() * sizeof(int32_t),
                    batch_labels.data(), static_cast<int64_t>(i));
        }
        for (int32_t j = 0; j < options.context_length; ++j) {
            if (batch_data[j] != expected.data[j] || batch_labels[j] != expected.labels[j] ||
                (weights && std::fabs(batch_weights[j] - expected.weights[j]) > 1e-7f)) {
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
    require(short_windows[0].weights == std::vector<float>({1.0f, 1.0f, 0.0f, 0.0f}),
            "short window should preserve target weights and clear padding weights");
    const auto no_bos_windows = make_training_windows(short_samples, window_options{4, -1, -1});
    require(no_bos_windows[0].data[3] == 0, "padding should use a valid fallback token when the vocabulary has no BOS");

    const std::vector<training_sample> long_samples = {make_sample(10)};
    const auto long_windows = make_training_windows(long_samples, window_options{4, 99, -1});
    require(long_windows.size() == 3, "long samples should discard an incomplete tail window");
    size_t supervised = 0;
    for (const training_window & window : long_windows) {
        require(window.data.size() == 4 && window.labels.size() == 4, "every window should have context length");
        for (int32_t label : window.labels) {
            supervised += label >= 0;
        }
    }
    require(supervised == 8, "only complete windows should contribute supervised targets");
    require(long_windows.back().data == std::vector<int32_t>({14, 15, 16, 17}),
            "the final retained window should have a complete context");
    require(long_windows.back().labels == std::vector<int32_t>({-1, -1, 17, 18}),
            "the final retained window should contribute a full stride of new targets");

    const auto no_overlap_windows = make_training_windows(long_samples, window_options{4, 99, 0});
    require(no_overlap_windows.size() == 2, "non-overlapping windows should also discard an incomplete tail");

    training_sample masked_sample;
    masked_sample.tokens = {10, 11, 12, 13};
    masked_sample.weights = {0.0f, 0.0f, 0.25f, 0.8f};
    const auto masked_windows = make_training_windows({masked_sample}, window_options{4, 99, -1});
    require(masked_windows.size() == 1, "weighted sample should produce a window");
    require(masked_windows[0].labels == std::vector<int32_t>({-1, 12, 13, -1}),
            "zero loss weights should mask labels and positive weights should preserve them");
    require(masked_windows[0].weights == std::vector<float>({0.0f, 0.25f, 0.8f, 0.0f}),
            "fractional loss weights should be preserved for each target position");

    training_sample tail_weight_sample = make_sample(10);
    tail_weight_sample.weights[9] = 0.25f;
    const auto tail_weight_windows = make_training_windows({tail_weight_sample}, window_options{4, 99, -1});
    require(tail_weight_windows.size() == 3, "fractional weights in a discarded tail should not create another window");
    require(std::all_of(tail_weight_windows.begin(), tail_weight_windows.end(), [](const training_window & window) {
        return std::none_of(window.weights.begin(), window.weights.end(), [](float weight) {
            return weight > 0.0f && weight < 1.0f;
        });
    }), "discarded tail weights should not leak into retained windows");

    require(make_training_windows(long_samples, window_options{0, 99, -1}).empty(),
            "invalid context length should return no windows");
    require(packed_matches_windows(short_samples, window_options{4, 99, -1}),
            "direct dataset packing should match materialized short windows");
    require(packed_matches_windows(long_samples, window_options{4, 99, -1}),
            "direct dataset packing should match materialized long windows");
    require(packed_matches_windows({masked_sample}, window_options{4, 99, -1}),
            "direct dataset packing should preserve masked labels");
    require(packed_matches_windows({tail_weight_sample}, window_options{4, 99, -1}),
            "direct dataset packing should ignore fractional weights in discarded tails");
    require(packed_matches_windows(long_samples, window_options{0, 99, -1}),
            "direct dataset packing should reject invalid context lengths");
    return 0;
}
