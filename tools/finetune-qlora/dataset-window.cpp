#include "dataset-window.h"

#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace {

struct window_statistics {
    int64_t token_count = 0;
    int64_t supervised_labels = 0;
    int64_t padded_tokens = 0;
};

void fill_window(const training_sample & sample,
                 size_t offset,
                 int32_t context_length,
                 int32_t overlap,
                 llama_token padding_token,
                 int32_t * data,
                 int32_t * labels,
                 float * weights,
                 window_statistics & statistics) {
    const size_t sample_size = sample.tokens.size();
    double supervised_weight_sum = 0.0;
    for (int32_t i = 0; i < context_length; ++i) {
        const size_t sample_index = offset + static_cast<size_t>(i);
        if (sample_index < sample_size) {
            data[i] = static_cast<int32_t>(sample.tokens[sample_index]);
        } else {
            data[i] = static_cast<int32_t>(padding_token);
            ++statistics.padded_tokens;
        }

        if (sample_index + 1 >= sample_size || (offset > 0 && i < overlap)) {
            labels[i] = -1;
            if (weights) {
                weights[i] = 0.0f;
            }
            continue;
        }

        const size_t target_index = sample_index + 1;
        const bool initial_text_context = sample.text_continuation && offset == 0 &&
            target_index < static_cast<size_t>(context_length / 2);
        const float weight = initial_text_context ? 0.0f : sample.weights[target_index];
        if (weights) {
            weights[i] = weight;
        }
        if (weight > 0.0f) {
            labels[i] = static_cast<int32_t>(sample.tokens[target_index]);
            ++statistics.supervised_labels;
            supervised_weight_sum += weight;
        } else {
            labels[i] = -1;
        }
    }

    if (weights && supervised_weight_sum > 0.0) {
        const float scale = static_cast<float>(context_length / supervised_weight_sum);
        for (int32_t i = 0; i < context_length; ++i) {
            weights[i] *= scale;
        }
    }
}

bool valid_sample_weights(const training_sample & sample) {
    return sample.weights.size() == sample.tokens.size() &&
        std::all_of(sample.weights.begin(), sample.weights.end(), [](float weight) {
            return std::isfinite(weight) && weight >= 0.0f && weight <= 1.0f;
        });
}

size_t get_window_offset_limit(size_t sample_size, int32_t context_length, int32_t overlap) {
    return sample_size <= static_cast<size_t>(context_length)
        ? 1
        : sample_size - static_cast<size_t>(overlap) - 1;
}

size_t get_retained_target_count(size_t sample_size, int32_t context_length,
                                 int32_t overlap, int32_t stride) {
    const size_t offset_limit = get_window_offset_limit(sample_size, context_length, overlap);
    const size_t window_count = 1 + (offset_limit - 1) / static_cast<size_t>(stride);
    const size_t target_count = static_cast<size_t>(context_length) +
        (window_count - 1) * static_cast<size_t>(stride);
    return std::min(sample_size - 1, target_count);
}

size_t count_window_supervised_labels(const training_sample & sample,
                                      size_t offset,
                                      int32_t context_length,
                                      int32_t overlap) {
    size_t count = 0;
    for (int32_t i = 0; i < context_length; ++i) {
        const size_t sample_index = offset + static_cast<size_t>(i);
        const size_t target_index = sample_index + 1;
        if (target_index >= sample.tokens.size() || (offset > 0 && i < overlap)) {
            continue;
        }
        if (sample.text_continuation && offset == 0 &&
            target_index < static_cast<size_t>(context_length / 2)) {
            continue;
        }
        if (sample.weights[target_index] > 0.0f) {
            ++count;
        }
    }
    return count;
}

}

std::vector<training_window> make_training_windows(const std::vector<training_sample> & samples,
                                                   const window_options & options) {
    const int32_t context_length = options.context_length;
    const llama_token padding_token = options.bos_token >= 0 ? options.bos_token : 0;
    std::vector<training_window> windows;
    if (context_length <= 0) {
        LOG_ERR("%s: context length must be greater than zero\n", __func__);
        return windows;
    }

    const int32_t overlap = options.overlap_tokens < 0
        ? context_length / 2 : std::clamp(options.overlap_tokens, 0, context_length - 1);
    const int32_t stride = context_length - overlap;
    window_statistics statistics;

    for (const training_sample & sample : samples) {
        const size_t sample_size = sample.tokens.size();
        if (sample_size <= 1) {
            continue;
        }
        if (!valid_sample_weights(sample)) {
            LOG_WRN("%s: skipping sample with invalid token weights or mismatched token and weight counts (%zu vs %zu)\n",
                    __func__, sample_size, sample.weights.size());
            continue;
        }
        const size_t offset_limit = get_window_offset_limit(sample_size, context_length, overlap);
        statistics.token_count += static_cast<int64_t>(get_retained_target_count(sample_size, context_length, overlap, stride));

        for (size_t offset = 0; offset < offset_limit;) {
            training_window window;
            window.data.resize(static_cast<size_t>(context_length));
            window.labels.resize(static_cast<size_t>(context_length));
            window.weights.resize(static_cast<size_t>(context_length));
            const int64_t supervised_before = statistics.supervised_labels;
            fill_window(sample, offset, context_length, overlap, padding_token,
                        window.data.data(), window.labels.data(), window.weights.data(), statistics);
            if (statistics.supervised_labels > supervised_before) {
                windows.push_back(std::move(window));
            }
            if (offset_limit - offset <= static_cast<size_t>(stride)) {
                break;
            }
            offset += static_cast<size_t>(stride);
        }
    }

    LOG_INF("%s: prepared %ld tokens into %zu windows (ctx=%d stride=%d, supervised labels=%ld, padded=%ld)\n",
            __func__, static_cast<long>(statistics.token_count), windows.size(), context_length, stride,
            static_cast<long>(statistics.supervised_labels), static_cast<long>(statistics.padded_tokens));
    return windows;
}

ggml_opt_dataset_t pack_training_windows(const std::vector<training_window> & windows, int32_t context_length) {
    if (context_length <= 0 || windows.empty()) {
        LOG_ERR("%s: no training windows or invalid context length\n", __func__);
        return nullptr;
    }
    for (size_t i = 0; i < windows.size(); ++i) {
        if (windows[i].data.size() != static_cast<size_t>(context_length) ||
            windows[i].labels.size() != static_cast<size_t>(context_length) ||
            windows[i].weights.size() != static_cast<size_t>(context_length) ||
            !std::all_of(windows[i].weights.begin(), windows[i].weights.end(), [](float weight) {
                return std::isfinite(weight) && weight >= 0.0f;
            })) {
            LOG_ERR("%s: window %zu does not match context length %d\n", __func__, i, context_length);
            return nullptr;
        }
    }

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init_with_weights(
        GGML_TYPE_I32, GGML_TYPE_I32, GGML_TYPE_F32,
        context_length, context_length, context_length,
        static_cast<int64_t>(windows.size()), 1);
    if (!dataset) {
        LOG_ERR("%s: failed to allocate optimization dataset\n", __func__);
        return nullptr;
    }

    auto * data = static_cast<int32_t *>(ggml_opt_dataset_data(dataset)->data);
    auto * labels = static_cast<int32_t *>(ggml_opt_dataset_labels(dataset)->data);
    auto * weights_tensor = ggml_opt_dataset_weights(dataset);
    auto * weights = weights_tensor ? static_cast<float *>(weights_tensor->data) : nullptr;
    for (size_t i = 0; i < windows.size(); ++i) {
        std::memcpy(data + i * context_length, windows[i].data.data(),
                    static_cast<size_t>(context_length) * sizeof(int32_t));
        std::memcpy(labels + i * context_length, windows[i].labels.data(),
                    static_cast<size_t>(context_length) * sizeof(int32_t));
        if (weights) {
            std::memcpy(weights + i * context_length, windows[i].weights.data(),
                        static_cast<size_t>(context_length) * sizeof(float));
        }
    }
    LOG_INF("%s: packed %zu windows into ggml_opt_dataset_t (ctx=%d)\n",
            __func__, windows.size(), context_length);
    return dataset;
}

ggml_opt_dataset_t build_training_dataset(const std::vector<training_sample> & samples,
                                          const window_options & options) {
    const int32_t context_length = options.context_length;
    if (context_length <= 0) {
        LOG_ERR("%s: context length must be greater than zero\n", __func__);
        return nullptr;
    }

    const int32_t overlap = options.overlap_tokens < 0
        ? context_length / 2 : std::clamp(options.overlap_tokens, 0, context_length - 1);
    const int32_t stride = context_length - overlap;
    size_t window_count = 0;
    window_statistics statistics;
    for (const training_sample & sample : samples) {
        const size_t sample_size = sample.tokens.size();
        if (sample_size <= 1) {
            continue;
        }
        if (!valid_sample_weights(sample)) {
            LOG_WRN("%s: skipping sample with invalid token weights or mismatched token and weight counts (%zu vs %zu)\n",
                    __func__, sample_size, sample.weights.size());
            continue;
        }
        const size_t offset_limit = get_window_offset_limit(sample_size, context_length, overlap);
        statistics.token_count += static_cast<int64_t>(get_retained_target_count(sample_size, context_length, overlap, stride));
        for (size_t offset = 0; offset < offset_limit;) {
            if (count_window_supervised_labels(sample, offset, context_length, overlap) > 0) {
                if (window_count >= static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
                    LOG_ERR("%s: number of training windows exceeds the supported range\n", __func__);
                    return nullptr;
                }
                ++window_count;
            }
            if (offset_limit - offset <= static_cast<size_t>(stride)) {
                break;
            }
            offset += static_cast<size_t>(stride);
        }
    }

    if (window_count == 0) {
        LOG_ERR("%s: no training windows or invalid context length\n", __func__);
        return nullptr;
    }

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init_with_weights(
        GGML_TYPE_I32, GGML_TYPE_I32, GGML_TYPE_F32,
        context_length, context_length, context_length,
        static_cast<int64_t>(window_count), 1);
    if (!dataset) {
        LOG_ERR("%s: failed to allocate optimization dataset\n", __func__);
        return nullptr;
    }

    auto * data = static_cast<int32_t *>(ggml_opt_dataset_data(dataset)->data);
    auto * labels = static_cast<int32_t *>(ggml_opt_dataset_labels(dataset)->data);
    auto * weights = static_cast<float *>(ggml_opt_dataset_weights(dataset)->data);
    const llama_token padding_token = options.bos_token >= 0 ? options.bos_token : 0;
    size_t window_index = 0;

    for (const training_sample & sample : samples) {
        const size_t sample_size = sample.tokens.size();
        if (sample_size <= 1 || !valid_sample_weights(sample)) {
            continue;
        }
        const size_t offset_limit = get_window_offset_limit(sample_size, context_length, overlap);

        for (size_t offset = 0; offset < offset_limit;) {
            if (count_window_supervised_labels(sample, offset, context_length, overlap) == 0) {
                if (offset_limit - offset <= static_cast<size_t>(stride)) {
                    break;
                }
                offset += static_cast<size_t>(stride);
                continue;
            }
            const size_t base = window_index * static_cast<size_t>(context_length);
            fill_window(sample, offset, context_length, overlap, padding_token,
                        data + base, labels + base, weights + base, statistics);
            ++window_index;
            if (offset_limit - offset <= static_cast<size_t>(stride)) {
                break;
            }
            offset += static_cast<size_t>(stride);
        }
    }

    GGML_ASSERT(window_index == window_count);

    LOG_INF("%s: prepared %ld tokens into %zu windows (ctx=%d stride=%d, supervised labels=%ld, padded=%ld)\n",
            __func__, static_cast<long>(statistics.token_count), window_count, context_length, stride,
            static_cast<long>(statistics.supervised_labels), static_cast<long>(statistics.padded_tokens));
    LOG_INF("%s: packed %zu windows into ggml_opt_dataset_t (ctx=%d)\n",
            __func__, window_count, context_length);
    return dataset;
}
