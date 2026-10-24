#include "dataset-window.h"

#include "log.h"

#include <algorithm>
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
                 window_statistics & statistics) {
    const size_t sample_size = sample.tokens.size();
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
            continue;
        }

        const size_t target_index = sample_index + 1;
        if (sample.weights[target_index] > 0.0f) {
            labels[i] = static_cast<int32_t>(sample.tokens[target_index]);
            ++statistics.supervised_labels;
        } else {
            labels[i] = -1;
        }
    }
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
        if (sample.weights.size() != sample_size) {
            LOG_WRN("%s: skipping sample with mismatched token and weight counts (%zu vs %zu)\n",
                    __func__, sample_size, sample.weights.size());
            continue;
        }
        statistics.token_count += static_cast<int64_t>(sample_size - 1);

        for (size_t offset = 0; offset < sample_size - 1; offset += static_cast<size_t>(stride)) {
            training_window window;
            window.data.resize(static_cast<size_t>(context_length));
            window.labels.resize(static_cast<size_t>(context_length));
            fill_window(sample, offset, context_length, overlap, padding_token,
                        window.data.data(), window.labels.data(), statistics);
            windows.push_back(std::move(window));
            if (offset + static_cast<size_t>(context_length) >= sample_size - 1) {
                break;
            }
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
            windows[i].labels.size() != static_cast<size_t>(context_length)) {
            LOG_ERR("%s: window %zu does not match context length %d\n", __func__, i, context_length);
            return nullptr;
        }
    }

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
        GGML_TYPE_I32, GGML_TYPE_I32, context_length, context_length,
        static_cast<int64_t>(windows.size()), 1);
    if (!dataset) {
        LOG_ERR("%s: failed to allocate optimization dataset\n", __func__);
        return nullptr;
    }

    auto * data = static_cast<int32_t *>(ggml_opt_dataset_data(dataset)->data);
    auto * labels = static_cast<int32_t *>(ggml_opt_dataset_labels(dataset)->data);
    for (size_t i = 0; i < windows.size(); ++i) {
        std::memcpy(data + i * context_length, windows[i].data.data(),
                    static_cast<size_t>(context_length) * sizeof(int32_t));
        std::memcpy(labels + i * context_length, windows[i].labels.data(),
                    static_cast<size_t>(context_length) * sizeof(int32_t));
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
    for (const training_sample & sample : samples) {
        const size_t sample_size = sample.tokens.size();
        if (sample_size <= 1) {
            continue;
        }
        if (sample.weights.size() != sample_size) {
            LOG_WRN("%s: skipping sample with mismatched token and weight counts (%zu vs %zu)\n",
                    __func__, sample_size, sample.weights.size());
            continue;
        }
        for (size_t offset = 0; offset < sample_size - 1; offset += static_cast<size_t>(stride)) {
            if (window_count == static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
                LOG_ERR("%s: number of training windows exceeds the supported range\n", __func__);
                return nullptr;
            }
            ++window_count;
            if (offset + static_cast<size_t>(context_length) >= sample_size - 1) {
                break;
            }
        }
    }

    if (window_count == 0) {
        LOG_ERR("%s: no training windows or invalid context length\n", __func__);
        return nullptr;
    }

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
        GGML_TYPE_I32, GGML_TYPE_I32, context_length, context_length,
        static_cast<int64_t>(window_count), 1);
    if (!dataset) {
        LOG_ERR("%s: failed to allocate optimization dataset\n", __func__);
        return nullptr;
    }

    auto * data = static_cast<int32_t *>(ggml_opt_dataset_data(dataset)->data);
    auto * labels = static_cast<int32_t *>(ggml_opt_dataset_labels(dataset)->data);
    const llama_token padding_token = options.bos_token >= 0 ? options.bos_token : 0;
    window_statistics statistics;
    size_t window_index = 0;

    for (const training_sample & sample : samples) {
        const size_t sample_size = sample.tokens.size();
        if (sample_size <= 1 || sample.weights.size() != sample_size) {
            continue;
        }
        statistics.token_count += static_cast<int64_t>(sample_size - 1);

        for (size_t offset = 0; offset < sample_size - 1; offset += static_cast<size_t>(stride)) {
            const size_t base = window_index * static_cast<size_t>(context_length);
            fill_window(sample, offset, context_length, overlap, padding_token,
                        data + base, labels + base, statistics);
            ++window_index;
            if (offset + static_cast<size_t>(context_length) >= sample_size - 1) {
                break;
            }
        }
    }

    LOG_INF("%s: prepared %ld tokens into %zu windows (ctx=%d stride=%d, supervised labels=%ld, padded=%ld)\n",
            __func__, static_cast<long>(statistics.token_count), window_count, context_length, stride,
            static_cast<long>(statistics.supervised_labels), static_cast<long>(statistics.padded_tokens));
    LOG_INF("%s: packed %zu windows into ggml_opt_dataset_t (ctx=%d)\n",
            __func__, window_count, context_length);
    return dataset;
}
