#include "dataset-tokenize.h"

#include "common.h"

#include <algorithm>
#include <cmath>
#include <exception>

bool tokenize_rendered_example(const llama_vocab * vocab,
                               const rendered_example & example,
                               training_sample & output,
                               std::string & error) {
    output = {};
    output.text_continuation = example.text_continuation;
    try {
        for (const rendered_piece & piece : example.pieces) {
            if (!std::isfinite(piece.loss_weight) || piece.loss_weight < 0.0f || piece.loss_weight > 1.0f) {
                error = "loss weights must be finite values between 0.0 and 1.0";
                return false;
            }
            const std::vector<llama_token> tokens = common_tokenize(
                vocab, piece.text, false, piece.parse_special);
            output.tokens.insert(output.tokens.end(), tokens.begin(), tokens.end());
            output.weights.insert(output.weights.end(), tokens.size(), piece.loss_weight);
        }

        if (example.add_special) {
            if (example.pieces.empty()) {
                error = "rendered example requests automatic special tokens but has no pieces";
                return false;
            }
            const std::vector<llama_token> special_tokens = common_tokenize(vocab, {}, true, false);
            size_t prefix_count = 0;
            const llama_token bos = llama_vocab_bos(vocab);
            if (!special_tokens.empty() && special_tokens.front() == bos) {
                prefix_count = 1;
            }

            if (prefix_count > 0 && (output.tokens.empty() || output.tokens.front() != bos)) {
                output.tokens.insert(output.tokens.begin(), special_tokens.begin(), special_tokens.begin() + 1);
                output.weights.insert(output.weights.begin(), 0.0f);
            }

            const auto suffix_begin = special_tokens.begin() + static_cast<std::ptrdiff_t>(prefix_count);
            const size_t suffix_size = static_cast<size_t>(special_tokens.end() - suffix_begin);
            const bool has_suffix = suffix_size > 0 && output.tokens.size() >= suffix_size &&
                std::equal(suffix_begin, special_tokens.end(), output.tokens.end() - static_cast<std::ptrdiff_t>(suffix_size));
            if (suffix_size > 0 && !has_suffix) {
                output.tokens.insert(output.tokens.end(), suffix_begin, special_tokens.end());
                output.weights.insert(output.weights.end(), suffix_size, example.pieces.back().loss_weight);
            }
        }
    } catch (const std::exception & exception) {
        error = exception.what();
        return false;
    }
    if (output.tokens.size() != output.weights.size()) {
        error = "token and loss-weight lengths differ";
        return false;
    }
    if (std::none_of(output.weights.begin(), output.weights.end(), [](float weight) { return weight > 0.0f; })) {
        error = "rendered example has no supervised tokens";
        return false;
    }
    return !output.tokens.empty();
}
