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
    try {
        for (const rendered_piece & piece : example.pieces) {
            if (!std::isfinite(piece.loss_weight) || piece.loss_weight < 0.0f || piece.loss_weight > 1.0f) {
                error = "loss weights must be finite values between 0.0 and 1.0";
                return false;
            }
            const std::vector<llama_token> tokens = common_tokenize(
                vocab, piece.text, piece.add_special, piece.parse_special);
            output.tokens.insert(output.tokens.end(), tokens.begin(), tokens.end());
            output.weights.insert(output.weights.end(), tokens.size(), piece.loss_weight);
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
