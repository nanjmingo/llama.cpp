#include "qlora_dataset.h"

#include "chat.h"
#include "common.h"
#include "ggml.h"
#include "log.h"

#include <algorithm>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

namespace {

struct tokenization_request {
    std::string prompt;
    std::string response;
};

std::string preview_text(const std::string & text, size_t max_len = 240) {
    std::string preview;
    preview.reserve(std::min(text.size(), max_len));

    for (char c : text) {
        if (preview.size() >= max_len) {
            preview += "...";
            break;
        }
        switch (c) {
            case '\n':
                preview += "\\n";
                break;
            case '\r':
                preview += "\\r";
                break;
            case '\t':
                preview += "\\t";
                break;
            default:
                preview += c;
                break;
        }
    }
    return preview;
}

common_chat_msg parse_chat_message(const nlohmann::json & data) {
    if (!data.is_object()) {
        throw std::runtime_error("each message must be an object");
    }

    common_chat_msg message;
    message.role = data.value("role", std::string("user"));

    common_chat_msg_content_part part;
    part.type = "text";
    part.text = data.value("content", std::string());
    message.content_parts.push_back(std::move(part));
    return message;
}

std::string apply_chatml_message(const common_chat_msg & message) {
    std::string rendered = "<|im_start|>" + message.role + "\n";
    for (const common_chat_msg_content_part & part : message.content_parts) {
        rendered += part.text;
    }
    rendered += "<|im_end|>\n";
    return rendered;
}

std::string apply_qlora_chat_template(common_chat_templates *              templates,
                                      const std::vector<common_chat_msg> & messages,
                                      bool                                 add_generation_prompt) {
    common_chat_templates_inputs inputs;
    inputs.messages              = messages;
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.tool_choice           = COMMON_CHAT_TOOL_CHOICE_NONE;
    inputs.reasoning_format      = COMMON_REASONING_FORMAT_DEEPSEEK;
    inputs.enable_thinking       = false;
    return common_chat_templates_apply(templates, inputs).prompt;
}

std::string trim_text(const std::string & text) {
    constexpr char whitespace[] = " \t\n\r\f\v";
    const size_t   begin        = text.find_first_not_of(whitespace);
    if (begin == std::string::npos) {
        return {};
    }
    const size_t end = text.find_last_not_of(whitespace);
    return text.substr(begin, end - begin + 1);
}

bool format_chat_record(const nlohmann::json &  data,
                        common_chat_templates * templates,
                        int                     line_number,
                        tokenization_request &  request) {
    std::vector<common_chat_msg> messages;
    try {
        for (const nlohmann::json & message : data.at("messages")) {
            messages.push_back(parse_chat_message(message));
        }
    } catch (const std::exception & e) {
        LOG_WRN("%s: skipping invalid message on line %d: %s\n", __func__, line_number, e.what());
        return false;
    }

    if (messages.empty()) {
        return false;
    }

    size_t last_assistant = messages.size();
    for (size_t i = messages.size(); i-- > 0;) {
        if (messages[i].role == "assistant") {
            last_assistant = i;
            break;
        }
    }
    if (last_assistant == messages.size()) {
        LOG_DBG("%s: skipping line %d: no assistant turn\n", __func__, line_number);
        return false;
    }

    // Without a model template, use the built-in ChatML representation as the full sample.
    if (!templates) {
        for (size_t i = 0; i <= last_assistant; ++i) {
            request.response += apply_chatml_message(messages[i]);
        }
        return true;
    }

    const std::vector<common_chat_msg> prompt_messages(messages.begin(), messages.begin() + last_assistant);

    try {
        request.prompt = apply_qlora_chat_template(templates, prompt_messages, true);

        // This path targets ordinary Gemma 4 text conversations.
        // Gemma 4's assistant turn ends with <turn|>.
        const std::string assistant_content =
            messages[last_assistant].content_parts.empty() ? "" : messages[last_assistant].content_parts[0].text;
        request.response = trim_text(assistant_content) + "<turn|>\n";
        return true;
    } catch (const std::exception & e) {
        LOG_WRN("%s: chat template failed on line %d: %s\n", __func__, line_number, e.what());
        return false;
    }
}

bool format_record(const nlohmann::json &  data,
                   common_chat_templates * templates,
                   int                     line_number,
                   tokenization_request &  request) {
    try {
        if (data.contains("messages")) {
            return format_chat_record(data, templates, line_number, request);
        }
        if (data.contains("prompt") && data.contains("response")) {
            request.prompt   = data.at("prompt").get<std::string>();
            request.response = data.at("response").get<std::string>();
            return true;
        }
        if (data.contains("text")) {
            request.response = data.at("text").get<std::string>();
            return true;
        }

        LOG_WRN("%s: unknown format on line %d, skipping\n", __func__, line_number);
    } catch (const std::exception & e) {
        LOG_WRN("%s: skipping invalid data on line %d: %s\n", __func__, line_number, e.what());
    }
    return false;
}

}  // namespace

std::vector<training_sample> load_jsonl(const std::string &     path,
                                        const llama_vocab *     vocab,
                                        common_chat_templates * tmpls) {
    std::ifstream input(path);
    if (!input) {
        LOG_ERR("%s: failed to open JSONL file: %s\n", __func__, path.c_str());
        return {};
    }

    std::vector<training_sample> samples;
    std::string                  line;
    int64_t                      line_number    = 0;
    size_t                       n_requests     = 0;
    bool                         logged_preview = false;

    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }

        nlohmann::json data;
        try {
            data = nlohmann::json::parse(line);
        } catch (const std::exception &) {
            LOG_WRN("%s: skipping invalid JSON on line %d\n", __func__, (int) line_number);
            continue;
        }

        tokenization_request request;
        if (!format_record(data, tmpls, (int) line_number, request)) {
            continue;
        }
        ++n_requests;

        try {
            const std::vector<llama_token> prompt_tokens =
                common_tokenize(vocab, request.prompt, /*add_special=*/true, /*parse_special=*/true);
            const std::vector<llama_token> response_tokens =
                common_tokenize(vocab, request.response, /*add_special=*/false, /*parse_special=*/true);
            if (prompt_tokens.empty() && response_tokens.empty()) {
                continue;
            }

            training_sample sample;
            sample.tokens.reserve(prompt_tokens.size() + response_tokens.size());
            sample.tokens.insert(sample.tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
            sample.tokens.insert(sample.tokens.end(), response_tokens.begin(), response_tokens.end());
            sample.is_label.resize(sample.tokens.size(), false);
            std::fill(sample.is_label.begin() + prompt_tokens.size(), sample.is_label.end(), true);

            if (!logged_preview) {
                logged_preview = true;
                LOG_INF("%s: first prompt preview: \"%s\"\n", __func__, preview_text(request.prompt).c_str());
                LOG_INF("%s: first response preview: \"%s\"\n", __func__, preview_text(request.response).c_str());
                LOG_INF("%s: first sample tokens: prompt=%zu response=%zu\n", __func__, prompt_tokens.size(),
                        response_tokens.size());
            }
            samples.push_back(std::move(sample));
        } catch (const std::exception & e) {
            LOG_WRN("%s: tokenization failed on line %d: %s\n", __func__, (int) line_number, e.what());
        }
    }

    LOG_INF("%s: tokenized %zu input lines with 1 worker\n", __func__, n_requests);
    LOG_INF("%s: loaded %zu samples from %s\n", __func__, samples.size(), path.c_str());
    return samples;
}

struct training_window {
    std::vector<int32_t> data;
    std::vector<int32_t> labels;
};

ggml_opt_dataset_t build_dataset(const std::vector<training_sample> & samples, int32_t n_ctx, llama_token bos_token) {
    const int32_t overlap = n_ctx / 2;
    const int32_t stride  = n_ctx - overlap;

    std::vector<training_window> windows;
    int64_t                      token_count = 0;
    int64_t                      n_labels    = 0;
    int64_t                      n_padded    = 0;

    for (const auto & s : samples) {
        const int64_t s_size = s.tokens.size();
        if (s_size <= 1) {
            continue;
        }
        // test
        if (s_size <= n_ctx) {
            continue;
        }

        token_count += s_size - 1;

        for (int64_t s_offset = 0; s_offset < s_size; s_offset += stride) {
            // test
            if (s_offset + n_ctx >= s_size) {
                break;
            }

            training_window window;
            window.data.reserve(n_ctx);
            window.labels.reserve(n_ctx);

            for (int32_t i = 0; i < n_ctx; ++i) {
                const int64_t s_index = s_offset + i;

                if (s_index + 1 < s_size) {
                    window.data.push_back(s.tokens[s_index]);

                    if (s_offset > 0 && i < overlap) {
                        window.labels.push_back(-1);
                    } else if (i < overlap / 2) {  // test
                        window.labels.push_back(-1);
                    } else {
                        const int32_t label = s.is_label[s_index + 1] ? s.tokens[s_index + 1] : -1;

                        window.labels.push_back(label);
                        n_labels += label >= 0;
                    }
                } else {
                    window.data.push_back(bos_token);
                    window.labels.push_back(-1);
                    ++n_padded;
                }
            }

            windows.push_back(std::move(window));
        }
    }

    if (windows.empty()) {
        LOG_ERR("%s: dataset has no tokens\n", __func__);
        return nullptr;
    }

    const int64_t window_count = windows.size();

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(GGML_TYPE_I32, GGML_TYPE_I32, n_ctx, n_ctx, window_count, 1);

    if (dataset == nullptr) {
        return nullptr;
    }

    auto * data   = static_cast<int32_t *>(ggml_opt_dataset_data(dataset)->data);
    auto * labels = static_cast<int32_t *>(ggml_opt_dataset_labels(dataset)->data);

    for (int64_t i = 0; i < window_count; ++i) {
        std::memcpy(data + i * n_ctx, windows[i].data.data(), n_ctx * sizeof(int32_t));
        std::memcpy(labels + i * n_ctx, windows[i].labels.data(), n_ctx * sizeof(int32_t));
    }

    LOG_INF(
        "%s: packed %ld tokens into %ld windows "
        "(ctx=%d stride=%ld, supervised labels=%ld, padded=%ld)\n",
        __func__, (long) token_count, (long) window_count, n_ctx, (long) stride, (long) n_labels, (long) n_padded);

    return dataset;
}
