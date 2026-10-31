#include "dataset-template.h"

#include "common.h"

#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

common_chat_prompt apply_chat_template(common_chat_templates * templates,
                                       std::vector<common_chat_msg> messages,
                                       bool add_generation_prompt) {
    common_chat_templates_inputs inputs;
    inputs.messages = std::move(messages);
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    inputs.enable_thinking = false;
    // Keep this call serial: the Jinja path in common_chat_templates_apply()
    // calls common_chat_extra_context() -> format_time(), which uses
    // non-thread-safe std::localtime() in common/chat.cpp.
    return common_chat_templates_apply(templates, inputs).prompt;
}

common_chat_prompt assistant_target_suffix(const common_chat_prompt & full) {
    size_t target_part = full.parts.size();
    for (size_t i = full.parts.size(); i > 0; --i) {
        const common_chat_prompt_part & part = full.parts[i - 1];
        if (part.is_input && !part.text.empty()) {
            target_part = i - 1;
            break;
        }
    }
    if (target_part == full.parts.size()) {
        throw std::runtime_error("rendered assistant message has no non-empty input part");
    }
    while (target_part > 0 && full.parts[target_part - 1].is_input) {
        --target_part;
    }
    common_chat_prompt suffix;
    for (size_t i = target_part; i < full.parts.size(); ++i) {
        suffix.append(full.parts[i].text, full.parts[i].is_input);
    }
    return suffix;
}

std::vector<common_chat_msg> convert_messages(const std::vector<dataset_message> & messages) {
    std::vector<common_chat_msg> result;
    result.reserve(messages.size());
    for (const dataset_message & source : messages) {
        common_chat_msg message;
        message.role = source.role;
        message.content = source.content;
        result.push_back(std::move(message));
    }
    return result;
}

void append_chat_prompt(rendered_example & output, const common_chat_prompt & prompt, float loss_weight) {
    for (const common_chat_prompt_part & part : prompt.parts) {
        output.pieces.push_back({part.text, !part.is_input, loss_weight});
    }
}

rendered_example render_prompt_response(const std::string & prompt, const std::string & response) {
    rendered_example output;
    output.add_special = true;
    if (prompt.empty()) {
        output.pieces.push_back({response, true, 1.0f});
    } else {
        output.pieces.push_back({prompt, true, 0.0f});
        output.pieces.push_back({response, true, 1.0f});
    }
    return output;
}

rendered_example render_assistant_turn(common_chat_templates * templates,
                                       const std::vector<common_chat_msg> & messages,
                                       size_t target_index) {
    std::vector<common_chat_msg> prompt_messages(messages.begin(), messages.begin() + target_index);
    const common_chat_prompt prompt = apply_chat_template(templates, std::move(prompt_messages), true);
    std::vector<common_chat_msg> full_messages(messages.begin(), messages.begin() + target_index + 1);
    const common_chat_prompt full_response = apply_chat_template(templates, std::move(full_messages), false);
    const common_chat_prompt response = assistant_target_suffix(full_response);
    rendered_example rendered;
    append_chat_prompt(rendered, prompt, 0.0f);
    append_chat_prompt(rendered, response, 1.0f);
    return rendered;
}

}

dataset_template_result render_dataset_example(const dataset_example & example,
                                                common_chat_templates * templates) {
    dataset_template_result result;
    if (const auto * chat = std::get_if<chat_example>(&example)) {
        if (!templates) {
            result.diagnostics.push_back("no model chat template is available; chat records cannot be rendered");
            return result;
        }
        const std::vector<common_chat_msg> messages = convert_messages(chat->messages);
        for (size_t i = 0; i < messages.size(); ++i) {
            if (messages[i].role != "assistant") {
                continue;
            }
            if (messages[i].content.empty()) {
                result.diagnostics.push_back("assistant turn " + std::to_string(i) + ": empty assistant content");
                continue;
            }
            try {
                result.examples.push_back(render_assistant_turn(templates, messages, i));
            } catch (const std::exception & error) {
                result.diagnostics.push_back("assistant turn " + std::to_string(i) + ": " + error.what());
            }
        }
        if (result.examples.empty() && result.diagnostics.empty()) {
            result.diagnostics.push_back("messages record has no assistant turns");
        }
        return result;
    }
    if (const auto * prompt_completion = std::get_if<prompt_completion_example>(&example)) {
        result.examples.push_back(render_prompt_response(prompt_completion->prompt, prompt_completion->completion));
        return result;
    }
    const auto & text = std::get<text_example>(example);
    rendered_example rendered = render_prompt_response({}, text.text);
    rendered.text_continuation = true;
    result.examples.push_back(std::move(rendered));
    return result;
}
