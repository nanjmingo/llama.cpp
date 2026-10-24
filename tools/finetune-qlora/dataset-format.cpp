#include "dataset-format.h"

#include <string>
#include <utility>
#include <vector>

namespace {

enum class record_schema {
    messages,
    prompt_completion,
    text,
    alpaca,
    sharegpt,
};

bool get_string(const nlohmann::json & object, const std::string & key, std::string & value,
                bool required, std::string & error) {
    const auto it = object.find(key);
    if (it == object.end()) {
        if (required) {
            error = "missing required field '" + key + "'";
            return false;
        }
        value.clear();
        return true;
    }
    if (!it->is_string()) {
        error = "field '" + key + "' must be a string";
        return false;
    }
    value = it->get<std::string>();
    return true;
}

bool parse_chat_messages(const nlohmann::json & values, bool sharegpt,
                         chat_example & output, std::string & error) {
    if (!values.is_array()) {
        error = sharegpt ? "conversations must be an array" : "messages must be an array";
        return false;
    }
    for (size_t i = 0; i < values.size(); ++i) {
        const nlohmann::json & value = values[i];
        if (!value.is_object()) {
            error = "message " + std::to_string(i) + " must be an object";
            return false;
        }
        dataset_message message;
        const char * role_key = sharegpt ? "from" : "role";
        const char * content_key = sharegpt ? "value" : "content";
        if (!get_string(value, role_key, message.role, true, error) ||
            !get_string(value, content_key, message.content, true, error)) {
            error = "message " + std::to_string(i) + ": " + error;
            return false;
        }
        if (message.role.empty()) {
            error = "message " + std::to_string(i) + " has an empty role";
            return false;
        }
        if (sharegpt) {
            if (message.role == "human") {
                message.role = "user";
            } else if (message.role == "gpt") {
                message.role = "assistant";
            }
        }
        output.messages.push_back(std::move(message));
    }
    if (output.messages.empty()) {
        error = sharegpt ? "conversations must not be empty" : "messages must not be empty";
        return false;
    }
    return true;
}

bool parse_prompt_completion(const nlohmann::json & input,
                             prompt_completion_example & output, std::string & error) {
    return get_string(input, "prompt", output.prompt, true, error) &&
           get_string(input, "completion", output.completion, true, error);
}

bool parse_alpaca(const nlohmann::json & input, prompt_completion_example & output,
                  std::string & error) {
    std::string instruction;
    std::string extra_input;
    if (!get_string(input, "instruction", instruction, true, error) ||
        !get_string(input, "output", output.completion, true, error) ||
        !get_string(input, "input", extra_input, false, error)) {
        return false;
    }
    output.prompt = std::move(instruction);
    if (!extra_input.empty()) {
        if (!output.prompt.empty()) {
            output.prompt += "\n\n";
        }
        output.prompt += extra_input;
    }
    return true;
}

bool has_schema(const nlohmann::json & input, record_schema schema) {
    switch (schema) {
        case record_schema::messages:
            return input.contains("messages");
        case record_schema::prompt_completion:
            return input.contains("prompt") && input.contains("completion");
        case record_schema::text:
            return input.contains("text");
        case record_schema::alpaca:
            return input.contains("instruction") && input.contains("output");
        case record_schema::sharegpt:
            return input.contains("conversations");
    }
    return false;
}

}

bool parse_dataset_record(const nlohmann::json & input,
                          dataset_example & output,
                          std::string & error) {
    if (!input.is_object()) {
        error = "record must be a JSON object";
        return false;
    }

    std::vector<record_schema> matches;
    for (record_schema candidate : {record_schema::messages, record_schema::prompt_completion,
                                    record_schema::text, record_schema::alpaca,
                                    record_schema::sharegpt}) {
        if (has_schema(input, candidate)) {
            matches.push_back(candidate);
        }
    }
    if (matches.empty()) {
        error = "unrecognized record schema";
        return false;
    }
    if (matches.size() > 1) {
        error = "record matches multiple dataset formats";
        return false;
    }

    switch (matches.front()) {
        case record_schema::messages: {
            chat_example example;
            if (!parse_chat_messages(input.at("messages"), false, example, error)) {
                return false;
            }
            output = std::move(example);
            return true;
        }
        case record_schema::sharegpt: {
            chat_example example;
            if (!parse_chat_messages(input.at("conversations"), true, example, error)) {
                return false;
            }
            output = std::move(example);
            return true;
        }
        case record_schema::prompt_completion: {
            prompt_completion_example example;
            if (!parse_prompt_completion(input, example, error)) {
                return false;
            }
            output = std::move(example);
            return true;
        }
        case record_schema::alpaca: {
            prompt_completion_example example;
            if (!parse_alpaca(input, example, error)) {
                return false;
            }
            output = std::move(example);
            return true;
        }
        case record_schema::text: {
            text_example example;
            if (!get_string(input, "text", example.text, true, error)) {
                return false;
            }
            output = std::move(example);
            return true;
        }
    }

    error = "unrecognized record schema";
    return false;
}
