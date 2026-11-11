#include "dataset-template.h"
#include "common.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string read_file(const std::string & path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "failed to open chat template: " + path);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void test_template(const std::string & data_dir, const std::string & model_dir) {
    const std::string template_path = data_dir + "/" + model_dir + "/chat_template.jinja";
    common_chat_templates_ptr templates = common_chat_templates_init(nullptr, read_file(template_path));

    chat_example chat;
    chat.messages.push_back({"user", "What is 2 + 2?"});
    chat.messages.push_back({"assistant", "4."});

    const dataset_template_result rendered = render_dataset_example(dataset_example{chat}, templates.get());
    require(rendered.diagnostics.empty(), model_dir + ": unexpected template diagnostics");
    require(rendered.examples.size() == 1, model_dir + ": expected one assistant training sample");

    std::string prompt;
    std::string target;
    for (const rendered_piece & piece : rendered.examples.front().pieces) {
        if (piece.loss_weight == 0.0f) {
            prompt += piece.text;
        } else {
            require(piece.loss_weight == 1.0f, model_dir + ": unexpected loss weight");
            target += piece.text;
        }
    }

    require(prompt.find("What is 2 + 2?") != std::string::npos,
            model_dir + ": user content is missing from the prompt");
    require(target.find("4.") != std::string::npos,
            model_dir + ": assistant content is missing from the supervised target");
    require(target.find("What is 2 + 2?") == std::string::npos,
            model_dir + ": user content leaked into the supervised target");

    std::cout << "[PASS] " << model_dir << '\n';
}

}

int main() {
    try {
        const std::string data_dir = std::string(QLORA_TEST_DATA_DIR);
        const std::vector<std::string> model_dirs = {
            "gemma-4-E2B-it",
            "gemma-4-26B-A4B-it",
            "Qwen3.5-0.8B",
            "Qwen3.6-35B-A3B",
        };
        for (const std::string & model_dir : model_dirs) {
            test_template(data_dir, model_dir);
        }
        std::cout << "passed " << model_dirs.size() << " Hugging Face chat-template checks\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
