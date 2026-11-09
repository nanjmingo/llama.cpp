#include "test-dataset-template-common.h"

#include <exception>

namespace {

void check_text_continuation_flag(common_chat_templates * templates) {
    const dataset_template_result text = render_dataset_example(text_example{"plain text"}, templates);
    qlora_template_test::require(text.examples.size() == 1 && text.examples.front().text_continuation,
                                 "text records must enable text continuation masking");

    const dataset_template_result prompt_completion = render_dataset_example(
        prompt_completion_example{"prompt", "completion"}, templates);
    qlora_template_test::require(prompt_completion.examples.size() == 1 &&
                                     !prompt_completion.examples.front().text_continuation,
                                 "prompt/completion records must not enable text continuation masking");

    chat_example chat;
    chat.messages = {{"user", "question"}, {"assistant", "answer"}};
    const dataset_template_result messages = render_dataset_example(chat, templates);
    qlora_template_test::require(messages.examples.size() == 1 &&
                                     !messages.examples.front().text_continuation,
                                 "chat records must not enable text continuation masking");
}

}

int main() {
    try {
        std::string template_source;
        const std::string template_path = std::string(QLORA_TEST_DATA_DIR) + "/templates/chatml.jinja";
        qlora_template_test::require(qlora_template_test::read_file(template_path, template_source),
                                     "failed to read test chat template: " + template_path);
        qlora_template_test::require(!template_source.empty(), "test chat template is empty: " + template_path);

        common_chat_templates_ptr templates = common_chat_templates_init(nullptr, template_source);
        check_text_continuation_flag(templates.get());
        return qlora_template_test::run_fixture_tests(templates.get(), nullptr, true);
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
