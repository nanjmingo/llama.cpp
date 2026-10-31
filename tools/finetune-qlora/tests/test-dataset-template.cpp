#include "test-dataset-template-common.h"

#include <exception>

int main() {
    try {
        std::string template_source;
        const std::string template_path = std::string(QLORA_TEST_DATA_DIR) + "/templates/chatml.jinja";
        qlora_template_test::require(qlora_template_test::read_file(template_path, template_source),
                                     "failed to read test chat template: " + template_path);
        qlora_template_test::require(!template_source.empty(), "test chat template is empty: " + template_path);

        common_chat_templates_ptr templates = common_chat_templates_init(nullptr, template_source);
        return qlora_template_test::run_fixture_tests(templates.get(), nullptr, true);
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
