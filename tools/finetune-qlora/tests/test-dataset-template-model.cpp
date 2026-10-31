#include "test-dataset-template-common.h"

#include <exception>
#include <memory>

namespace {

struct backend_scope {
    backend_scope() { llama_backend_init(); }
    ~backend_scope() { llama_backend_free(); }
};

void print_usage(const char * program) {
    std::cerr << "Usage: " << program << " --model MODEL.gguf\n";
}

}

int main(int argc, char ** argv) {
    try {
        std::string model_path;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--model" && i + 1 < argc) {
                model_path = argv[++i];
            } else {
                print_usage(argv[0]);
                return 2;
            }
        }
        if (model_path.empty()) {
            print_usage(argv[0]);
            return 2;
        }

        backend_scope backend;
        llama_model_params params = llama_model_default_params();
        params.n_gpu_layers = 0;
        params.vocab_only = true;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(model_path.c_str(), params), llama_model_free);
        qlora_template_test::require(static_cast<bool>(model), "failed to load model vocabulary and metadata: " + model_path);

        const char * model_template = llama_model_chat_template(model.get(), nullptr);
        qlora_template_test::require(model_template && model_template[0] != '\0',
                                     "model has no default chat template");

        std::cout << "Testing the model's embedded chat template with metadata and vocabulary only; model weights are not loaded.\n";
        common_chat_templates_ptr templates = common_chat_templates_init(model.get(), "");
        return qlora_template_test::run_fixture_tests(
            templates.get(), llama_model_get_vocab(model.get()), false);
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
