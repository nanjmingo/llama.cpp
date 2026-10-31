#include "test-dataset-template-common.h"

#include "dataset-tokenize.h"

#include <exception>
#include <memory>

namespace {

void check_sample_special_tokens(const llama_vocab * vocab, bool text_format) {
    dataset_example example;
    if (text_format) {
        example = text_example{"A plain text sample."};
    } else {
        example = prompt_completion_example{"A prompt.", "A completion."};
    }

    const dataset_template_result rendered = render_dataset_example(example, nullptr);
    qlora_template_test::require(rendered.examples.size() == 1,
                                 "plain-text and prompt/completion records should render one sample");
    training_sample sample;
    std::string error;
    qlora_template_test::require(
        tokenize_rendered_example(vocab, rendered.examples.front(), sample, error),
        "plain-text and prompt/completion tokenization failed: " + error);

    const llama_token bos = llama_vocab_bos(vocab);
    const std::vector<llama_token> automatic_special_tokens = common_tokenize(vocab, {}, true, false);
    const size_t prefix_count = !automatic_special_tokens.empty() && automatic_special_tokens.front() == bos ? 1 : 0;
    const auto suffix_begin = automatic_special_tokens.begin() + static_cast<std::ptrdiff_t>(prefix_count);
    const size_t suffix_size = static_cast<size_t>(automatic_special_tokens.end() - suffix_begin);
    const bool has_prefix = prefix_count == 0 || (!sample.tokens.empty() && sample.tokens.front() == bos);
    const bool has_suffix = suffix_size == 0 ||
        (sample.tokens.size() >= suffix_size &&
         std::equal(suffix_begin, automatic_special_tokens.end(),
                    sample.tokens.end() - static_cast<std::ptrdiff_t>(suffix_size)));
    const size_t bos_count = static_cast<size_t>(std::count(sample.tokens.begin(), sample.tokens.end(), bos));
    const size_t expected_bos_count = static_cast<size_t>(std::count(
        automatic_special_tokens.begin(), automatic_special_tokens.end(), bos));
    qlora_template_test::require(has_prefix && has_suffix && bos_count == expected_bos_count,
                                 text_format ? "text tokenizer special tokens must wrap the sample exactly once" :
                                               "prompt/completion tokenizer special tokens must wrap the sample exactly once");
}

}

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
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        check_sample_special_tokens(vocab, true);
        check_sample_special_tokens(vocab, false);
        return qlora_template_test::run_fixture_tests(templates.get(), vocab, false);
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
