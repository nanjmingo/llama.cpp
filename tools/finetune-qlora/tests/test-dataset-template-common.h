#pragma once

#include "dataset-format.h"
#include "dataset-template.h"
#include "common.h"
#include "nlohmann/json.hpp"

#ifdef QLORA_TEMPLATE_TEST_WITH_VOCAB
#include "dataset-tokenize.h"
#include "llama.h"
#else
struct llama_vocab;
#endif

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef QLORA_TEST_DATA_DIR
#define QLORA_TEST_DATA_DIR "tests/data"
#endif

namespace qlora_template_test {

using json = nlohmann::json;

struct template_test_case {
    json input;
    json expected;
};

inline void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline std::vector<json> read_jsonl(const std::string & path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "failed to open test fixture: " + path);
    std::vector<json> result;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            result.push_back(json::parse(line));
        }
    }
    return result;
}

inline bool read_file(const std::string & path, std::string & contents) {
    std::ifstream input(path);
    if (!input) {
        return false;
    }
    contents.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    return true;
}

inline std::vector<template_test_case> load_cases() {
    const std::vector<json> inputs = read_jsonl(std::string(QLORA_TEST_DATA_DIR) + "/chat-template-input.jsonl");
    const std::vector<json> expected = read_jsonl(std::string(QLORA_TEST_DATA_DIR) + "/chat-template-expected.jsonl");
    require(inputs.size() == expected.size(), "input and expected fixture counts must match");

    std::vector<template_test_case> result;
    result.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        require(inputs[i].at("id") == expected[i].at("id"), "input and expected case IDs must match");
        result.push_back({inputs[i], expected[i]});
    }
    return result;
}

inline void collect_weighted_text(const rendered_example & example,
                                  std::string & prompt,
                                  std::string & target,
                                  const std::string & case_id) {
    bool target_started = false;
    for (const rendered_piece & piece : example.pieces) {
        if (piece.loss_weight == 0.0f) {
            require(!target_started, case_id + ": prompt pieces must precede supervised pieces");
            prompt += piece.text;
        } else {
            require(piece.loss_weight == 1.0f, case_id + ": unexpected loss weight in template output");
            target_started = true;
            target += piece.text;
        }
    }
    require(!target.empty(), case_id + ": the supervised response must not be empty");
}

inline void check_literal_piece(const rendered_example & example,
                               const std::string & literal,
                               float expected_weight,
                               const std::string & case_id) {
    const auto found = std::find_if(example.pieces.begin(), example.pieces.end(), [&](const rendered_piece & piece) {
        return piece.text.find(literal) != std::string::npos;
    });
    require(found != example.pieces.end(), case_id + ": literal input piece was not preserved: " + literal);
    require(!found->parse_special, case_id + ": special-token-looking content must not be parsed as template tokens");
    require(found->loss_weight == expected_weight, case_id + ": literal content has an unexpected loss weight");
}

inline void check_literal_input(const rendered_example & example, const json & expected, const std::string & case_id) {
    if (expected.contains("literal_input")) {
        check_literal_piece(example, expected.at("literal_input").get<std::string>(), 0.0f, case_id);
    }
    if (expected.contains("literal_target_input")) {
        check_literal_piece(example, expected.at("literal_target_input").get<std::string>(), 1.0f, case_id);
    }
}

inline void check_semantics(const rendered_example & example,
                            const json & expected,
                            const std::string & prompt,
                            const std::string & target,
                            const std::string & case_id) {
    const std::string target_content = expected.at("target_content").get<std::string>();
    require(target.find(target_content) != std::string::npos, case_id + ": target content is missing from supervised text");
    for (const json & text : expected.at("must_be_in_prompt")) {
        const std::string value = text.get<std::string>();
        require(prompt.find(value) != std::string::npos, case_id + ": prior conversation content is missing from prompt: " + value);
    }
    for (const json & text : expected.at("must_not_be_in_target")) {
        const std::string value = text.get<std::string>();
        require(target.find(value) == std::string::npos, case_id + ": context text leaked into supervised target: " + value);
    }
    check_literal_input(example, expected, case_id);
}

inline void check_case(const template_test_case & test_case,
                       common_chat_templates * templates,
                       const llama_vocab * vocab,
                       bool exact_golden) {
    const std::string case_id = test_case.input.at("id").get<std::string>();
    dataset_example example;
    std::string error;
    require(parse_dataset_record(test_case.input, example, error),
            case_id + ": fixture record failed to parse: " + error);

    const dataset_template_result rendered = render_dataset_example(example, templates);
    const json & expected = test_case.expected;
    require(rendered.examples.size() == expected.at("samples").size(),
            case_id + ": unexpected number of assistant training samples");

    if (expected.contains("diagnostic_contains")) {
        const std::string expected_diagnostic = expected.at("diagnostic_contains").get<std::string>();
        const bool found = std::any_of(rendered.diagnostics.begin(), rendered.diagnostics.end(), [&](const std::string & diagnostic) {
            return diagnostic.find(expected_diagnostic) != std::string::npos;
        });
        require(found, case_id + ": expected diagnostic was not produced");
    } else {
        require(rendered.diagnostics.empty(), case_id + ": valid fixture produced unexpected diagnostics");
    }

    for (size_t i = 0; i < rendered.examples.size(); ++i) {
        const rendered_example & actual = rendered.examples[i];
        const json & expected_sample = expected.at("samples")[i];
        std::string prompt;
        std::string target;
        collect_weighted_text(actual, prompt, target, case_id);
        check_semantics(actual, expected_sample, prompt, target, case_id);

        if (exact_golden) {
            require(prompt == expected_sample.at("prompt").get<std::string>(),
                    case_id + ": rendered prompt differs from the template golden output");
            require(target == expected_sample.at("target").get<std::string>(),
                    case_id + ": supervised response differs from the template golden output");
        }

#ifdef QLORA_TEMPLATE_TEST_WITH_VOCAB
        if (vocab) {
            training_sample sample;
            std::string tokenize_error;
            require(tokenize_rendered_example(vocab, actual, sample, tokenize_error),
                    case_id + ": tokenization failed: " + tokenize_error);
            require(sample.tokens.size() == sample.weights.size(), case_id + ": token and weight lengths differ");
            require(std::any_of(sample.weights.begin(), sample.weights.end(), [](float weight) { return weight > 0.0f; }),
                    case_id + ": tokenized example has no supervised tokens");
        }
#else
        (void) vocab;
#endif
    }
}

inline int run_fixture_tests(common_chat_templates * templates, const llama_vocab * vocab, bool exact_golden) {
    const std::vector<template_test_case> cases = load_cases();
    for (const template_test_case & test_case : cases) {
        check_case(test_case, templates, vocab, exact_golden);
        std::cout << "[PASS] " << test_case.input.at("id").get<std::string>() << '\n';
    }
    std::cout << "passed " << cases.size() << " chat-template fixture cases\n";
    return 0;
}

}
