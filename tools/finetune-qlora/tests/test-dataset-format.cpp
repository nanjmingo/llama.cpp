#include "dataset-format.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <variant>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

}

int main() {
    dataset_example example;
    std::string error;

    require(parse_dataset_record(nlohmann::json::parse(
        R"({"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]})"),
        example, error), "messages record should parse");
    require(std::holds_alternative<chat_example>(example), "messages should normalize to chat example");
    require(std::get<chat_example>(example).messages.size() == 2, "messages count should be preserved");

    require(parse_dataset_record(nlohmann::json::parse(
        R"({"prompt":"Question","completion":"Answer"})"),
        example, error), "prompt/completion record should parse");
    require(std::holds_alternative<prompt_completion_example>(example), "prompt/completion should normalize correctly");
    require(std::get<prompt_completion_example>(example).completion == "Answer", "completion text should be preserved");

    require(parse_dataset_record(nlohmann::json::parse(R"({"text":"A text sample"})"),
        example, error), "text record should parse");
    require(std::get<text_example>(example).text == "A text sample", "text should be preserved");

    require(!parse_dataset_record(nlohmann::json::parse(R"({"messages":"A plain-text sample"})"),
        example, error), "string-valued messages should be rejected");
    require(error == "messages must be an array", "invalid messages field should report its expected type");

    require(parse_dataset_record(nlohmann::json::parse(
        R"({"instruction":"Do this","input":"with this","output":"Done"})"),
        example, error), "Alpaca record should parse");
    const auto & alpaca = std::get<prompt_completion_example>(example);
    require(alpaca.prompt == "Do this\n\nwith this", "Alpaca instruction and input should be joined deterministically");
    require(alpaca.completion == "Done", "Alpaca output should become the completion");

    require(parse_dataset_record(nlohmann::json::parse(
        R"({"conversations":[{"from":"human","value":"Hi"},{"from":"gpt","value":"Hello"}]})"),
        example, error), "ShareGPT record should parse");
    const auto & sharegpt = std::get<chat_example>(example);
    require(sharegpt.messages[0].role == "user", "ShareGPT human role should map to user");
    require(sharegpt.messages[1].role == "assistant", "ShareGPT gpt role should map to assistant");

    require(!parse_dataset_record(nlohmann::json::parse(R"({"prompt":"Q","response":"A"})"),
        example, error), "nonstandard prompt/response schema should be rejected");
    require(!parse_dataset_record(nlohmann::json::parse(
        R"({"messages":[],"text":"ambiguous"})"), example, error),
        "ambiguous record schemas should be rejected");
    require(error == "record matches multiple dataset formats", "ambiguous schema diagnostic should be clear");
    require(!parse_dataset_record(nlohmann::json::parse(R"({"messages":[{"role":"user"}]})"),
        example, error), "missing message content should be rejected");
    return 0;
}
