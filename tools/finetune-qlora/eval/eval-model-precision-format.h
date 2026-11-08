#pragma once

#include <iomanip>
#include <sstream>
#include <string>

namespace qlora_model_eval {

inline std::string format_learning_rate(float value) {
    std::ostringstream output;
    output << std::scientific << std::setprecision(5) << value;
    std::string result = output.str();
    const std::string::size_type exponent_position = result.find('e');
    if (exponent_position == std::string::npos) {
        return result;
    }

    std::string mantissa = result.substr(0, exponent_position);
    while (!mantissa.empty() && mantissa.back() == '0') {
        mantissa.pop_back();
    }
    if (!mantissa.empty() && mantissa.back() == '.') {
        mantissa.pop_back();
    }

    const int exponent = std::stoi(result.substr(exponent_position + 1));
    const std::string exponent_text = exponent >= 0
        ? "+" + std::to_string(exponent)
        : std::to_string(exponent);
    return mantissa + "e" + exponent_text;
}

inline std::string format_initial_loss(double value) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(6) << value;
    return output.str();
}

inline std::string format_loss_delta(double value) {
    if (value == 0.0) {
        return "-0";
    }

    std::ostringstream output;
    output << std::scientific << std::setprecision(2) << value;
    const std::string result = output.str();
    return value > 0.0 ? "+" + result : result;
}

} // namespace qlora_model_eval
