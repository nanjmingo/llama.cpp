#include "ggml-opt.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace {

template<typename T>
void append_integer(std::vector<uint8_t> & data, T value) {
    for (size_t i = 0; i < sizeof(T); ++i) {
        data.push_back(static_cast<uint8_t>((static_cast<uint64_t>(value) >> (i * 8)) & 0xff));
    }
}

std::vector<uint8_t> make_optimizer_state(ggml_type type, float value) {
    std::vector<uint8_t> data = {'G', 'G', 'O', 'P', 'T', 'S', 'T', '1'};
    append_integer<uint32_t>(data, 2);
    append_integer<uint32_t>(data, GGML_OPT_OPTIMIZER_TYPE_ADAMW);
    append_integer<uint32_t>(data, type);
    append_integer<uint64_t>(data, 1);
    append_integer<uint64_t>(data, 0);
    append_integer<uint32_t>(data, 1);
    append_integer<uint32_t>(data, 1);
    data.push_back('m');
    append_integer<uint32_t>(data, type);
    append_integer<uint32_t>(data, 1);
    append_integer<uint64_t>(data, 2);
    append_integer<uint64_t>(data, 1);
    append_integer<uint64_t>(data, 1);
    append_integer<uint64_t>(data, 1);
    append_integer<uint64_t>(data, type == GGML_TYPE_F32 ? 8 : 4);

    if (type == GGML_TYPE_F32) {
        append_integer<uint32_t>(data, 0);
        append_integer<uint32_t>(data, 0);
        std::memcpy(data.data() + data.size() - 8, &value, sizeof(value));
        std::memcpy(data.data() + data.size() - 4, &value, sizeof(value));
    } else if (type == GGML_TYPE_F16) {
        const ggml_fp16_t encoded = ggml_fp32_to_fp16(value);
        const auto * bytes = reinterpret_cast<const uint8_t *>(&encoded);
        data.insert(data.end(), bytes, bytes + sizeof(encoded));
        data.insert(data.end(), bytes, bytes + sizeof(encoded));
    } else if (type == GGML_TYPE_BF16) {
        const ggml_bf16_t encoded = ggml_fp32_to_bf16(value);
        const auto * bytes = reinterpret_cast<const uint8_t *>(&encoded);
        data.insert(data.end(), bytes, bytes + sizeof(encoded));
        data.insert(data.end(), bytes, bytes + sizeof(encoded));
    }
    return data;
}

void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

void test_finite_and_nonfinite_payloads() {
    for (const ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
        const std::vector<uint8_t> finite = make_optimizer_state(type, 1.0f);
        require(ggml_opt_state_data_is_finite(finite.data(), finite.size()),
                "finite optimizer-state payload was rejected");

        const std::vector<uint8_t> nan = make_optimizer_state(type, std::numeric_limits<float>::quiet_NaN());
        require(!ggml_opt_state_data_is_finite(nan.data(), nan.size()),
                "NaN optimizer-state payload was accepted");

        const std::vector<uint8_t> inf = make_optimizer_state(type, std::numeric_limits<float>::infinity());
        require(!ggml_opt_state_data_is_finite(inf.data(), inf.size()),
                "Inf optimizer-state payload was accepted");

        require(!ggml_opt_state_data_is_finite(finite.data(), finite.size() - 1),
                "truncated optimizer-state payload was accepted");
        std::vector<uint8_t> trailing = finite;
        trailing.push_back(0);
        require(!ggml_opt_state_data_is_finite(trailing.data(), trailing.size()),
                "optimizer-state payload with trailing data was accepted");
    }
}

void test_version_one_payload() {
    std::vector<uint8_t> state = make_optimizer_state(GGML_TYPE_F32, 1.0f);
    state.erase(state.begin() + 16, state.begin() + 20);
    state[8] = 1;
    state[9] = 0;
    state[10] = 0;
    state[11] = 0;
    require(ggml_opt_state_data_is_finite(state.data(), state.size()),
            "version 1 optimizer-state payload was rejected");
}

} // namespace

int main() {
    test_finite_and_nonfinite_payloads();
    test_version_one_payload();
    require(!ggml_opt_state_data_is_finite(nullptr, 0), "null optimizer-state payload was accepted");
    std::cout << "optimizer state data validation tests passed\n";
    return 0;
}
