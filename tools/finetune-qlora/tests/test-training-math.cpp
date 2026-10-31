#include "test-training-math-common.h"

#include <exception>
#include <iostream>

int main() {
    try {
        qlora_training_math::run_training_math_loss_tests();
        qlora_training_math::run_training_math_lora_rank_tests();
        qlora_training_math::run_training_math_accumulation_tests();
        qlora_training_math::run_training_math_ggml_opt_tests();
        qlora_training_math::run_training_math_lora_gradient_tests();
        qlora_training_math::run_training_math_adamw_test();
        qlora_training_math::run_training_math_quantized_test();
        std::cout << "training math tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
