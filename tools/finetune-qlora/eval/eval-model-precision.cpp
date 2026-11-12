#include "eval-model-precision-format.h"
#include "eval-model-precision-options.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace qlora_model_eval;

namespace {

std::string format_elapsed_time(std::chrono::steady_clock::duration elapsed) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(3)
           << std::chrono::duration<double>(elapsed).count() << 's';
    return output.str();
}

double evaluate_f32_loss(Model & model, EvaluationRun & run, ggml_prec training_precision) {
    model.set_forward_precision(GGML_PREC_F32);
    try {
        const double loss = run.evaluate_inference_loss();
        model.set_forward_precision(training_precision);
        return loss;
    } catch (...) {
        model.set_forward_precision(training_precision);
        throw;
    }
}

}

int main(int argc, char ** argv) {
    try {
        Options options = parse_options(argc, argv);
        options.model.report_configuration = true;
        Model model(options.model, options.precision);

        double initial_loss = 0.0;
        {
            EvaluationRun baseline_run(model, EvaluationMode::InferenceOnly);
            initial_loss = evaluate_f32_loss(
                model, baseline_run, options.precision.forward_precision);
        }
        std::cout << "[initial_loss]=" << format_initial_loss(initial_loss) << '\n';

        for (const float learning_rate : options.learning_rates) {
            const auto run_started = std::chrono::steady_clock::now();
            EvaluationRun run(model, EvaluationMode::TrainingAndInference);
            double previous_loss = initial_loss;

            std::cout << "[lr=" << format_learning_rate(learning_rate) << "]";
            for (size_t epoch = 1; epoch <= static_cast<size_t>(options.epochs); ++epoch) {
                model.set_forward_precision(options.precision.forward_precision);
                run.train_one_step(learning_rate);
                const double current_loss = evaluate_f32_loss(
                    model, run, options.precision.forward_precision);
                const double loss_delta = current_loss - previous_loss;
                std::cout << " [" << epoch << "]=" << format_loss_delta(loss_delta);
                previous_loss = current_loss;
            }
            std::cout << " [sum]=" << format_loss_delta(previous_loss - initial_loss)
                      << " [time]=" << format_elapsed_time(std::chrono::steady_clock::now() - run_started);
            std::cout << '\n';
        }

        return 0;
    } catch (const std::exception & error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 1;
    }
}
