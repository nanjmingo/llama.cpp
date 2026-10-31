#include "eval-model-precision-format.h"
#include "eval-model-precision-options.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace qlora_model_experiment;
using namespace qlora_model_eval;

namespace {

std::string format_elapsed_time(std::chrono::steady_clock::duration elapsed) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(3)
           << std::chrono::duration<double>(elapsed).count() << 's';
    return output.str();
}

double evaluate_f32_loss(Model & model, Experiment & experiment, ggml_prec training_precision) {
    model.set_activation_precision(GGML_PREC_F32);
    try {
        const double loss = experiment.evaluate_inference_loss();
        model.set_activation_precision(training_precision);
        return loss;
    } catch (...) {
        model.set_activation_precision(training_precision);
        throw;
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        Options options = parse_options(argc, argv);
        options.model.report_configuration = true;
        Model model(options.model, options.precision);

        double initial_loss = 0.0;
        {
            Experiment baseline_experiment(model, ContextMode::InferenceOnly);
            initial_loss = evaluate_f32_loss(
                model, baseline_experiment, options.precision.activation_precision);
        }
        std::cout << "[initial_loss]=" << format_initial_loss(initial_loss) << '\n';

        for (const float learning_rate : options.learning_rates) {
            const auto run_started = std::chrono::steady_clock::now();
            Experiment experiment(model, ContextMode::TrainingAndInference);
            double previous_loss = initial_loss;

            std::cout << "[lr=" << format_learning_rate(learning_rate) << "]";
            for (size_t epoch = 1; epoch <= static_cast<size_t>(options.epochs); ++epoch) {
                model.set_activation_precision(options.precision.activation_precision);
                experiment.train_one_step(learning_rate);
                const double current_loss = evaluate_f32_loss(
                    model, experiment, options.precision.activation_precision);
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
