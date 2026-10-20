// QLoRA fine-tuning for quantized GGUF models.
//
// The base model weights stay frozen (quantized tensors are skipped by
// llama_set_param because they are not GGML_TYPE_F32).  Only the freshly
// allocated F32 LoRA A/B tensors are trained.  After training the adapter
// is saved as a GGUF file that is directly compatible with the existing
// llama_adapter_lora_init() loader and llama-export-lora merge tool.
//
// Usage example:
/*   llama-finetune-qlora \
         --model model-q4_k_m.gguf \
         --train-file train.jsonl \
         --lora-rank 16 --lora-alpha 16 \
         --lr-scheduler cosine --warmup-steps 10 --warmup-init-ratio 0.1 \
         --lora-out adapter.gguf \
         --epochs 3 -c 4096 -b 4096 -ub 512
*/
// Default targets: attn_q, attn_output, ffn_gate, ffn_up, ffn_down
// Override with --lora-targets "comma,separated,substrings"
//
// NOTE: attn_k and attn_v are excluded from defaults.  The KV write path uses
// ggml_set_rows (scatter op) — backward cannot propagate gradients through it.
// LoRA K/V would receive zero gradient.
//
// NOTE: ssm_in and ssm_out (Mamba/NemotronH) are excluded from defaults.
// SSM_SCAN/SSM_CONV have no backward implementation — LoRA on these layers
// would receive zero gradient.  Adding them wastes memory with no benefit.
//
// NOTE: MoE expert tensors (*_exps) are excluded regardless of --lora-targets.
// The quantized expert weights are frozen (stop-gradient), but LoRA on dense
// FFN layers (ffn_gate, ffn_up, ffn_down) works via MUL_MAT_ID backward.
//
// Target substrings use llama.cpp internal GGUF names (NOT HuggingFace names):
//   attn_q      = q_proj       attn_k     = k_proj
//   attn_v      = v_proj       attn_output= o_proj
//   ffn_gate    = gate_proj    ffn_up     = up_proj    ffn_down = down_proj
//   ssm_in      = in_proj (Mamba/NemotronH)  — zero gradient, not in defaults
//   ssm_out     = out_proj (Mamba/NemotronH)  — zero gradient, not in defaults

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "log.h"
#include "qlora_adapter.h"
#include "qlora_dataset.h"

// Internal adapter struct — included directly to avoid the temp-GGUF roundtrip
// for wiring trainable LoRA tensors into the compute graph.
#include "../../src/llama-adapter.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<std::string> split_csv(const std::string & input) {
    std::vector<std::string> values;
    std::istringstream       stream(input);
    std::string              value;

    while (std::getline(stream, value, ',')) {
        if (!value.empty()) {
            values.push_back(value);
        }
    }
    return values;
}

struct qlora_lr_schedule {
    const lr_opt * lr;
    std::string    type;
    int64_t        warmup_steps;
    float          warmup_init_ratio;
    int64_t        total_steps;
    int64_t        step;
    float          current_lr = 0.0f;

    float get_lr() const {
        const float base_lr = lr->lr0;

        // Linearly increase the learning rate during warmup.
        if (step < warmup_steps) {
            const float warmup_start_lr = base_lr * warmup_init_ratio;
            const float warmup_progress = (float) step / (float) warmup_steps;
            return warmup_start_lr + (base_lr - warmup_start_lr) * warmup_progress;
        }

        // Use the configured schedule unless cosine decay is applicable.
        if (type != "cosine" || total_steps <= warmup_steps) {
            return lr->get_lr();
        }

        const float  min_lr         = std::max(0.0f, lr->lr_min);
        const double decay_progress = (double) (step - warmup_steps) / (double) (total_steps - warmup_steps);
        const double cosine_weight  = 0.5 * (1.0 + std::cos(std::acos(-1.0) * std::min(1.0, decay_progress)));
        return min_lr + (base_lr - min_lr) * (float) cosine_weight;
    }
};

static ggml_opt_optimizer_params qlora_opt_lr_pars(void * userdata) {
    qlora_lr_schedule &       schedule = *(qlora_lr_schedule *) userdata;
    ggml_opt_optimizer_params result   = ggml_opt_get_default_optimizer_params(nullptr);
    schedule.current_lr                = schedule.get_lr();
    result.adamw.alpha = result.sgd.alpha = schedule.current_lr;
    result.sgd.wd = result.adamw.wd = schedule.lr->wd;
    return result;
}

// ---------------------------------------------------------------------------
// Periodic checkpoint callback
// ---------------------------------------------------------------------------

struct save_ctx {
    const lora_tensors * lt;
    const std::string *  lora_out;
    const std::string *  arch;
    const std::string *  base_model_path;
    float                lora_alpha;
    int32_t              save_every;     // 0 = disabled
    int32_t              ubatch_per_ctx;
    int64_t              window_offset;  // completed windows before this epoch call
    int64_t              last_saved;     // last window index at which we saved
    int64_t              epoch;
    int64_t              dataset_windows;
    int64_t              context_length;
    bool                 shuffle;
    bool                 verbose_loss;
    qlora_lr_schedule *  schedule;
    bool                 loss_ema_initialized  = false;
    double               loss_ema16            = 0.0;
    double               loss_ema64            = 0.0;
    double               loss_cumulative_sum   = 0.0;
    int64_t              loss_cumulative_count = 0;
    double               loss_epoch_sum        = 0.0;
    int64_t              loss_epoch_count      = 0;
};

// TLS pointer set before each epoch so the static callback can access it.
static thread_local save_ctx * g_save_ctx = nullptr;

static void save_every_callback(bool               train,
                                ggml_opt_context_t opt_ctx,
                                ggml_opt_dataset_t dataset,
                                ggml_opt_result_t  result,
                                int64_t            ibatch,
                                int64_t            ibatch_max,
                                int64_t            t_start_us) {
    ggml_opt_epoch_callback_progress_bar(train, opt_ctx, dataset, result, ibatch, ibatch_max, t_start_us);

    // Log loss at every window boundary so we can see if/when it diverges.
    if (train && g_save_ctx) {
        const int64_t window           = g_save_ctx->window_offset + ibatch / g_save_ctx->ubatch_per_ctx;
        const int64_t ubatch_in_window = ibatch % g_save_ctx->ubatch_per_ctx;
        if (ibatch > 0 && ubatch_in_window == 0) {
            if (g_save_ctx->verbose_loss) {
                double cumulative_loss_mean = 0.0;
                ggml_opt_result_loss(result, &cumulative_loss_mean, nullptr);

                const int64_t cumulative_loss_count = ibatch;
                const double  cumulative_loss_sum   = cumulative_loss_mean * cumulative_loss_count;
                const double  window_loss_sum       = cumulative_loss_sum - g_save_ctx->loss_cumulative_sum;
                const int64_t window_loss_count     = cumulative_loss_count - g_save_ctx->loss_cumulative_count;
                const double  window_loss           = window_loss_sum / std::max<int64_t>(1, window_loss_count);

                g_save_ctx->loss_ema16            = g_save_ctx->loss_ema_initialized ?
                                                        0.8825 * g_save_ctx->loss_ema16 + 0.1175 * window_loss :
                                                        window_loss;
                g_save_ctx->loss_ema64            = g_save_ctx->loss_ema_initialized ?
                                                        0.9692 * g_save_ctx->loss_ema64 + 0.0308 * window_loss :
                                                        window_loss;
                g_save_ctx->loss_ema_initialized  = true;
                g_save_ctx->loss_cumulative_sum   = cumulative_loss_sum;
                g_save_ctx->loss_cumulative_count = cumulative_loss_count;
                g_save_ctx->loss_epoch_sum += window_loss_sum;
                g_save_ctx->loss_epoch_count += window_loss_count;

                const double epoch_mean =
                    g_save_ctx->loss_epoch_sum / std::max<int64_t>(1, g_save_ctx->loss_epoch_count);
                fprintf(stderr,
                        "\nepoch=%ld window=%ld window_loss=%.6f ema16=%.6f ema64=%.6f epoch_mean=%.6f lr=%.6g\n",
                        (long) g_save_ctx->epoch, (long) window, window_loss, g_save_ctx->loss_ema16,
                        g_save_ctx->loss_ema64, epoch_mean, (double) g_save_ctx->schedule->current_lr);
            }
            ++g_save_ctx->schedule->step;
        }
    }

    if (!train || !g_save_ctx || g_save_ctx->save_every <= 0) {
        return;
    }
    const int64_t window = g_save_ctx->window_offset + ibatch / g_save_ctx->ubatch_per_ctx;
    if (window > 0 && window != g_save_ctx->last_saved && window % g_save_ctx->save_every == 0) {
        g_save_ctx->last_saved = window;
        const std::string ckpt = *g_save_ctx->lora_out + ".epoch" + std::to_string(g_save_ctx->epoch) + ".ckpt" +
                                 std::to_string(window) + ".gguf";
        checkpoint_state  state;
        state.mode            = "sft";
        state.epoch           = g_save_ctx->epoch;
        state.window          = window;
        state.dataset_windows = g_save_ctx->dataset_windows;
        state.context_length  = g_save_ctx->context_length;
        state.shuffle         = g_save_ctx->shuffle;
        state.schedule_step   = g_save_ctx->schedule->step;
        save_adapter(*g_save_ctx->lt, ckpt, *g_save_ctx->arch, g_save_ctx->lora_alpha, *g_save_ctx->base_model_path,
                     &state);
        fprintf(stderr, "\n");
        LOG_INF("save_every_callback: checkpoint saved -> %s (window %ld)\n", ckpt.c_str(), (long) window);
    }
}

static void log_epoch_loss(int epoch, int total_epochs, ggml_opt_result_t train_result) {
    double train_loss  = 0.0;
    double uncertainty = 0.0;
    ggml_opt_result_loss(train_result, &train_loss, &uncertainty);
    LOG_INF("epoch %d/%d: train_loss=%.4f ± %.4f\n", epoch, total_epochs, train_loss, uncertainty);
}

// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.escape = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE_QLORA)) {
        return 1;
    }

    if (params.train_file.empty()) {
        LOG_ERR("%s: --train-file is required\n", __func__);
        return 1;
    }

    checkpoint_state resume_state;
    const bool       resume_requested = !params.lora_resume.empty();
    if (resume_requested) {
        if (!params.lora_adapters.empty()) {
            LOG_ERR("%s: --resume cannot be combined with --lora\n", __func__);
            return 1;
        }
        const std::string mode = "sft";
        if (!load_checkpoint_state(params.lora_resume, mode, resume_state)) {
            return 1;
        }
        common_adapter_lora_info adapter_info;
        adapter_info.path  = params.lora_resume;
        adapter_info.scale = 1.0f;
        params.lora_adapters.push_back(adapter_info);
    }

    // Force settings required for training
    params.load_mode       = LLAMA_LOAD_MODE_NONE;
    params.cache_type_k    = GGML_TYPE_F32;
    params.cache_type_v    = GGML_TYPE_F32;
    // Warmup runs inference with PARAM-flagged tensors which causes a segfault;
    // training never benefits from warmup, so disable it unconditionally.
    params.warmup          = false;
    // Flash attention has no backward implementation; force standard attention for training.
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    float lora_alpha = (params.lora_alpha > 0.0f) ? params.lora_alpha : (float) params.lora_rank;
    if (resume_requested && resume_state.has_alpha) {
        if (params.lora_alpha > 0.0f && std::abs(params.lora_alpha - resume_state.alpha) > 1e-6f) {
            LOG_WRN("%s: ignoring --lora-alpha %.3f; checkpoint alpha is %.3f\n", __func__, (double) params.lora_alpha,
                    (double) resume_state.alpha);
        }
        lora_alpha = resume_state.alpha;
    }
    const auto targets = split_csv(params.lora_targets);

    // --- Step 1: Discover tensor shapes from model GGUF (no model load yet) ---
    std::string arch;
    {
        struct ggml_context *   ctx_meta = nullptr;
        struct gguf_init_params gp       = { true, &ctx_meta };
        struct gguf_context *   ctx_gguf = gguf_init_from_file(params.model.path.c_str(), gp);
        if (!ctx_gguf) {
            LOG_ERR("failed to open model GGUF\n");
            return 1;
        }
        int kid = gguf_find_key(ctx_gguf, "general.architecture");
        if (kid >= 0) {
            arch = gguf_get_val_str(ctx_gguf, kid);
        }
        gguf_free(ctx_gguf);
        ggml_free(ctx_meta);
    }

    // --- Step 2: Allocate LoRA tensors and save initial adapter GGUF ---
    // If the user already supplied a --lora adapter we reuse it (resume training).
    // Otherwise we allocate fresh tensors (B=0, A=random), write them to a temp
    // .init.gguf so common_init_from_params can load them before context creation
    // (this makes sched_reserve size the graph to include LoRA nodes).
    const bool resume_from_lora = !params.lora_adapters.empty();

    std::mt19937 rng(42);
    lora_tensors lt;  // will be populated after context load (Step 4)
    std::string  init_adapter_path;

    if (!resume_from_lora) {
        lt = alloc_lora_tensors(params.model.path, targets, params.lora_rank, rng, params.lora_freeze_layers);
        if (lt.ab.empty()) {
            return 1;
        }

        init_adapter_path = params.lora_out + ".init.gguf";
        save_adapter(lt, init_adapter_path, arch, lora_alpha, params.model.path);

        // Register adapter so common_init_from_params loads it before context creation
        common_adapter_lora_info adapter_info;
        adapter_info.path  = init_adapter_path;
        adapter_info.scale = 1.0f;
        params.lora_adapters.push_back(adapter_info);
    } else {
        LOG_INF("%s: loading existing LoRA adapter: %s\n", __func__, params.lora_adapters.back().path.c_str());
    }

    // --- Step 3: Load model + context (graph sized with LoRA nodes) ---
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto   llama_init = common_init_from_params(params);
    auto * model      = llama_init->model();
    auto * ctx        = llama_init->context();

    if (!model) {
        LOG_ERR("failed to load model\n");
        return 1;
    }

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    // Arch fallback if not in GGUF metadata
    if (arch.empty()) {
        char buf[256] = {};
        llama_model_desc(model, buf, sizeof(buf));
        arch = std::string(buf);
        arch = arch.substr(0, arch.find_first_of(" /"));
    }

    // --- Step 4: Mark the loaded adapter tensors as trainable ---
    // common_init_from_params loaded the adapter; params.lora_adapters[back].ptr
    // points to the live llama_adapter_lora with its own tensor copies in device
    // memory. Mark those tensors trainable so the optimizer graph includes them.
    {
        llama_adapter_lora * loaded = params.lora_adapters.back().ptr;
        if (!loaded) {
            LOG_ERR("%s: adapter was not loaded by common_init_from_params\n", __func__);
            return 1;
        }
        for (auto & kv : loaded->ab_map) {
            ggml_set_param(kv.second.a);  // lora_a → trainable
            ggml_set_param(kv.second.b);  // lora_b → trainable
        }
        // Point lt.ab at the live device tensors so save_adapter writes
        // the trained weights (not the original init tensors).
        lt.ab.clear();
        for (auto & kv : loaded->ab_map) {
            lt.ab[kv.first] = { kv.second.a, kv.second.b };
        }
    }

    // Remove temp init file when we created it (resume path has no init file)
    if (!resume_from_lora && !init_adapter_path.empty()) {
        std::remove(init_adapter_path.c_str());
    }

    // --- Step 5: Load dataset ---
    auto                tmpls   = common_chat_templates_init(model, "");
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    auto                samples = load_jsonl(params.train_file, vocab, tmpls.get());
    if (samples.empty()) {
        LOG_ERR("%s: no training samples loaded\n", __func__);
        return 1;
    }

    const int32_t     n_ctx   = llama_n_ctx(ctx);
    const llama_token bos     = llama_vocab_bos(llama_model_get_vocab(model));
    auto              dataset = build_dataset(samples, n_ctx, bos);
    if (!dataset) {
        return 1;
    }

    qlora_lr_schedule schedule{ &params.lr, params.lr_scheduler, params.warmup_steps, params.warmup_init_ratio, 0, 0 };

    // Initialize optimizer - our custom param filter restricts training to lora_a/b.
    struct llama_opt_params lopt_params{
        /*.n_ctx_train              =*/0,
        /*.param_filter             =*/lora_param_filter,
        /*.param_filter_ud          =*/nullptr,
        /*.get_opt_pars             =*/qlora_opt_lr_pars,
        /*.get_opt_pars_ud          =*/&schedule,
        /*.optimizer_type           =*/params.optimizer,
    };
    llama_opt_init(ctx, model, lopt_params);

    const int64_t ndata       = ggml_opt_dataset_ndata(dataset);
    const int64_t idata_split = ndata;
    if (idata_split <= 0) {
        LOG_ERR("%s: no training windows (ndata=%ld)\n", __func__, (long) ndata);
        return 1;
    }
    if (params.lr.epochs > 0 && idata_split > INT64_MAX / params.lr.epochs) {
        LOG_ERR("%s: total training step count overflows int64\n", __func__);
        return 1;
    }
    schedule.total_steps = idata_split * params.lr.epochs;
    if (schedule.total_steps <= 0) {
        LOG_ERR("%s: total training step count must be positive\n", __func__);
        return 1;
    }
    if (params.warmup_steps > schedule.total_steps) {
        LOG_ERR("%s: --warmup-steps %d exceeds total training steps %ld\n", __func__, params.warmup_steps,
                (long) schedule.total_steps);
        return 1;
    }
    if (params.lr.lr0 <= 0.0f || (params.lr_scheduler == "cosine" && params.lr.lr_min > params.lr.lr0)) {
        LOG_ERR("%s: learning-rate must be positive and cosine requires learning-rate-min <= learning-rate\n",
                __func__);
        return 1;
    }
    unsigned epoch_start  = 0;
    int64_t  window_start = 0;
    if (resume_requested) {
        if ((uint64_t) resume_state.epoch > UINT32_MAX) {
            LOG_ERR("%s: checkpoint epoch is too large\n", __func__);
            return 1;
        }
        epoch_start  = (unsigned) resume_state.epoch - 1;
        window_start = resume_state.window;

        if (resume_state.context_length >= 0 && resume_state.context_length != n_ctx) {
            LOG_ERR("%s: checkpoint context length is %ld, current context length is %d\n", __func__,
                    (long) resume_state.context_length, n_ctx);
            return 1;
        }
        if (resume_state.dataset_windows >= 0 && resume_state.dataset_windows != idata_split) {
            LOG_ERR("%s: checkpoint has %ld training windows, current dataset has %ld\n", __func__,
                    (long) resume_state.dataset_windows, (long) idata_split);
            return 1;
        }
        if (window_start > idata_split) {
            LOG_ERR("%s: checkpoint window %ld exceeds the %ld training windows in this dataset\n", __func__,
                    (long) window_start, (long) idata_split);
            return 1;
        }
        if (resume_state.has_metadata && resume_state.shuffle != params.shuffle_dataset) {
            LOG_ERR("%s: checkpoint shuffle setting does not match --shuffle-dataset\n", __func__);
            return 1;
        }
        if (window_start == idata_split) {
            ++epoch_start;
            window_start = 0;
        }
        LOG_INF(
            "%s: resuming SFT at epoch %u/%u after window %ld; next window is %ld/%ld; optimizer state starts fresh\n",
            __func__, epoch_start + 1, params.lr.epochs, (long) window_start, (long) window_start + 1,
            (long) idata_split);
    }
    if (epoch_start > params.lr.epochs) {
        LOG_ERR("%s: checkpoint epoch %u exceeds target epochs %u\n", __func__, epoch_start, params.lr.epochs);
        return 1;
    }
    const int64_t schedule_step = (int64_t) epoch_start * idata_split + window_start;
    if (schedule_step > schedule.total_steps) {
        LOG_ERR("%s: checkpoint step %ld exceeds total training steps %ld\n", __func__, (long) schedule_step,
                (long) schedule.total_steps);
        return 1;
    }
    if (resume_requested && resume_state.schedule_step >= 0 && resume_state.schedule_step != schedule_step) {
        LOG_ERR("%s: checkpoint scheduler step %ld does not match derived step %ld\n", __func__,
                (long) resume_state.schedule_step, (long) schedule_step);
        return 1;
    }
    schedule.step = schedule_step;

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    const int32_t n_ubatch       = llama_n_ubatch(ctx);
    const int32_t ubatch_per_ctx = (n_ubatch > 0) ? (n_ctx / n_ubatch) : 1;

    save_ctx sctx{ &lt,
                   &params.lora_out,
                   &arch,
                   &params.model.path,
                   lora_alpha,
                   params.save_every,
                   ubatch_per_ctx,
                   0,
                   0,
                   0,
                   idata_split,
                   n_ctx,
                   params.shuffle_dataset,
                   params.verbose_loss,
                   &schedule };
    g_save_ctx = &sctx;

    if (resume_requested && params.shuffle_dataset && epoch_start > 0) {
        for (unsigned epoch = 0; epoch < epoch_start; ++epoch) {
            llama_opt_dataset_shuffle(ctx, dataset, idata_split);
        }
        LOG_INF("%s: replayed %u completed epoch shuffle(s)\n", __func__, epoch_start);
    }

    const int64_t total_windows = ggml_opt_dataset_ndata(dataset);
    LOG_INF("%s: starting QLoRA training — rank=%d alpha=%.1f epochs=%d loss=%s\n", __func__, params.lora_rank,
            lora_alpha, params.lr.epochs, "response-only");
    LOG_INF("%s: dataset: %ld windows × %d ubatches = %ld steps per epoch  (n_ctx=%d n_ubatch=%d stride=%d)\n",
            __func__, (long) total_windows, ubatch_per_ctx, (long) (idata_split * ubatch_per_ctx), n_ctx, n_ubatch,
            n_ctx / 2);
    LOG_INF(
        "%s: lr scheduler=%s warmup_steps=%d warmup_init_ratio=%.3g total_steps=%ld start_step=%ld lr=%.3g "
        "lr_min=%.3g\n",
        __func__, params.lr_scheduler.c_str(), params.warmup_steps, (double) params.warmup_init_ratio,
        (long) schedule.total_steps, (long) schedule.step, (double) params.lr.lr0,
        (double) std::max(0.0f, params.lr.lr_min));
    if (params.save_every > 0) {
        LOG_INF("%s: will save checkpoint every %d windows → %s.ckptN.gguf\n", __func__, params.save_every,
                params.lora_out.c_str());
    }

    ggml_opt_epoch_callback cb_train = save_every_callback;

    for (params.lr.epoch = epoch_start; params.lr.epoch < params.lr.epochs; ++params.lr.epoch) {
        const int64_t idata_start  = params.lr.epoch == epoch_start ? window_start : 0;
        sctx.window_offset         = idata_start;
        sctx.last_saved            = idata_start;
        sctx.epoch                 = params.lr.epoch + 1;
        sctx.loss_cumulative_sum   = 0.0;
        sctx.loss_cumulative_count = 0;
        sctx.loss_epoch_sum        = 0.0;
        sctx.loss_epoch_count      = 0;
        llama_opt_epoch_range(ctx, dataset, result_train, result_eval, idata_start, idata_split, cb_train,
                              ggml_opt_epoch_callback_progress_bar, params.shuffle_dataset);
        fprintf(stderr, "\n");

        log_epoch_loss(params.lr.epoch + 1, params.lr.epochs, result_train);

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }

    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    // Save final trained adapter
    save_adapter(lt, params.lora_out, arch, lora_alpha, params.model.path);

    // Free scratch buffers only when we allocated them (not in resume path)
    if (lt.buf) {
        ggml_backend_buffer_free(lt.buf);
    }
    if (lt.ctx) {
        ggml_free(lt.ctx);
    }
    ggml_opt_dataset_free(dataset);
    llama_backend_free();

    return 0;
}
