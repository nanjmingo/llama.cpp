#include "adapter.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_transform_preserves_delta_adds_targets_and_keeps_frozen_tensors() {
    const std::string temp_root = (std::filesystem::temp_directory_path() /
        "llama-qlora-adapter-transform-test").string();
    const std::string input_path = temp_root + ".input.gguf";
    const std::string output_path = temp_root + ".output.gguf";
    std::remove(input_path.c_str());
    std::remove(output_path.c_str());

    ggml_context_ptr source_ctx(ggml_init({ 32 * ggml_tensor_overhead(), nullptr, true }));
    require(static_cast<bool>(source_ctx), "failed to allocate source tensor context");
    ggml_tensor * source_q_a = ggml_new_tensor_2d(source_ctx.get(), GGML_TYPE_F32, 4, 2);
    ggml_tensor * source_q_b = ggml_new_tensor_2d(source_ctx.get(), GGML_TYPE_F32, 2, 3);
    ggml_tensor * source_o_a = ggml_new_tensor_2d(source_ctx.get(), GGML_TYPE_F32, 4, 2);
    ggml_tensor * source_o_b = ggml_new_tensor_2d(source_ctx.get(), GGML_TYPE_F32, 2, 3);
    ggml_tensor * source_mtp_a = ggml_new_tensor_2d(source_ctx.get(), GGML_TYPE_F32, 4, 2);
    ggml_tensor * source_mtp_b = ggml_new_tensor_2d(source_ctx.get(), GGML_TYPE_F32, 2, 3);
    ggml_set_name(source_q_a, "blk.0.attn_q.weight.lora_a");
    ggml_set_name(source_q_b, "blk.0.attn_q.weight.lora_b");
    ggml_set_name(source_o_a, "blk.0.attn_output.weight.lora_a");
    ggml_set_name(source_o_b, "blk.0.attn_output.weight.lora_b");
    ggml_set_name(source_mtp_a, "blk.1.attn_q.weight.lora_a");
    ggml_set_name(source_mtp_b, "blk.1.attn_q.weight.lora_b");
    lora_tensor_refs source_refs = {
        { "blk.0.attn_q.weight", { source_q_a, source_q_b } },
        { "blk.0.attn_output.weight", { source_o_a, source_o_b } },
        { "blk.1.attn_q.weight", { source_mtp_a, source_mtp_b } },
    };
    ggml_backend_buffer_ptr source_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(source_ctx.get(), ggml_backend_cpu_buffer_type()));
    require(static_cast<bool>(source_buffer), "failed to allocate source tensor buffer");

    float * q_a_data = static_cast<float *>(source_q_a->data);
    float * q_b_data = static_cast<float *>(source_q_b->data);
    float * o_a_data = static_cast<float *>(source_o_a->data);
    float * o_b_data = static_cast<float *>(source_o_b->data);
    float * mtp_a_data = static_cast<float *>(source_mtp_a->data);
    float * mtp_b_data = static_cast<float *>(source_mtp_b->data);
    for (int i = 0; i < 8; ++i) {
        q_a_data[i] = static_cast<float>(i + 1);
        o_a_data[i] = static_cast<float>(i + 11);
        mtp_a_data[i] = static_cast<float>(i + 31);
    }
    for (int i = 0; i < 6; ++i) {
        q_b_data[i] = static_cast<float>(i + 1);
        o_b_data[i] = static_cast<float>(i + 21);
        mtp_b_data[i] = static_cast<float>(i + 41);
    }
    const std::vector<float> original_q_a(q_a_data, q_a_data + 8);
    const std::vector<float> original_q_b(q_b_data, q_b_data + 6);
    const std::vector<float> original_o_a(o_a_data, o_a_data + 8);
    const std::vector<float> original_o_b(o_b_data, o_b_data + 6);
    require(save_adapter(source_refs, input_path, "llama", 2.0f, "base.gguf"),
            "failed to save source adapter fixture");

    ggml_context_ptr model_ctx(ggml_init({ 20 * ggml_tensor_overhead(), nullptr, true }));
    require(static_cast<bool>(model_ctx), "failed to allocate model tensor context");
    ggml_tensor * q_weight = ggml_new_tensor_2d(model_ctx.get(), GGML_TYPE_F32, 4, 3);
    ggml_tensor * k_weight = ggml_new_tensor_2d(model_ctx.get(), GGML_TYPE_F32, 4, 3);
    ggml_tensor * o_weight = ggml_new_tensor_2d(model_ctx.get(), GGML_TYPE_F32, 4, 3);
    ggml_tensor * expert_weight = ggml_new_tensor_3d(model_ctx.get(), GGML_TYPE_F32, 4, 3, 2);
    ggml_tensor * mtp_weight = ggml_new_tensor_2d(model_ctx.get(), GGML_TYPE_F32, 4, 3);
    ggml_set_name(q_weight, "blk.0.attn_q.weight");
    ggml_set_name(k_weight, "blk.0.attn_k.weight");
    ggml_set_name(o_weight, "blk.0.attn_output.weight");
    ggml_set_name(expert_weight, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(mtp_weight, "blk.1.attn_q.weight");
    const std::vector<std::pair<std::string, ggml_tensor *>> model_tensors = {
        { "blk.0.attn_q.weight", q_weight },
        { "blk.0.attn_k.weight", k_weight },
        { "blk.0.attn_output.weight", o_weight },
        { "blk.0.ffn_gate_exps.weight", expert_weight },
        { "blk.1.attn_q.weight", mtp_weight },
    };

    std::mt19937 rng(42);
    lora_tensors transformed = transform_lora_tensors(
        input_path, "llama", model_tensors, { "attn_q", "attn_k" },
        3, 6.0f, 1.0f, rng, 1, 0);
    require(transformed.refs.size() == 4,
            "expected existing attn_q, frozen attn_output/MTP tensors, and newly added attn_k");
    require(lora_tensor_matches_targets("blk.0.attn_q.weight", { "attn_q", "attn_k" }, 0),
            "requested target should be trainable");
    require(!lora_tensor_matches_targets("blk.0.attn_output.weight", { "attn_q", "attn_k" }, 0),
            "existing non-target adapter tensor should be frozen");
    require(!lora_tensor_matches_targets("blk.0.ffn_gate_exps.weight", { "attn_q", "attn_k" }, 0),
            "unselected MoE expert tensors must remain frozen");
    require(!lora_tensor_matches_targets("blk.0.ffn_gate_exps.weight", { "ffn_gate" }, 0),
            "dense FFN targets must not select MoE expert tensors");
    require(!lora_tensor_matches_targets("blk.0.ffn_gate_up_exps.weight", { "ffn_gate_up" }, 0),
            "dense fused FFN targets must not select fused MoE expert tensors");
    require(lora_tensor_matches_targets("blk.0.ffn_gate_up.weight", { "ffn_gate" }, 0) &&
            lora_tensor_matches_targets("blk.0.ffn_gate_up.weight", { "ffn_up" }, 0),
            "dense FFN targets must retain support for fused gate/up tensors");
    require(lora_tensor_matches_targets("blk.0.ffn_gate_exps.weight", { "ffn_gate_exps" }, 0),
            "explicit MoE expert targets must be selectable");
    require(!lora_tensor_matches_targets("blk.1.attn_q.weight", { "attn_q", "attn_k" }, 0, 1),
            "MTP/NextN tensors at or beyond the main model layer count must be frozen");
    require(lora_tensor_matches_targets("blk.0.attn_q.weight", { "attn_q", "attn_k" }, 0, 1),
            "main-model target tensors should remain trainable");
    require(validate_lora_targets(model_tensors, { "attn_q", "ffn_gate_exps" }, 1, 0),
            "explicit MoE expert targets should pass target validation");
    require(validate_lora_adapter_rank(input_path, "llama", 3),
            "rank expansion preflight should pass");
    require(!validate_lora_adapter_rank(input_path, "llama", 1),
            "rank reduction preflight should fail");

    const auto q_it = transformed.refs.find("blk.0.attn_q.weight");
    require(q_it != transformed.refs.end(), "transformed adapter is missing attn_q");
    ggml_tensor * new_q_a = q_it->second.first;
    ggml_tensor * new_q_b = q_it->second.second;
    require(new_q_a->ne[0] == 4 && new_q_a->ne[1] == 3, "attn_q A rank was not expanded");
    require(new_q_b->ne[0] == 3 && new_q_b->ne[1] == 3, "attn_q B rank was not expanded");

    const float * new_q_a_data = static_cast<const float *>(new_q_a->data);
    const float * new_q_b_data = static_cast<const float *>(new_q_b->data);
    for (int64_t r = 0; r < 2; ++r) {
        for (int64_t i = 0; i < 4; ++i) {
            require(new_q_a_data[i + 4 * r] == original_q_a[i + 4 * r],
                    "rank expansion changed the existing A components");
        }
    }
    for (int64_t j = 0; j < 3; ++j) {
        for (int64_t r = 0; r < 2; ++r) {
            require(std::fabs(new_q_b_data[r + 3 * j] - original_q_b[r + 2 * j] * 0.5f) < 1e-6f,
                    "B was not rescaled to preserve the effective weight delta");
        }
        require(new_q_b_data[2 + 3 * j] == 0.0f, "new B component must start at zero");
    }

    const float old_scale = 2.0f / 2.0f;
    const float new_scale = 6.0f / 3.0f;
    for (int64_t j = 0; j < 3; ++j) {
        for (int64_t i = 0; i < 4; ++i) {
            float old_delta = 0.0f;
            float new_delta = 0.0f;
            for (int64_t r = 0; r < 2; ++r) {
                old_delta += original_q_a[i + 4 * r] * original_q_b[r + 2 * j] * old_scale;
            }
            for (int64_t r = 0; r < 3; ++r) {
                new_delta += new_q_a_data[i + 4 * r] * new_q_b_data[r + 3 * j] * new_scale;
            }
            require(std::fabs(old_delta - new_delta) < 1e-5f,
                    "rank/alpha conversion changed the initial effective weight delta");
        }
    }

    const auto k_it = transformed.refs.find("blk.0.attn_k.weight");
    require(k_it != transformed.refs.end(), "missing target tensor was not added");
    const float * added_a = static_cast<const float *>(k_it->second.first->data);
    const float * added_b = static_cast<const float *>(k_it->second.second->data);
    const float added_a_bound = 1.0f / std::sqrt(static_cast<float>(k_weight->ne[0]));
    for (int64_t i = 0; i < ggml_nelements(k_it->second.first); ++i) {
        require(std::fabs(added_a[i]) <= added_a_bound,
                "new target A must use Kaiming-uniform initialization");
    }
    for (int64_t i = 0; i < ggml_nelements(k_it->second.second); ++i) {
        require(added_b[i] == 0.0f, "new target B must be zero initialized");
    }

    const auto o_it = transformed.refs.find("blk.0.attn_output.weight");
    require(o_it != transformed.refs.end(), "non-target source tensor was not retained");
    const float * frozen_a = static_cast<const float *>(o_it->second.first->data);
    const float * frozen_b = static_cast<const float *>(o_it->second.second->data);
    for (int i = 0; i < 8; ++i) {
        require(frozen_a[i] == original_o_a[i], "frozen tensor A was changed by conversion");
    }
    for (int64_t j = 0; j < 3; ++j) {
        for (int64_t r = 0; r < 2; ++r) {
            require(std::fabs(frozen_b[r + 3 * j] - original_o_b[r + 2 * j] * 0.5f) < 1e-6f,
                    "frozen tensor B was not converted to the requested alpha/rank");
        }
        require(frozen_b[2 + 3 * j] == 0.0f, "frozen tensor's extended B component must be zero");
    }

    require(save_adapter(transformed.refs, output_path, "llama", 6.0f, "base.gguf"),
            "failed to save converted adapter fixture");
    ggml_context * output_tensor_ctx_raw = nullptr;
    gguf_context_ptr output_gguf(gguf_init_from_file(output_path.c_str(), { true, &output_tensor_ctx_raw }));
    ggml_context_ptr output_tensor_ctx(output_tensor_ctx_raw);
    require(static_cast<bool>(output_gguf), "failed to read converted adapter fixture");
    const int64_t alpha_id = gguf_find_key(output_gguf.get(), "adapter.lora.alpha");
    require(alpha_id >= 0 && std::fabs(gguf_get_val_f32(output_gguf.get(), alpha_id) - 6.0f) < 1e-6f,
            "converted adapter did not store the requested alpha");
    require(gguf_find_tensor(output_gguf.get(), "blk.0.attn_k.weight.lora_a") >= 0,
            "converted adapter file is missing the added target");
    require(gguf_find_tensor(output_gguf.get(), "blk.0.attn_output.weight.lora_a") >= 0 &&
            gguf_find_tensor(output_gguf.get(), "blk.0.attn_output.weight.lora_b") >= 0,
            "converted adapter file dropped the frozen source tensor");
    require(gguf_find_tensor(output_gguf.get(), "blk.1.attn_q.weight.lora_a") >= 0 &&
            gguf_find_tensor(output_gguf.get(), "blk.1.attn_q.weight.lora_b") >= 0,
            "converted adapter file dropped the frozen MTP/NextN tensor");

    lora_tensors rejected = transform_lora_tensors(
        input_path, "llama", model_tensors, { "attn_q", "attn_k" }, 1, 1.0f, 1.0f, rng, 1, 0);
    require(rejected.refs.empty(), "rank reduction should be rejected");

    lora_tensors bad_target = transform_lora_tensors(
        input_path, "llama", model_tensors, { "attn_q", "not_a_real_tensor" },
        3, 3.0f, 1.0f, rng, 1, 0);
    require(bad_target.refs.empty(), "an unmatched target should be rejected");

    std::remove(input_path.c_str());
    std::remove(output_path.c_str());
}

void test_peft_default_lora_initialization() {
    ggml_context_ptr model_ctx(ggml_init({ 8 * ggml_tensor_overhead(), nullptr, true }));
    require(static_cast<bool>(model_ctx), "failed to allocate model context for LoRA initialization test");
    ggml_tensor * q_weight = ggml_new_tensor_2d(model_ctx.get(), GGML_TYPE_F32, 16, 8);
    ggml_tensor * token_embd_weight = ggml_new_tensor_2d(model_ctx.get(), GGML_TYPE_F32, 8, 32);
    ggml_set_name(q_weight, "blk.0.attn_q.weight");
    ggml_set_name(token_embd_weight, "token_embd.weight");
    const std::vector<std::pair<std::string, ggml_tensor *>> model_tensors = {
        { "blk.0.attn_q.weight", q_weight },
        { "token_embd.weight", token_embd_weight },
    };

    std::mt19937 rng(123);
    lora_tensors initialized = alloc_lora_tensors(
        model_tensors, { "attn_q", "token_embd.weight" }, 4, rng, 1);
    require(initialized.refs.size() == 2, "expected linear and embedding LoRA weights");

    const auto q_it = initialized.refs.find("blk.0.attn_q.weight");
    require(q_it != initialized.refs.end(), "missing initialized linear LoRA weights");
    const float * q_a = static_cast<const float *>(q_it->second.first->data);
    const float * q_b = static_cast<const float *>(q_it->second.second->data);
    const float q_a_bound = 1.0f / std::sqrt(static_cast<float>(q_weight->ne[0]));
    bool q_a_nonzero = false;
    for (int64_t i = 0; i < ggml_nelements(q_it->second.first); ++i) {
        require(std::fabs(q_a[i]) <= q_a_bound,
                "linear LoRA A must use the PEFT Kaiming-uniform distribution");
        q_a_nonzero = q_a_nonzero || q_a[i] != 0.0f;
    }
    require(q_a_nonzero, "linear LoRA A should not be all zeros");
    for (int64_t i = 0; i < ggml_nelements(q_it->second.second); ++i) {
        require(q_b[i] == 0.0f, "linear LoRA B must be zero initialized");
    }

    const auto emb_it = initialized.refs.find("token_embd.weight");
    require(emb_it != initialized.refs.end(), "missing initialized embedding LoRA weights");
    const float * emb_a = static_cast<const float *>(emb_it->second.first->data);
    const float * emb_b = static_cast<const float *>(emb_it->second.second->data);
    for (int64_t i = 0; i < ggml_nelements(emb_it->second.first); ++i) {
        require(emb_a[i] == 0.0f, "embedding LoRA A must be zero initialized");
    }
    float emb_b_square_sum = 0.0f;
    for (int64_t i = 0; i < ggml_nelements(emb_it->second.second); ++i) {
        require(std::isfinite(emb_b[i]), "embedding LoRA B contains a non-finite value");
        emb_b_square_sum += emb_b[i] * emb_b[i];
    }
    const float emb_b_rms = std::sqrt(emb_b_square_sum / ggml_nelements(emb_it->second.second));
    require(emb_b_rms > 0.5f && emb_b_rms < 1.5f,
            "embedding LoRA B must use a unit-standard-deviation normal distribution");
}

void test_lora_storage_types() {
    ggml_context_ptr model_ctx(ggml_init({ 8 * ggml_tensor_overhead(), nullptr, true }));
    require(static_cast<bool>(model_ctx), "failed to allocate model context for type test");
    ggml_tensor * q_weight = ggml_new_tensor_2d(model_ctx.get(), GGML_TYPE_F32, 64, 64);
    ggml_set_name(q_weight, "blk.0.attn_q.weight");
    const std::vector<std::pair<std::string, ggml_tensor *>> model_tensors = {
        { "blk.0.attn_q.weight", q_weight },
    };

    const std::vector<std::pair<ggml_type, int32_t>> types = {
        { GGML_TYPE_F32, 8 },
        { GGML_TYPE_F16, 8 },
        { GGML_TYPE_BF16, 8 },
    };
    for (const auto & selected : types) {
        std::mt19937 rng(1234);
        lora_tensors adapter = alloc_lora_tensors(model_tensors, { "attn_q" }, selected.second, rng, 1, 0, selected.first);
        require(adapter.refs.size() == 1, "LoRA allocation failed for selected type " + std::string(ggml_type_name(selected.first)));
        const auto pair = adapter.refs.at("blk.0.attn_q.weight");
        require(pair.first->type == selected.first && pair.second->type == selected.first,
                "LoRA tensor types do not match selected storage type");
        std::vector<float> a(static_cast<size_t>(ggml_nelements(pair.first)));
        std::vector<float> b(static_cast<size_t>(ggml_nelements(pair.second)));
        if (selected.first == GGML_TYPE_F32) {
            std::memcpy(a.data(), pair.first->data, a.size() * sizeof(float));
            std::memcpy(b.data(), pair.second->data, b.size() * sizeof(float));
        } else {
            const ggml_type_traits * traits = ggml_get_type_traits(selected.first);
            require(traits && traits->to_float, "selected LoRA type has no float conversion");
            traits->to_float(pair.first->data, a.data(), ggml_nelements(pair.first));
            traits->to_float(pair.second->data, b.data(), ggml_nelements(pair.second));
        }
        for (float value : a) require(std::isfinite(value), "LoRA A contains a non-finite value");
        for (float value : b) require(std::isfinite(value), "LoRA B contains a non-finite value");

        const std::string path = (std::filesystem::temp_directory_path() /
            ("llama-qlora-type-" + std::string(ggml_type_name(selected.first)) + ".gguf")).string();
        std::remove(path.c_str());
        require(save_adapter(adapter.refs, path, "llama", static_cast<float>(selected.second), "base.gguf"),
                "failed to save adapter with selected type");
        gguf_context_ptr output(gguf_init_from_file(path.c_str(), { true, nullptr }));
        require(static_cast<bool>(output), "failed to reopen adapter with selected type");
        const int64_t saved_a = gguf_find_tensor(output.get(), "blk.0.attn_q.weight.lora_a");
        const int64_t saved_b = gguf_find_tensor(output.get(), "blk.0.attn_q.weight.lora_b");
        require(saved_a >= 0 && saved_b >= 0, "saved adapter is missing a LoRA tensor");
        require(gguf_get_tensor_type(output.get(), saved_a) == selected.first &&
                gguf_get_tensor_type(output.get(), saved_b) == selected.first,
                "saved adapter did not preserve selected tensor type");
        output.reset();

        std::mt19937 convert_rng(4321);
        lora_tensors converted = transform_lora_tensors(
            path, "llama", model_tensors, { "attn_q" }, selected.second,
            static_cast<float>(selected.second), 1.0f, convert_rng, 1, 0, GGML_TYPE_F32);
        require(converted.refs.size() == 1, "selected adapter type could not be converted back to F32");
        require(converted.refs.at("blk.0.attn_q.weight").first->type == GGML_TYPE_F32 &&
                converted.refs.at("blk.0.attn_q.weight").second->type == GGML_TYPE_F32,
                "adapter type conversion did not produce F32 tensors");
        std::remove(path.c_str());
    }

    std::mt19937 rng(42);
    lora_tensors unsupported_q8 = alloc_lora_tensors(model_tensors, { "attn_q" }, 16, rng, 1, 0, GGML_TYPE_Q8_0);
    require(unsupported_q8.refs.empty(), "Q8_0 LoRA storage type should be rejected");
}

void test_moe_expert_lora_allocation_and_transform() {
    const std::string input_path = (std::filesystem::temp_directory_path() /
        "llama-qlora-moe-expert-adapter-test.gguf").string();
    std::remove(input_path.c_str());

    constexpr int64_t input_dim = 4;
    constexpr int64_t source_rank = 2;
    constexpr int64_t output_dim = 3;
    constexpr int64_t n_expert = 3;
    constexpr int64_t target_rank = 3;
    constexpr float source_alpha = 2.0f;
    constexpr float target_alpha = 6.0f;

    ggml_context_ptr source_ctx(ggml_init({ 8 * ggml_tensor_overhead(), nullptr, true }));
    require(static_cast<bool>(source_ctx), "failed to allocate MoE source context");
    ggml_tensor * expert_a = ggml_new_tensor_3d(
        source_ctx.get(), GGML_TYPE_F32, input_dim, source_rank, n_expert);
    ggml_tensor * expert_b = ggml_new_tensor_3d(
        source_ctx.get(), GGML_TYPE_F32, source_rank, output_dim, n_expert);
    ggml_set_name(expert_a, "blk.0.ffn_gate_exps.weight.lora_a");
    ggml_set_name(expert_b, "blk.0.ffn_gate_exps.weight.lora_b");
    lora_tensor_refs source_refs = {
        { "blk.0.ffn_gate_exps.weight", { expert_a, expert_b } },
    };
    ggml_backend_buffer_ptr source_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(source_ctx.get(), ggml_backend_cpu_buffer_type()));
    require(static_cast<bool>(source_buffer), "failed to allocate MoE source tensor buffer");

    float * source_a = static_cast<float *>(expert_a->data);
    float * source_b = static_cast<float *>(expert_b->data);
    for (int64_t e = 0; e < n_expert; ++e) {
        for (int64_t r = 0; r < source_rank; ++r) {
            for (int64_t i = 0; i < input_dim; ++i) {
                source_a[i + input_dim * (r + source_rank * e)] =
                    0.01f * static_cast<float>(1 + i + 2 * r + 5 * e);
            }
            for (int64_t j = 0; j < output_dim; ++j) {
                source_b[r + source_rank * (j + output_dim * e)] =
                    0.1f * static_cast<float>(1 + r + 2 * j + 4 * e);
            }
        }
    }
    require(save_adapter(source_refs, input_path, "llama", source_alpha, "base.gguf"),
            "failed to save MoE source adapter fixture");

    ggml_context_ptr model_ctx(ggml_init({ 8 * ggml_tensor_overhead(), nullptr, true }));
    require(static_cast<bool>(model_ctx), "failed to allocate target model context");
    ggml_tensor * expert_weight = ggml_new_tensor_3d(
        model_ctx.get(), GGML_TYPE_F32, input_dim, output_dim, n_expert);
    ggml_set_name(expert_weight, "blk.0.ffn_gate_exps.weight");
    const std::vector<std::pair<std::string, ggml_tensor *>> model_tensors = {
        { "blk.0.ffn_gate_exps.weight", expert_weight },
    };

    require(validate_lora_targets(model_tensors, { "ffn_gate_exps" }, 1, 0),
            "explicit MoE expert targets should pass target validation");
    std::mt19937 allocation_rng(42);
    lora_tensors allocated = alloc_lora_tensors(
        model_tensors, { "ffn_gate_exps" }, target_rank, allocation_rng, 1);
    require(allocated.refs.size() == 1, "MoE expert LoRA allocation failed");
    const auto allocated_pair = allocated.refs.at("blk.0.ffn_gate_exps.weight");
    require(allocated_pair.first->ne[0] == input_dim && allocated_pair.first->ne[1] == target_rank &&
            allocated_pair.first->ne[2] == n_expert,
            "MoE LoRA A must have shape [input_dim, rank, n_expert]");
    require(allocated_pair.second->ne[0] == target_rank && allocated_pair.second->ne[1] == output_dim &&
            allocated_pair.second->ne[2] == n_expert,
            "MoE LoRA B must have shape [rank, output_dim, n_expert]");

    require(validate_lora_adapter_rank(input_path, "llama", target_rank),
            "rank expansion preflight should accept MoE expert adapters");
    require(!validate_lora_adapter_rank(input_path, "llama", 1),
            "rank reduction preflight should reject MoE expert adapters");

    std::mt19937 transform_rng(1234);
    lora_tensors transformed = transform_lora_tensors(
        input_path, "llama", model_tensors, { "ffn_gate_exps" }, target_rank,
        target_alpha, 1.0f, transform_rng, 1, 0);
    require(transformed.refs.size() == 1, "MoE expert adapter transformation failed");
    const auto transformed_pair = transformed.refs.at("blk.0.ffn_gate_exps.weight");
    ggml_tensor * transformed_a = transformed_pair.first;
    ggml_tensor * transformed_b = transformed_pair.second;
    require(transformed_a->ne[0] == input_dim && transformed_a->ne[1] == target_rank &&
            transformed_a->ne[2] == n_expert,
            "transformed MoE LoRA A shape is incorrect");
    require(transformed_b->ne[0] == target_rank && transformed_b->ne[1] == output_dim &&
            transformed_b->ne[2] == n_expert,
            "transformed MoE LoRA B shape is incorrect");

    const float * converted_a = static_cast<const float *>(transformed_a->data);
    const float * converted_b = static_cast<const float *>(transformed_b->data);
    const float source_scale = source_alpha / static_cast<float>(source_rank);
    const float target_scale = target_alpha / static_cast<float>(target_rank);
    for (int64_t e = 0; e < n_expert; ++e) {
        for (int64_t r = 0; r < source_rank; ++r) {
            for (int64_t i = 0; i < input_dim; ++i) {
                const size_t source_offset = static_cast<size_t>(i + input_dim * (r + source_rank * e));
                const size_t target_offset = static_cast<size_t>(i + input_dim * (r + target_rank * e));
                require(converted_a[target_offset] == source_a[source_offset],
                        "rank expansion changed an existing expert A component");
            }
            for (int64_t j = 0; j < output_dim; ++j) {
                const size_t source_offset = static_cast<size_t>(r + source_rank * (j + output_dim * e));
                const size_t target_offset = static_cast<size_t>(r + target_rank * (j + output_dim * e));
                require(std::fabs(converted_b[target_offset] - source_b[source_offset] * 0.5f) < 1e-7f,
                        "rank expansion did not preserve expert B scale");
            }
        }
        for (int64_t j = 0; j < output_dim; ++j) {
            require(converted_b[target_rank - 1 + target_rank * (j + output_dim * e)] == 0.0f,
                    "new rank components of expert B must remain zero initialized");
        }
        for (int64_t j = 0; j < output_dim; ++j) {
            for (int64_t i = 0; i < input_dim; ++i) {
                float old_delta = 0.0f;
                float new_delta = 0.0f;
                for (int64_t r = 0; r < source_rank; ++r) {
                    old_delta += source_b[r + source_rank * (j + output_dim * e)] *
                                 source_a[i + input_dim * (r + source_rank * e)];
                }
                for (int64_t r = 0; r < target_rank; ++r) {
                    new_delta += converted_b[r + target_rank * (j + output_dim * e)] *
                                 converted_a[i + input_dim * (r + target_rank * e)];
                }
                require(std::fabs(source_scale * old_delta - target_scale * new_delta) < 1e-6f,
                        "MoE expert rank expansion changed the effective LoRA delta");
            }
        }
    }

    std::remove(input_path.c_str());
}

} // namespace

int main() {
    try {
        llama_backend_init();
        test_transform_preserves_delta_adds_targets_and_keeps_frozen_tensors();
        test_peft_default_lora_initialization();
        test_lora_storage_types();
        test_moe_expert_lora_allocation_and_transform();
        llama_backend_free();
        std::cout << "adapter transformation tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        llama_backend_free();
        return 1;
    }
}
