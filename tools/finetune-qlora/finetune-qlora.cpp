// QLoRA fine-tuning for quantized GGUF models.
//
// LoRA A/B tensors can use F32, F16, or BF16 storage. QLoRA-only arguments are parsed
// locally so this example does not need to modify common/arg.h or common/arg.cpp.
// Resumable checkpoints keep optimizer and training progress separate from adapter weights.

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "json.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-opt.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "log.h"
#include "adapter.h"
#include "args.h"
#include "dataset.h"
#include "schedule.h"

// Internal adapter struct is needed to mark the loaded LoRA tensors trainable.
#include "../../src/llama-adapter.h"
#include "../../src/llama-model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cinttypes>
#include <clocale>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <iomanip>
#include <limits>
#include <system_error>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <cctype>
#include <vector>

namespace {

enum ggml_type qlora_type_from_name(const std::string & name) {
    if (name == "f32") return GGML_TYPE_F32;
    if (name == "f16") return GGML_TYPE_F16;
    if (name == "bf16") return GGML_TYPE_BF16;
    throw std::invalid_argument("unsupported tensor type: " + name);
}

std::string step_name(const char * prefix, int64_t step, const char * suffix = "") {
    std::ostringstream name;
    name << prefix << std::setw(6) << std::setfill('0') << step << suffix;
    return name.str();
}

bool parse_step_name(const std::string & name,
                     const std::string & prefix,
                     const std::string & suffix,
                     int64_t & step) {
    if (name.size() <= prefix.size() + suffix.size() ||
        name.compare(0, prefix.size(), prefix) != 0 ||
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return false;
    }
    const std::string digits = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
    if (digits.empty() || !std::all_of(digits.begin(), digits.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) {
        return false;
    }
    try {
        size_t consumed = 0;
        const long long value = std::stoll(digits, &consumed);
        if (consumed != digits.size() || value < 0) return false;
        step = static_cast<int64_t>(value);
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

std::string unique_init_adapter_path(const std::string & output_path) {
    const std::string base = (std::filesystem::path(output_path) / ".adapter-init").string();
    std::string candidate = base + ".gguf";
    for (uint64_t suffix = 1; ; ++suffix) {
        std::error_code ec;
        const bool exists = std::filesystem::exists(candidate, ec);
        if (ec) {
            LOG_ERR("%s: cannot check whether temporary adapter exists: %s\n", __func__, candidate.c_str());
            return {};
        }
        const bool temp_exists = std::filesystem::exists(candidate + ".tmp", ec);
        if (ec) {
            LOG_ERR("%s: cannot check whether temporary adapter exists: %s\n", __func__, (candidate + ".tmp").c_str());
            return {};
        }
        if (!exists && !temp_exists) {
            return candidate;
        }
        candidate = base + "." + std::to_string(suffix) + ".gguf";
    }
}

struct ggml_opt_dataset_deleter {
    void operator()(ggml_opt_dataset * dataset) const {
        ggml_opt_dataset_free(dataset);
    }
};

struct ggml_opt_result_deleter {
    void operator()(ggml_opt_result * result) const {
        ggml_opt_result_free(result);
    }
};

using ggml_opt_dataset_ptr = std::unique_ptr<ggml_opt_dataset, ggml_opt_dataset_deleter>;
using ggml_opt_result_ptr = std::unique_ptr<ggml_opt_result, ggml_opt_result_deleter>;

struct llama_backend_scope {
    ~llama_backend_scope() {
        llama_backend_free();
    }
};

struct remove_file_scope {
    std::string path;

    void remove_now() {
        if (!path.empty()) {
            std::remove(path.c_str());
            path.clear();
        }
    }

    ~remove_file_scope() {
        remove_now();
    }
};

struct qlora_checkpoint {
    std::filesystem::path directory;
    common_json metadata = common_json::object();
    std::vector<uint8_t> optimizer_state;
    std::vector<int64_t> dataset_permutation;
    int64_t next_epoch = 0;
    int64_t next_window = 0;
    int64_t scheduler_step = 0;
};

struct dataset_file_identity {
    std::string path;
    int64_t size = -1;
};

std::string canonical_path_string(const std::filesystem::path & path) {
    std::error_code ec;
    std::filesystem::path canonical = std::filesystem::weakly_canonical(path, ec);
    if (ec) {
        ec.clear();
        canonical = std::filesystem::absolute(path, ec);
        if (ec) canonical = path;
    }
    return canonical.lexically_normal().string();
}

bool get_dataset_file_identity(const std::string & path, dataset_file_identity & result) {
    std::error_code ec;
    const std::filesystem::path file_path(path);
    const uintmax_t size = std::filesystem::file_size(file_path, ec);
    if (ec || size > static_cast<uintmax_t>(INT64_MAX)) return false;
    result.path = canonical_path_string(file_path);
    result.size = static_cast<int64_t>(size);
    return true;
}

common_json make_dataset_file_metadata(const qlora_params & qparams) {
    common_json result = common_json::array();
    for (const std::string & path : qparams.dataset.input_paths) {
        dataset_file_identity identity;
        if (!get_dataset_file_identity(path, identity)) {
            throw std::runtime_error("cannot inspect training dataset file: " + path);
        }
        common_json item = common_json::object();
        item["path"] = identity.path;
        item["size"] = identity.size;
        result.push_back(item);
    }
    return result;
}

bool write_text_file(const std::filesystem::path & path, const std::string & contents) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) return false;
    stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    stream.flush();
    const bool write_ok = static_cast<bool>(stream);
    stream.close();
    return write_ok && !stream.fail();
}

bool read_text_file(const std::filesystem::path & path, std::string & contents) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return false;
    contents.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    return stream.eof() || static_cast<bool>(stream);
}

bool is_checkpoint_directory(const std::filesystem::path & directory, int64_t * optimizer_step = nullptr) {
    std::error_code ec;
    if (!std::filesystem::is_directory(directory, ec) || ec ||
        !std::filesystem::is_regular_file(directory / "adapter.gguf", ec) || ec ||
        !std::filesystem::is_regular_file(directory / "optimizer.gguf", ec) || ec ||
        !std::filesystem::is_regular_file(directory / "training-state.json", ec) || ec ||
        !std::filesystem::is_regular_file(directory / "manifest.json", ec) || ec) {
        return false;
    }

    std::string manifest_text;
    std::string state_text;
    if (!read_text_file(directory / "manifest.json", manifest_text) ||
        !read_text_file(directory / "training-state.json", state_text)) {
        return false;
    }
    try {
        const common_json manifest = common_json::parse(manifest_text);
        const common_json state = common_json::parse(state_text);
        if (manifest.at("format_version").get<int>() != 2 ||
            manifest.at("adapter_file").get<std::string>() != "adapter.gguf" ||
            manifest.at("optimizer_file").get<std::string>() != "optimizer.gguf" ||
            manifest.at("training_state_file").get<std::string>() != "training-state.json" ||
            state.at("format_version").get<int>() != 3) {
            return false;
        }
        const int64_t manifest_step = manifest.at("optimizer_step").get<int64_t>();
        if (manifest_step < 0 || state.at("scheduler_step").get<int64_t>() != manifest_step) {
            return false;
        }
        int64_t directory_step = -1;
        if (parse_step_name(directory.filename().string(), "checkpoint-", "", directory_step) &&
            directory_step != manifest_step) {
            return false;
        }
        if (optimizer_step) *optimizer_step = manifest_step;
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

bool recover_checkpoint_directories(const std::filesystem::path & root) {
    std::error_code ec;
    if (!std::filesystem::exists(root, ec)) return !ec;
    if (ec || !std::filesystem::is_directory(root, ec) || ec) return false;

    std::vector<std::filesystem::path> backups;
    std::vector<std::filesystem::path> temporary;
    for (std::filesystem::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        int64_t step = 0;
        if (parse_step_name(name, "checkpoint-", ".previous", step)) {
            backups.push_back(it->path());
        } else if (name.rfind(".checkpoint-", 0) == 0 && name.find(".tmp-") != std::string::npos) {
            temporary.push_back(it->path());
        }
    }
    if (ec) return false;

    for (const std::filesystem::path & backup : backups) {
        const std::string backup_name = backup.filename().string();
        int64_t step = 0;
        if (!parse_step_name(backup_name, "checkpoint-", ".previous", step)) continue;
        const std::filesystem::path final_dir = root / step_name("checkpoint-", step);
        const bool final_valid = is_checkpoint_directory(final_dir);
        const bool backup_valid = is_checkpoint_directory(backup);
        if (final_valid) {
            std::filesystem::remove_all(backup, ec);
            if (ec) return false;
            continue;
        }

        std::filesystem::path completed_temp;
        for (const std::filesystem::path & candidate : temporary) {
            const std::string candidate_name = candidate.filename().string();
            const std::string expected_prefix = ".checkpoint-" + step_name("", step) + ".tmp-";
            if (candidate_name.rfind(expected_prefix, 0) == 0 && is_checkpoint_directory(candidate)) {
                completed_temp = candidate;
                break;
            }
        }

        if (!completed_temp.empty()) {
            if (std::filesystem::exists(final_dir, ec)) {
                std::filesystem::remove_all(final_dir, ec);
                if (ec) return false;
            }
            std::filesystem::rename(completed_temp, final_dir, ec);
            if (ec) return false;
            std::filesystem::remove_all(backup, ec);
            if (ec) return false;
        } else if (backup_valid) {
            if (std::filesystem::exists(final_dir, ec)) {
                std::filesystem::remove_all(final_dir, ec);
                if (ec) return false;
            }
            std::filesystem::rename(backup, final_dir, ec);
            if (ec) return false;
        }
    }

    for (const std::filesystem::path & candidate : temporary) {
        if (!std::filesystem::exists(candidate, ec) || ec) continue;
        const std::string candidate_name = candidate.filename().string();
        const size_t marker = candidate_name.find(".tmp-");
        if (marker == std::string::npos) continue;
        const std::string checkpoint_name = candidate_name.substr(1, marker - 1);
        int64_t step = 0;
        if (!parse_step_name(checkpoint_name, "checkpoint-", "", step)) continue;
        const std::filesystem::path final_dir = root / step_name("checkpoint-", step);
        if (is_checkpoint_directory(final_dir)) {
            std::filesystem::remove_all(candidate, ec);
            if (ec) return false;
            continue;
        }
        if (is_checkpoint_directory(candidate)) {
            if (std::filesystem::exists(final_dir, ec)) {
                std::filesystem::remove_all(final_dir, ec);
                if (ec) return false;
            }
            ec.clear();
            std::filesystem::rename(candidate, final_dir, ec);
            if (ec) return false;
            const std::filesystem::path backup = final_dir.string() + ".previous";
            if (std::filesystem::exists(backup, ec) && !ec) {
                std::filesystem::remove_all(backup, ec);
                if (ec) return false;
            }
        } else {
            std::filesystem::remove_all(candidate, ec);
            if (ec) return false;
        }
    }
    return true;
}

bool collect_checkpoints(const std::filesystem::path & root,
                         std::vector<std::pair<int64_t, std::filesystem::path>> & checkpoints) {
    checkpoints.clear();
    std::error_code ec;
    for (std::filesystem::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
        int64_t step = 0;
        int64_t manifest_step = -1;
        if (parse_step_name(it->path().filename().string(), "checkpoint-", "", step) &&
            is_checkpoint_directory(it->path(), &manifest_step) && manifest_step == step) {
            checkpoints.emplace_back(step, it->path());
        }
    }
    if (ec) return false;
    std::sort(checkpoints.begin(), checkpoints.end(), [](const auto & a, const auto & b) {
        return a.first < b.first;
    });
    return true;
}

bool prune_checkpoints(const std::filesystem::path & root, int32_t limit) {
    if (limit <= 0) return true;
    std::vector<std::pair<int64_t, std::filesystem::path>> checkpoints;
    if (!collect_checkpoints(root, checkpoints)) return false;
    while (checkpoints.size() > static_cast<size_t>(limit)) {
        std::error_code ec;
        std::filesystem::remove_all(checkpoints.front().second, ec);
        if (ec) return false;
        checkpoints.erase(checkpoints.begin());
    }
    return true;
}

bool prune_snapshots(const std::filesystem::path & root, int32_t limit) {
    if (limit <= 0) return true;
    std::vector<std::pair<int64_t, std::filesystem::path>> snapshots;
    std::error_code ec;
    for (std::filesystem::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
        int64_t step = 0;
        if (parse_step_name(it->path().filename().string(), "snapshot-", ".gguf", step) &&
            std::filesystem::is_regular_file(it->path(), ec) && !ec) {
            snapshots.emplace_back(step, it->path());
        }
        if (ec) return false;
    }
    if (ec) return false;
    std::sort(snapshots.begin(), snapshots.end(), [](const auto & a, const auto & b) {
        return a.first < b.first;
    });
    while (snapshots.size() > static_cast<size_t>(limit)) {
        std::filesystem::remove(snapshots.front().second, ec);
        if (ec) return false;
        snapshots.erase(snapshots.begin());
    }
    return true;
}

bool link_or_copy_file(const std::filesystem::path & source, const std::filesystem::path & destination) {
    std::error_code ec;
    std::filesystem::create_hard_link(source, destination, ec);
    if (!ec) return true;
    ec.clear();
    std::filesystem::copy_file(source, destination, std::filesystem::copy_options::none, ec);
    return !ec;
}

bool install_file_alias(const std::filesystem::path & source, const std::filesystem::path & destination) {
    const std::filesystem::path temporary = destination.string() + ".tmp";
    const std::filesystem::path backup = destination.string() + ".previous";
    std::error_code ec;
    std::filesystem::remove(temporary, ec);
    ec.clear();
    if (!link_or_copy_file(source, temporary)) return false;

    std::filesystem::rename(temporary, destination, ec);
    if (!ec) return true;

    ec.clear();
    if (!std::filesystem::exists(destination, ec) || ec) {
        std::filesystem::remove(temporary, ec);
        return false;
    }
    if (std::filesystem::exists(backup, ec)) {
        std::filesystem::remove(backup, ec);
        if (ec) {
            std::filesystem::remove(temporary, ec);
            return false;
        }
    }
    std::filesystem::rename(destination, backup, ec);
    if (ec) {
        std::filesystem::remove(temporary, ec);
        return false;
    }
    std::filesystem::rename(temporary, destination, ec);
    if (ec) {
        std::error_code restore_ec;
        std::filesystem::rename(backup, destination, restore_ec);
        return false;
    }
    std::filesystem::remove(backup, ec);
    return true;
}

std::filesystem::path resolve_checkpoint_directory(const std::filesystem::path & input) {
    std::error_code ec;
    const std::filesystem::path normalized = std::filesystem::absolute(input, ec).lexically_normal();
    const std::filesystem::path candidate = ec ? input : normalized;
    if (is_checkpoint_directory(candidate)) {
        return candidate;
    }
    std::vector<std::pair<int64_t, std::filesystem::path>> checkpoints;
    if (!collect_checkpoints(candidate, checkpoints) || checkpoints.empty()) return {};
    return checkpoints.back().second;
}

bool read_gguf_tensor_bytes(const std::filesystem::path & path,
                            const gguf_context * gguf,
                            const char * name,
                            ggml_type expected_type,
                            std::vector<uint8_t> & data) {
    const int64_t id = gguf_find_tensor(gguf, name);
    if (id < 0 || gguf_get_tensor_type(gguf, id) != expected_type) return false;
    const size_t size = gguf_get_tensor_size(gguf, id);
    const size_t base = gguf_get_data_offset(gguf);
    const size_t offset = gguf_get_tensor_offset(gguf, id);
    if (offset > SIZE_MAX - base || size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) return false;
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return false;
    stream.seekg(static_cast<std::streamoff>(base + offset));
    if (!stream) return false;
    data.resize(size);
    if (size > 0) stream.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(size));
    return static_cast<bool>(stream) || (size == 0 && static_cast<bool>(stream));
}

bool write_optimizer_file(const std::filesystem::path & path,
                          const std::vector<uint8_t> & optimizer_state,
                          const char * optimizer_name,
                          const std::string & optimizer_state_type) {
    if (optimizer_state.empty() || optimizer_state.size() > static_cast<size_t>(INT64_MAX)) return false;
    ggml_init_params tensor_params = { ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr tensor_ctx(ggml_init(tensor_params));
    if (!tensor_ctx) return false;
    ggml_tensor * tensor = ggml_new_tensor_1d(tensor_ctx.get(), GGML_TYPE_I8, static_cast<int64_t>(optimizer_state.size()));
    if (!tensor) return false;
    ggml_set_name(tensor, "optimizer.state");

    gguf_context_ptr gguf(gguf_init_empty());
    if (!gguf) return false;
    gguf_set_val_u32(gguf.get(), "qlora.checkpoint.version", 3);
    gguf_set_val_str(gguf.get(), "qlora.optimizer.name", optimizer_name);
    gguf_set_val_str(gguf.get(), "qlora.optimizer.state_type", optimizer_state_type.c_str());
    gguf_add_tensor(gguf.get(), tensor);
    gguf_set_tensor_data(gguf.get(), "optimizer.state", optimizer_state.data());
    return gguf_write_to_file(gguf.get(), path.string().c_str(), false);
}

bool load_qlora_checkpoint(const std::string & requested_path, qlora_checkpoint & result) {
    const std::filesystem::path requested(requested_path);
    std::error_code ec;
    const std::filesystem::path normalized = std::filesystem::absolute(requested, ec).lexically_normal();
    const std::filesystem::path candidate = ec ? requested : normalized;
    int64_t requested_step = 0;
    const bool checkpoint_path = is_checkpoint_directory(candidate) ||
        parse_step_name(candidate.filename().string(), "checkpoint-", "", requested_step);
    const std::filesystem::path recovery_root = checkpoint_path ? candidate.parent_path() : candidate;
    if (!recover_checkpoint_directories(recovery_root)) {
        LOG_ERR("%s: failed to recover checkpoint output directory '%s'\n", __func__, recovery_root.string().c_str());
        return false;
    }
    result.directory = resolve_checkpoint_directory(requested_path);
    if (result.directory.empty()) {
        LOG_ERR("%s: cannot resolve checkpoint '%s' (expected a checkpoint directory or an output directory containing checkpoints)\n",
                __func__, requested_path.c_str());
        return false;
    }

    std::string metadata_text;
    if (!read_text_file(result.directory / "training-state.json", metadata_text)) {
        LOG_ERR("%s: failed to read checkpoint metadata\n", __func__);
        return false;
    }
    try {
        result.metadata = common_json::parse(metadata_text);
        const int format_version = result.metadata.at("format_version").get<int>();
        if (format_version != 3) {
            LOG_ERR("%s: unsupported checkpoint format version\n", __func__);
            return false;
        }
        result.next_epoch = result.metadata.at("next_epoch").get<int64_t>();
        result.next_window = result.metadata.at("next_window").get<int64_t>();
        result.scheduler_step = result.metadata.at("scheduler_step").get<int64_t>();
        const common_json & permutation = result.metadata.at("dataset_permutation");
        if (!permutation.is_array()) return false;
        result.dataset_permutation.reserve(permutation.size());
        for (size_t i = 0; i < permutation.size(); ++i) {
            result.dataset_permutation.push_back(permutation.at(i).get<int64_t>());
        }
    } catch (const std::exception & ex) {
        LOG_ERR("%s: invalid checkpoint metadata: %s\n", __func__, ex.what());
        return false;
    }

    std::string manifest_text;
    if (!read_text_file(result.directory / "manifest.json", manifest_text)) {
        LOG_ERR("%s: failed to read checkpoint manifest\n", __func__);
        return false;
    }
    try {
        const common_json manifest = common_json::parse(manifest_text);
        if (manifest.at("format_version").get<int>() != 2 ||
            manifest.at("optimizer_step").get<int64_t>() != result.scheduler_step ||
            manifest.at("adapter_file").get<std::string>() != "adapter.gguf" ||
            manifest.at("optimizer_file").get<std::string>() != "optimizer.gguf" ||
            manifest.at("training_state_file").get<std::string>() != "training-state.json") {
            LOG_ERR("%s: invalid checkpoint manifest\n", __func__);
            return false;
        }
    } catch (const std::exception & ex) {
        LOG_ERR("%s: invalid checkpoint manifest: %s\n", __func__, ex.what());
        return false;
    }

    gguf_init_params init_params = { true, nullptr };
    gguf_context_ptr optimizer_gguf(gguf_init_from_file((result.directory / "optimizer.gguf").string().c_str(), init_params));
    if (!optimizer_gguf) {
        LOG_ERR("%s: failed to open optimizer checkpoint\n", __func__);
        return false;
    }
    const int64_t version_id = gguf_find_key(optimizer_gguf.get(), "qlora.checkpoint.version");
    const int64_t optimizer_name_id = gguf_find_key(optimizer_gguf.get(), "qlora.optimizer.name");
    const int64_t optimizer_state_type_id = gguf_find_key(optimizer_gguf.get(), "qlora.optimizer.state_type");
    bool optimizer_matches = false;
    bool optimizer_state_type_matches = false;
    if (optimizer_name_id >= 0) {
        const char * stored_name = gguf_get_val_str(optimizer_gguf.get(), optimizer_name_id);
        try {
            optimizer_matches = stored_name && result.metadata.at("optimizer").get<std::string>() == stored_name;
        } catch (const std::exception &) {
            optimizer_matches = false;
        }
    }
    if (optimizer_state_type_id >= 0) {
        const char * stored_type = gguf_get_val_str(optimizer_gguf.get(), optimizer_state_type_id);
        try {
            optimizer_state_type_matches = stored_type && result.metadata.at("optimizer_state_type").get<std::string>() == stored_type;
        } catch (const std::exception &) {
            optimizer_state_type_matches = false;
        }
    }
    if (version_id < 0 || optimizer_name_id < 0 || optimizer_state_type_id < 0 ||
        gguf_get_val_u32(optimizer_gguf.get(), version_id) != 3 ||
        !optimizer_matches || !optimizer_state_type_matches ||
        !read_gguf_tensor_bytes(result.directory / "optimizer.gguf", optimizer_gguf.get(),
                                "optimizer.state", GGML_TYPE_I8, result.optimizer_state)) {
        LOG_ERR("%s: invalid optimizer checkpoint contents\n", __func__);
        return false;
    }
    return true;
}

bool validate_resume_config(const qlora_checkpoint & checkpoint,
                            const qlora_params & qparams,
                            const common_params & params,
                            const std::string & arch,
                            float lora_alpha,
                            int32_t n_ctx,
                            int64_t dataset_windows) {
    const common_json & state = checkpoint.metadata;
    try {
        auto fail = [](const char * key) {
            LOG_ERR("%s: resume configuration mismatch for '%s'; use the settings from the checkpoint\n", __func__, key);
            return false;
        };
        if (state.at("base_model_path").get<std::string>() != canonical_path_string(params.model.path)) return fail("base_model_path");
        dataset_file_identity base_model_identity;
        if (!get_dataset_file_identity(params.model.path, base_model_identity)) return fail("base_model_file");
        const common_json & saved_model_file = state.at("base_model_file");
        if (saved_model_file.at("path").get<std::string>() != base_model_identity.path ||
            saved_model_file.at("size").get<int64_t>() != base_model_identity.size) return fail("base_model_file");
        if (state.at("architecture").get<std::string>() != arch) return fail("architecture");
        if (state.at("lora_rank").get<int32_t>() != qparams.lora_rank) return fail("lora_rank");
        if (std::fabs(state.at("lora_alpha").get<double>() - lora_alpha) > 1e-6) return fail("lora_alpha");
        if (state.at("lora_targets").get<std::string>() != qparams.lora_targets) return fail("lora_targets");
        if (state.at("freeze_layers").get<int32_t>() != qparams.freeze_layers) return fail("freeze_layers");
        if (state.at("gradient_accumulation_steps").get<int32_t>() != qparams.gradient_accumulation_steps) return fail("gradient_accumulation_steps");
        if (state.value("max_steps", int32_t(0)) != qparams.max_steps) return fail("max_steps");
        if (state.value("val_windows", int32_t(-1)) != qparams.val_windows) return fail("val_windows");
        if (state.at("context_length").get<int32_t>() != n_ctx) return fail("context_length");
        if (state.at("dataset_windows").get<int64_t>() != dataset_windows) return fail("dataset_windows");
        if (state.at("shuffle_dataset").get<bool>() != qparams.shuffle_dataset) return fail("shuffle_dataset");
        if (state.at("skip_invalid_records").get<bool>() != qparams.dataset.skip_invalid) return fail("skip_invalid_records");
        if (state.at("lr_scheduler").get<std::string>() != qparams.lr_scheduler) return fail("lr_scheduler");
        if (state.at("warmup_steps").get<int32_t>() != qparams.warmup_steps) return fail("warmup_steps");
        if (std::fabs(state.at("warmup_init_ratio").get<double>() - qparams.warmup_init_ratio) > 1e-7) return fail("warmup_init_ratio");
        if (state.at("learning_rate").get<double>() != params.lr.lr0) return fail("learning_rate");
        if (state.at("learning_rate_min").get<double>() != params.lr.lr_min) return fail("learning_rate_min");
        if (state.at("weight_decay").get<double>() != params.lr.wd) return fail("weight_decay");
        if (state.at("total_epochs").get<unsigned int>() != params.lr.epochs) return fail("total_epochs");
        if (state.at("optimizer").get<std::string>() != ggml_opt_optimizer_name(params.optimizer)) return fail("optimizer");
        if (state.at("lora_type").get<std::string>() != qparams.lora_type) return fail("lora_type");
        if (state.at("optimizer_state_type").get<std::string>() != qparams.optimizer_state_type) return fail("optimizer_state_type");
        const ggml_opt_optimizer_params optimizer_defaults = ggml_opt_get_default_optimizer_params(nullptr);
        if (std::fabs(state.at("adamw_beta1").get<float>() - optimizer_defaults.adamw.beta1) > 1e-7f ||
            std::fabs(state.at("adamw_beta2").get<float>() - optimizer_defaults.adamw.beta2) > 1e-7f ||
            std::fabs(state.at("adamw_eps").get<float>() - optimizer_defaults.adamw.eps) > 1e-12f) return fail("optimizer_hyperparameters");

        const common_json current_files = make_dataset_file_metadata(qparams);
        const common_json & saved_files = state.at("dataset_files");
        if (current_files.size() != saved_files.size()) return fail("dataset_files");
        for (size_t i = 0; i < current_files.size(); ++i) {
            if (current_files.at(i).at("path").get<std::string>() != saved_files.at(i).at("path").get<std::string>() ||
                current_files.at(i).at("size").get<int64_t>() != saved_files.at(i).at("size").get<int64_t>()) {
                return fail("dataset_files");
            }
        }
        if (checkpoint.next_epoch < 0 || checkpoint.next_epoch > params.lr.epochs ||
            checkpoint.next_window < 0 || (checkpoint.next_window >= dataset_windows && checkpoint.next_window != 0) ||
            checkpoint.scheduler_step < 0) {
            LOG_ERR("%s: checkpoint training position is out of range\n", __func__);
            return false;
        }
        if (checkpoint.next_epoch == params.lr.epochs && checkpoint.next_window != 0) return fail("training_position");
        if (checkpoint.next_window > 0 && checkpoint.next_window % qparams.gradient_accumulation_steps != 0) {
            LOG_ERR("%s: checkpoint is not at an accumulation boundary\n", __func__);
            return false;
        }
        const common_json & permutation = state.at("dataset_permutation");
        if (permutation.size() != static_cast<size_t>(dataset_windows)) return fail("dataset_permutation");
    } catch (const std::exception & ex) {
        LOG_ERR("%s: failed to validate resume metadata: %s\n", __func__, ex.what());
        return false;
    }
    return true;
}

bool save_qlora_checkpoint(const std::filesystem::path & root,
                           const lora_tensor_refs & output_refs,
                           const std::string & arch,
                           float lora_alpha,
                           const std::string & base_model_path,
                           const qlora_params & qparams,
                           const common_params & params,
                           llama_context * ctx,
                           ggml_opt_dataset_t dataset,
                           const qlora_lr_schedule & schedule,
                           int32_t n_ctx,
                           int64_t dataset_windows,
                           int64_t next_epoch,
                           int64_t next_window,
                           const std::filesystem::path & adapter_source,
                           int32_t limit,
                           std::filesystem::path * saved_directory = nullptr) {
    if (root.empty() || limit <= 0) return false;
    const size_t state_size = llama_opt_state_size(ctx);
    if (state_size == 0) {
        LOG_ERR("%s: optimizer state is not available at a checkpoint boundary\n", __func__);
        return false;
    }
    std::vector<uint8_t> optimizer_state(state_size);
    if (!llama_opt_state_export(ctx, optimizer_state.data(), optimizer_state.size())) {
        LOG_ERR("%s: failed to export optimizer state\n", __func__);
        return false;
    }
    const int64_t permutation_size = ggml_opt_dataset_permutation_size(dataset);
    if (permutation_size != dataset_windows) return false;
    std::vector<int64_t> permutation(static_cast<size_t>(permutation_size));
    if (!ggml_opt_dataset_get_permutation(dataset, permutation.data(), permutation_size)) return false;

    common_json state = common_json::object();
    try {
        state["format_version"] = 3;
        dataset_file_identity base_model_identity;
        if (!get_dataset_file_identity(base_model_path, base_model_identity)) {
            throw std::runtime_error("cannot inspect base model file: " + base_model_path);
        }
        common_json base_model_file = common_json::object();
        base_model_file["path"] = base_model_identity.path;
        base_model_file["size"] = base_model_identity.size;
        state["base_model_path"] = canonical_path_string(base_model_path);
        state["base_model_file"] = base_model_file;
        state["architecture"] = arch;
        state["lora_rank"] = qparams.lora_rank;
        state["lora_alpha"] = lora_alpha;
        state["lora_targets"] = qparams.lora_targets;
        state["freeze_layers"] = qparams.freeze_layers;
        state["gradient_accumulation_steps"] = qparams.gradient_accumulation_steps;
        state["max_steps"] = qparams.max_steps;
        state["val_windows"] = qparams.val_windows;
        state["context_length"] = n_ctx;
        state["dataset_windows"] = dataset_windows;
        state["dataset_files"] = make_dataset_file_metadata(qparams);
        state["shuffle_dataset"] = qparams.shuffle_dataset;
        state["skip_invalid_records"] = qparams.dataset.skip_invalid;
        state["lr_scheduler"] = qparams.lr_scheduler;
        state["warmup_steps"] = qparams.warmup_steps;
        state["warmup_init_ratio"] = qparams.warmup_init_ratio;
        state["learning_rate"] = params.lr.lr0;
        state["learning_rate_min"] = params.lr.lr_min;
        state["weight_decay"] = params.lr.wd;
        state["optimizer"] = ggml_opt_optimizer_name(params.optimizer);
        state["lora_type"] = qparams.lora_type;
        state["optimizer_state_type"] = qparams.optimizer_state_type;
        const ggml_opt_optimizer_params optimizer_defaults = ggml_opt_get_default_optimizer_params(nullptr);
        state["adamw_beta1"] = optimizer_defaults.adamw.beta1;
        state["adamw_beta2"] = optimizer_defaults.adamw.beta2;
        state["adamw_eps"] = optimizer_defaults.adamw.eps;
        state["total_epochs"] = params.lr.epochs;
        state["next_epoch"] = next_epoch;
        state["next_window"] = next_window;
        state["scheduler_step"] = schedule.step;
        common_json order = common_json::array();
        for (const int64_t id : permutation) order.push_back(id);
        state["dataset_permutation"] = order;
    } catch (const std::exception & ex) {
        LOG_ERR("%s: failed to build checkpoint metadata: %s\n", __func__, ex.what());
        return false;
    }

    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    if (ec) {
        LOG_ERR("%s: cannot create checkpoint directory '%s': %s\n", __func__, root.string().c_str(), ec.message().c_str());
        return false;
    }
    const int64_t global_step = schedule.step;
    const std::filesystem::path final_dir = root / step_name("checkpoint-", global_step);
    const std::filesystem::path backup_dir = final_dir.string() + ".previous";
    const auto temp_stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path temp_dir = root / ("." + step_name("checkpoint-", global_step) + ".tmp-" + std::to_string(temp_stamp));
    std::filesystem::create_directory(temp_dir, ec);
    if (ec) return false;

    bool adapter_ok = false;
    if (!adapter_source.empty()) {
        adapter_ok = link_or_copy_file(adapter_source, temp_dir / "adapter.gguf");
    } else {
        adapter_ok = save_adapter(output_refs, (temp_dir / "adapter.gguf").string(),
                                  arch, lora_alpha, base_model_path, false);
    }
    bool ok = adapter_ok &&
              write_optimizer_file(temp_dir / "optimizer.gguf", optimizer_state, ggml_opt_optimizer_name(params.optimizer), qparams.optimizer_state_type) &&
              write_text_file(temp_dir / "training-state.json", state.dump(2) + "\n");
    common_json manifest = common_json::object();
    manifest["format_version"] = 2;
    manifest["optimizer_file"] = "optimizer.gguf";
    manifest["adapter_file"] = "adapter.gguf";
    manifest["training_state_file"] = "training-state.json";
    manifest["optimizer_step"] = global_step;
    if (ok) ok = write_text_file(temp_dir / "manifest.json", manifest.dump(2) + "\n");

    if (ok) {
        const bool final_exists = std::filesystem::exists(final_dir, ec);
        if (ec) {
            ok = false;
        } else if (final_exists) {
            const bool backup_exists = std::filesystem::exists(backup_dir, ec);
            if (ec || backup_exists) {
                ok = false;
            } else {
                std::filesystem::rename(final_dir, backup_dir, ec);
                ok = !ec;
            }
        }
    }
    if (ok) {
        std::filesystem::rename(temp_dir, final_dir, ec);
        ok = !ec;
    }
    if (!ok) {
        std::error_code cleanup_ec;
        std::filesystem::remove_all(temp_dir, cleanup_ec);
        cleanup_ec.clear();
        const bool backup_exists = std::filesystem::exists(backup_dir, cleanup_ec);
        cleanup_ec.clear();
        const bool final_exists = std::filesystem::exists(final_dir, cleanup_ec);
        if (backup_exists && !cleanup_ec && !final_exists) {
            cleanup_ec.clear();
            std::filesystem::rename(backup_dir, final_dir, cleanup_ec);
        }
        LOG_ERR("%s: failed to write checkpoint under '%s'\n", __func__, root.string().c_str());
        return false;
    }

    ec.clear();
    if (std::filesystem::exists(backup_dir, ec) && !ec) {
        std::filesystem::remove_all(backup_dir, ec);
        if (ec) {
            LOG_WRN("%s: new checkpoint is valid but previous checkpoint could not be removed: %s\n",
                    __func__, ec.message().c_str());
        }
    }
    if (saved_directory) *saved_directory = final_dir;
    if (!prune_checkpoints(root, limit)) {
        LOG_WRN("%s: checkpoint was saved but old checkpoint pruning failed\n", __func__);
    }
    return true;
}

struct save_context {
    const lora_tensor_refs * output_tensors = nullptr;
    std::filesystem::path output_root;
    const std::string * arch = nullptr;
    const std::string * base_model_path = nullptr;
    const qlora_params * qparams = nullptr;
    const common_params * params = nullptr;
    llama_context * ctx = nullptr;
    ggml_opt_dataset_t dataset = nullptr;
    float lora_alpha = 0.0f;
    int32_t save_checkpoint_steps = 0;
    int32_t save_checkpoint_limit = 0;
    int32_t save_snapshot_steps = 0;
    int32_t save_snapshot_limit = 0;
    int32_t gradient_accumulation_steps = 1;
    int64_t dataset_windows = 0;
    std::vector<int64_t> dataset_permutation;
    int64_t epoch = 0; // one-based
    int64_t window_start = 0; // windows already consumed in this epoch
    int64_t validation_window_start = 0;
    bool validation_progress_started = false;
    int64_t last_checkpoint_step = -1;
    int64_t last_checkpoint_next_epoch = -1;
    int64_t last_checkpoint_next_window = -1;
    int64_t cached_adapter_step = -1;
    std::filesystem::path cached_adapter_path;
    qlora_lr_schedule * schedule = nullptr;
};

// The callback is intentionally an orchestrator only: progress, logical-step
// updates, and snapshot writes stay separate so each can be extended in place.
static thread_local save_context * g_save_context = nullptr;

struct save_context_scope {
    explicit save_context_scope(save_context & state) {
        g_save_context = &state;
    }

    ~save_context_scope() {
        g_save_context = nullptr;
    }
};

bool is_optimizer_update(const save_context & state, int64_t ibatch) {
    const int64_t absolute_window = state.window_start + ibatch;
    return absolute_window % state.gradient_accumulation_steps == 0 || absolute_window == state.dataset_windows;
}

bool save_checkpoint_if_due(save_context & state, int64_t ibatch) {
    if (state.save_checkpoint_limit <= 0 || state.save_checkpoint_steps <= 0 ||
        !is_optimizer_update(state, ibatch) ||
        state.schedule->step <= 0 || state.schedule->step % state.save_checkpoint_steps != 0) {
        return false;
    }

    const int64_t consumed = state.window_start + ibatch;
    const int64_t next_epoch = consumed == state.dataset_windows ? state.epoch : state.epoch - 1;
    const int64_t next_window = consumed == state.dataset_windows ? 0 : consumed;
    if (state.last_checkpoint_step == state.schedule->step &&
        state.last_checkpoint_next_epoch == next_epoch && state.last_checkpoint_next_window == next_window) {
        return false;
    }

    const std::filesystem::path adapter_source = state.cached_adapter_step == state.schedule->step
        ? state.cached_adapter_path : std::filesystem::path();
    std::filesystem::path saved_directory;
    if (save_qlora_checkpoint(state.output_root, *state.output_tensors, *state.arch, state.lora_alpha,
                              *state.base_model_path, *state.qparams, *state.params, state.ctx, state.dataset,
                              *state.schedule, llama_n_ctx(state.ctx), state.dataset_windows, next_epoch, next_window,
                              adapter_source, state.save_checkpoint_limit, &saved_directory)) {
        state.last_checkpoint_step = state.schedule->step;
        state.last_checkpoint_next_epoch = next_epoch;
        state.last_checkpoint_next_window = next_window;
        state.cached_adapter_step = state.schedule->step;
        state.cached_adapter_path = saved_directory / "adapter.gguf";
        return true;
    }
    LOG_ERR("periodic resumable checkpoint failed; training will continue\n");
    return false;
}

void update_training_step(save_context & state, int64_t ibatch) {
    if (is_optimizer_update(state, ibatch)) {
        state.schedule->advance_step();
    }
}

bool save_snapshot_if_due(save_context & state, int64_t ibatch) {
    if (state.save_snapshot_limit <= 0 || state.save_snapshot_steps <= 0 ||
        !is_optimizer_update(state, ibatch) ||
        state.schedule->step <= 0 || state.schedule->step % state.save_snapshot_steps != 0) {
        return false;
    }

    const std::filesystem::path path = state.output_root / step_name("snapshot-", state.schedule->step, ".gguf");
    if (save_adapter(*state.output_tensors, path.string(), *state.arch, state.lora_alpha,
                     *state.base_model_path, false)) {
        state.cached_adapter_step = state.schedule->step;
        state.cached_adapter_path = path;
        if (!prune_snapshots(state.output_root, state.save_snapshot_limit)) {
            LOG_WRN("snapshot was saved but old snapshot pruning failed\n");
        }
        return true;
    }
    // Snapshot failures are deliberately non-fatal: keep the training loop running.
    LOG_ERR("periodic adapter snapshot failed: %s; training will continue\n", path.string().c_str());
    return false;
}

int64_t progress_number_width(int64_t value) {
    return static_cast<int64_t>(std::to_string(std::max<int64_t>(value, 1)).size());
}

std::string format_four_significant_digits(double value) {
    if (!std::isfinite(value)) {
        std::ostringstream stream;
        stream << value;
        return stream.str();
    }
    if (value == 0.0) {
        return "0.000";
    }

    int exponent = static_cast<int>(std::floor(std::log10(std::fabs(value))));
    if (exponent >= -4 && exponent <= 3) {
        const double scale = std::pow(10.0, 3 - exponent);
        const double rounded = std::round(std::fabs(value) * scale) / scale;
        if (rounded >= std::pow(10.0, exponent + 1)) {
            ++exponent;
        }
    }

    std::ostringstream stream;
    if (exponent >= 4 || exponent < -4) {
        stream << std::scientific << std::setprecision(3) << value;
    } else {
        stream << std::fixed << std::setprecision(3 - exponent) << value;
    }
    return stream.str();
}

void split_progress_time(int64_t elapsed_us, int64_t & hours, int64_t & minutes, int64_t & seconds) {
    int64_t remaining_seconds = std::max<int64_t>(elapsed_us, 0) / 1000000;
    hours = remaining_seconds / 3600;
    remaining_seconds -= hours * 3600;
    minutes = remaining_seconds / 60;
    seconds = remaining_seconds - minutes * 60;
}

bool get_window_training_statistics(const save_context & state,
                                    int64_t window_index,
                                    int64_t & supervised,
                                    double & weight_sum) {
    const ggml_tensor * labels = ggml_opt_dataset_labels(state.dataset);
    const ggml_tensor * weights = ggml_opt_dataset_weights(state.dataset);
    if (!labels || labels->type != GGML_TYPE_I32 || labels->ne[1] != state.dataset_windows ||
        window_index < 0 || window_index >= state.dataset_windows ||
        state.dataset_permutation.size() != static_cast<size_t>(state.dataset_windows)) {
        return false;
    }
    if (weights && (weights->type != GGML_TYPE_F32 || weights->ne[0] != labels->ne[0] ||
                    weights->ne[1] != labels->ne[1])) {
        return false;
    }

    const int64_t data_index = state.dataset_permutation[static_cast<size_t>(window_index)];
    if (data_index < 0 || data_index >= labels->ne[1]) {
        return false;
    }

    const int32_t * label_data = static_cast<const int32_t *>(labels->data) + data_index * labels->ne[0];
    const float * weight_data = weights ?
        static_cast<const float *>(weights->data) + data_index * weights->ne[0] : nullptr;
    supervised = 0;
    weight_sum = 0.0;
    for (int64_t i = 0; i < labels->ne[0]; ++i) {
        if (label_data[i] < 0) {
            continue;
        }
        const float weight = weight_data ? weight_data[i] : 1.0f;
        if (!std::isfinite(weight) || weight < 0.0f || weight > 1.0f) {
            return false;
        }
        if (weight > 0.0f) {
            ++supervised;
            weight_sum += weight;
        }
    }
    return true;
}

void qlora_epoch_callback_progress_line(bool               train,
                                       ggml_opt_context_t opt_ctx,
                                       ggml_opt_dataset_t dataset,
                                       ggml_opt_result_t  result,
                                       int64_t            ibatch,
                                       int64_t            ibatch_max,
                                       int64_t            t_start_us) {
    if (!g_save_context) {
        ggml_opt_epoch_callback_progress_bar(train, opt_ctx, dataset, result, ibatch, ibatch_max, t_start_us);
        return;
    }

    save_context & state = *g_save_context;
    if (!train && !state.validation_progress_started) {
        fprintf(stderr, "\n");
        state.validation_progress_started = true;
    }

    const int64_t window_start = train ? state.window_start : state.validation_window_start;
    const int64_t window_index = window_start + ibatch - 1;
    const int64_t current_window = window_index + 1;
    int64_t supervised = 0;
    double weight_sum = 0.0;
    if (!get_window_training_statistics(state, window_index, supervised, weight_sum)) {
        LOG_ERR("%s: failed to read statistics for window %ld\n", __func__, static_cast<long>(current_window));
        return;
    }

    double loss = 0.0;
    double average_loss = 0.0;
    ggml_opt_result_last_loss(result, &loss);
    ggml_opt_result_loss(result, &average_loss, nullptr);
    const std::string loss_text = format_four_significant_digits(loss);
    const std::string average_loss_text = format_four_significant_digits(average_loss);
    const std::string weight_sum_text = format_four_significant_digits(weight_sum);

    const int64_t elapsed_us = std::max<int64_t>(0, ggml_time_us() - t_start_us);
    const int64_t eta_us = ibatch > 0 ? static_cast<int64_t>(
        static_cast<double>(elapsed_us) * static_cast<double>(ibatch_max - ibatch) / static_cast<double>(ibatch)) : 0;
    int64_t elapsed_h = 0;
    int64_t elapsed_m = 0;
    int64_t elapsed_s = 0;
    int64_t eta_h = 0;
    int64_t eta_m = 0;
    int64_t eta_s = 0;
    split_progress_time(elapsed_us, elapsed_h, elapsed_m, elapsed_s);
    split_progress_time(eta_us, eta_h, eta_m, eta_s);

    const int64_t window_width = progress_number_width(state.dataset_windows);
    if (train) {
        const int64_t configured_period = state.gradient_accumulation_steps;
        const int64_t accumulation_group_start = (window_index / configured_period) * configured_period;
        const int64_t accumulation_period = std::min<int64_t>(
            configured_period, state.dataset_windows - accumulation_group_start);
        const int64_t accumulation_step = window_index - accumulation_group_start + 1;
        const int64_t accumulation_width = progress_number_width(accumulation_period);

        fprintf(stderr, "[%0*" PRId64 "/%0*" PRId64 " (%0*" PRId64 "/%0*" PRId64 ")] "
                "loss=%s avg=%s supervised=%" PRId64 " weight_sum=%s "
                "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
                static_cast<int>(window_width), current_window,
                static_cast<int>(window_width), state.dataset_windows,
                static_cast<int>(accumulation_width), accumulation_step,
                static_cast<int>(accumulation_width), accumulation_period,
                loss_text.c_str(), average_loss_text.c_str(), supervised, weight_sum_text.c_str(),
                elapsed_h, elapsed_m, elapsed_s, eta_h, eta_m, eta_s);
    } else {
        fprintf(stderr, "[val %0*" PRId64 "/%0*" PRId64 "] "
                "loss=%s avg=%s supervised=%" PRId64 " weight_sum=%s "
                "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
                static_cast<int>(window_width), current_window,
                static_cast<int>(window_width), state.dataset_windows,
                loss_text.c_str(), average_loss_text.c_str(), supervised, weight_sum_text.c_str(),
                elapsed_h, elapsed_m, elapsed_s, eta_h, eta_m, eta_s);
    }
    fflush(stderr);
}

void save_step_callback(bool               train,
                        ggml_opt_context_t opt_ctx,
                        ggml_opt_dataset_t dataset,
                        ggml_opt_result_t  result,
                        int64_t            ibatch,
                        int64_t            ibatch_max,
                        int64_t            t_start_us) {
    qlora_epoch_callback_progress_line(train, opt_ctx, dataset, result, ibatch, ibatch_max, t_start_us);

    if (!train || !g_save_context) {
        return;
    }
    update_training_step(*g_save_context, ibatch);
    const bool snapshot_saved = save_snapshot_if_due(*g_save_context, ibatch);
    const bool checkpoint_saved = save_checkpoint_if_due(*g_save_context, ibatch);
    if (snapshot_saved && checkpoint_saved) {
        LOG_INF("saved checkpoint and snapshot (optimizer step %ld)\n",
                static_cast<long>(g_save_context->schedule->step));
    } else if (snapshot_saved) {
        LOG_INF("saved snapshot (optimizer step %ld)\n", static_cast<long>(g_save_context->schedule->step));
    } else if (checkpoint_saved) {
        LOG_INF("saved checkpoint (optimizer step %ld)\n", static_cast<long>(g_save_context->schedule->step));
    }
}

void log_epoch_loss(int epoch, int total_epochs, ggml_opt_result_t train_result) {
    double train_loss = 0.0;
    ggml_opt_result_loss(train_result, &train_loss, nullptr);
    const std::string train_loss_text = format_four_significant_digits(train_loss);
    LOG_INF("epoch %d/%d: train_loss=%s\n", epoch, total_epochs, train_loss_text.c_str());
}

ggml_opt_dataset_ptr create_training_dataset(const qlora_params & qparams,
                                             const llama_model * model,
                                             const llama_vocab * vocab,
                                             int32_t context_length) {
    common_chat_templates_ptr templates;
    const char * model_template = llama_model_chat_template(model, nullptr);
    if (model_template && model_template[0] != '\0') {
        templates = common_chat_templates_init(model, "");
    } else {
        LOG_WRN("%s: model has no chat template; chat records cannot be rendered\n", __func__);
    }

    dataset_load_result loaded = load_training_samples(qparams.dataset, vocab, templates.get());
    if (!loaded.success || loaded.samples.empty()) {
        LOG_ERR("%s: no training samples loaded\n", __func__);
        return {};
    }

    const llama_token bos = llama_vocab_bos(vocab);
    ggml_opt_dataset_ptr dataset(build_training_dataset(
        loaded.samples, window_options{context_length, bos, -1}));
    if (!dataset) {
        return {};
    }
    return dataset;
}

bool train_adapter(const qlora_params & qparams,
                   common_params & params,
                   llama_model * model,
                   llama_context * ctx,
                   const lora_tensor_refs & output_refs,
                   const std::string & arch,
                   float lora_alpha,
                   qlora_checkpoint * resume) {
    const int32_t n_ctx = llama_n_ctx(ctx);
    const int32_t n_batch = llama_n_batch(ctx);
    const int32_t n_ubatch = llama_n_ubatch(ctx);
    if (n_ubatch != n_ctx || n_batch < n_ctx) {
        LOG_ERR("%s: QLoRA requires one full-context microbatch per training window: "
                "set -ub equal to -c and -b to at least -c (n_batch=%d, n_ubatch=%d, n_ctx=%d)\n",
                __func__, n_batch, n_ubatch, n_ctx);
        return false;
    }

    if (params.lr.lr0 <= 0.0f || (qparams.lr_scheduler == "cosine" && params.lr.lr_min > params.lr.lr0)) {
        LOG_ERR("%s: learning-rate must be positive and cosine requires learning-rate-min <= learning-rate\n", __func__);
        return false;
    }

    ggml_opt_dataset_ptr dataset = create_training_dataset(qparams, model, llama_model_get_vocab(model), n_ctx);
    if (!dataset) {
        return false;
    }

    qlora_lr_schedule schedule{ &params.lr, qparams.lr_scheduler, qparams.warmup_steps,
                                qparams.warmup_init_ratio, 0, 0 };
    llama_opt_params opt_params{
        0,
        lora_param_filter,
        nullptr,
        qlora_opt_lr_pars,
        &schedule,
        params.optimizer,
        qlora_type_from_name(qparams.optimizer_state_type),
    };
    const int64_t dataset_windows = ggml_opt_dataset_ndata(dataset.get());
    if (dataset_windows <= 0) {
        LOG_ERR("%s: dataset has no training windows\n", __func__);
        return false;
    }

    const int64_t accumulation_steps = qparams.gradient_accumulation_steps;
    const int64_t optimizer_steps_per_epoch = dataset_windows / accumulation_steps +
        (dataset_windows % accumulation_steps != 0);
    if (qparams.max_steps > 0) {
        params.lr.epochs = 1 + (qparams.max_steps - 1) / optimizer_steps_per_epoch;
    }
    if (qparams.max_steps == 0 && params.lr.epochs > 0 && optimizer_steps_per_epoch > INT64_MAX / params.lr.epochs) {
        LOG_ERR("%s: total optimizer step count overflows int64\n", __func__);
        return false;
    }
    schedule.total_steps = qparams.max_steps > 0 ? qparams.max_steps : optimizer_steps_per_epoch * params.lr.epochs;
    if (schedule.total_steps <= 0) {
        LOG_ERR("%s: total training step count must be positive\n", __func__);
        return false;
    }

    if (resume) {
        if (!validate_resume_config(*resume, qparams, params, arch, lora_alpha, n_ctx, dataset_windows)) {
            return false;
        }
        if (!ggml_opt_dataset_set_permutation(dataset.get(), resume->dataset_permutation.data(),
                                               static_cast<int64_t>(resume->dataset_permutation.size()))) {
            LOG_ERR("%s: invalid dataset permutation in checkpoint\n", __func__);
            return false;
        }
    }
    if (qparams.warmup_steps > schedule.total_steps) {
        LOG_ERR("%s: --warmup-steps %d exceeds total training steps %ld\n", __func__, qparams.warmup_steps,
                static_cast<long>(schedule.total_steps));
        return false;
    }
    if (resume) {
        if (resume->next_epoch >= params.lr.epochs) {
            LOG_ERR("%s: checkpoint already completed the requested %d epochs\n", __func__, params.lr.epochs);
            return false;
        }
        if (resume->scheduler_step > schedule.total_steps) {
            LOG_ERR("%s: checkpoint optimizer step exceeds the configured schedule\n", __func__);
            return false;
        }
        if (qparams.max_steps > 0 && resume->scheduler_step >= schedule.total_steps) {
            LOG_ERR("%s: checkpoint has already reached --max-steps=%d\n", __func__, qparams.max_steps);
            return false;
        }
        schedule.step = resume->scheduler_step;
    }
    llama_opt_init(ctx, model, opt_params);
    llama_opt_set_opt_period(ctx, qparams.gradient_accumulation_steps);
    if (resume) {
        if (!llama_opt_state_import(ctx, resume->optimizer_state.data(), resume->optimizer_state.size())) {
            LOG_ERR("%s: failed to restore optimizer state from checkpoint\n", __func__);
            return false;
        }
        LOG_INF("%s: restored optimizer state from checkpoint '%s' (optimizer step %ld)\n",
                __func__, resume->directory.string().c_str(), static_cast<long>(resume->scheduler_step));
    }

    ggml_opt_result_ptr result_train(ggml_opt_result_init());
    ggml_opt_result_ptr result_eval(ggml_opt_result_init());
    if (!result_train || !result_eval) {
        LOG_ERR("%s: failed to allocate optimizer results\n", __func__);
        return false;
    }
    save_context save_state;
    save_state.output_tensors = &output_refs;
    save_state.output_root = qparams.output_dir;
    save_state.arch = &arch;
    save_state.base_model_path = &params.model.path;
    save_state.qparams = &qparams;
    save_state.params = &params;
    save_state.ctx = ctx;
    save_state.dataset = dataset.get();
    save_state.lora_alpha = lora_alpha;
    save_state.save_checkpoint_steps = qparams.save_checkpoint_steps;
    save_state.save_checkpoint_limit = qparams.save_checkpoint_limit;
    save_state.save_snapshot_steps = qparams.save_snapshot_steps;
    save_state.save_snapshot_limit = qparams.save_snapshot_limit;
    save_state.gradient_accumulation_steps = qparams.gradient_accumulation_steps;
    save_state.dataset_windows = dataset_windows;
    save_state.schedule = &schedule;
    save_context_scope callback_scope(save_state);

    LOG_INF("%s: starting QLoRA training — rank=%d alpha=%.1f epochs=%d loss=assistant-response-only\n", __func__,
            qparams.lora_rank, lora_alpha, params.lr.epochs);
    LOG_INF("%s: dataset: %ld full-context windows; accumulate %d windows/update = %ld optimizer steps/epoch "
            "(n_ctx=%d n_ubatch=%d stride=%d)\n",
            __func__, static_cast<long>(dataset_windows), qparams.gradient_accumulation_steps,
            static_cast<long>(optimizer_steps_per_epoch), n_ctx, n_ubatch, n_ctx / 2);
    if (qparams.max_steps > 0) {
        LOG_INF("%s: --max-steps=%d overrides --epochs; training will stop after %d optimizer updates\n",
                __func__, qparams.max_steps, qparams.max_steps);
    }
    if (qparams.val_windows >= 0) {
        LOG_INF("%s: validation window limit=%d (0 disables validation)\n", __func__, qparams.val_windows);
    }
    LOG_INF("%s: lr scheduler=%s warmup_steps=%d warmup_init_ratio=%.3g total_steps=%ld lr=%.3g lr_min=%.3g\n",
            __func__, qparams.lr_scheduler.c_str(), qparams.warmup_steps, static_cast<double>(qparams.warmup_init_ratio),
            static_cast<long>(schedule.total_steps), static_cast<double>(params.lr.lr0),
            static_cast<double>(std::max(0.0f, params.lr.lr_min)));
    if (qparams.save_snapshot_limit > 0 && qparams.save_snapshot_steps > 0) {
        LOG_INF("%s: saving adapter snapshots every %d optimizer steps (limit %d)\n",
                __func__, qparams.save_snapshot_steps, qparams.save_snapshot_limit);
    }
    if (qparams.save_checkpoint_limit > 0) {
        if (qparams.save_checkpoint_steps > 0) {
            LOG_INF("%s: saving resumable checkpoints every %d optimizer steps (limit %d)\n",
                    __func__, qparams.save_checkpoint_steps, qparams.save_checkpoint_limit);
        } else {
            LOG_INF("%s: saving only the final resumable checkpoint\n", __func__);
        }
    }

    const ggml_opt_epoch_callback train_callback = save_step_callback;
    const int64_t first_epoch = resume ? resume->next_epoch : 0;
    for (params.lr.epoch = first_epoch; params.lr.epoch < params.lr.epochs; ++params.lr.epoch) {
        const int64_t window_start = resume && params.lr.epoch == resume->next_epoch ? resume->next_window : 0;
        save_state.epoch = params.lr.epoch + 1;
        save_state.window_start = window_start;
        if (qparams.shuffle_dataset && dataset_windows > 1 && window_start == 0) {
            llama_opt_dataset_shuffle(ctx, dataset.get(), dataset_windows);
        }
        save_state.dataset_permutation.resize(static_cast<size_t>(dataset_windows));
        if (!ggml_opt_dataset_get_permutation(dataset.get(), save_state.dataset_permutation.data(), dataset_windows)) {
            LOG_ERR("%s: failed to read the dataset permutation for progress reporting\n", __func__);
            return false;
        }
        int64_t window_end = dataset_windows;
        if (qparams.max_steps > 0) {
            const int64_t remaining_steps = qparams.max_steps - schedule.step;
            const int64_t remaining_windows = dataset_windows - window_start;
            const int64_t remaining_epoch_steps = remaining_windows / accumulation_steps +
                (remaining_windows % accumulation_steps != 0);
            if (remaining_steps < remaining_epoch_steps) {
                window_end = window_start + remaining_steps * accumulation_steps;
            }
        }
        const int64_t eval_end = qparams.val_windows < 0 ? dataset_windows :
            window_end + std::min<int64_t>(qparams.val_windows, dataset_windows - window_end);
        save_state.validation_window_start = window_end;
        save_state.validation_progress_started = false;
        llama_opt_epoch_range_with_eval_end(ctx, dataset.get(), result_train.get(), result_eval.get(),
                                            window_start, window_end, eval_end,
                                            train_callback, qlora_epoch_callback_progress_line);
        std::fprintf(stderr, "\n");
        log_epoch_loss(params.lr.epoch + 1, params.lr.epochs, result_train.get());
        ggml_opt_result_reset(result_train.get());
        ggml_opt_result_reset(result_eval.get());
        resume = nullptr;
        if (qparams.max_steps > 0 && schedule.step >= qparams.max_steps) {
            break;
        }
    }

    const int64_t final_next_epoch = params.lr.epochs;
    const int64_t final_next_window = 0;
    if (save_state.save_checkpoint_limit > 0) {
        if (save_state.last_checkpoint_step != schedule.step ||
            save_state.last_checkpoint_next_epoch != final_next_epoch ||
            save_state.last_checkpoint_next_window != final_next_window) {
            const std::filesystem::path adapter_source = save_state.cached_adapter_step == schedule.step
                ? save_state.cached_adapter_path : std::filesystem::path();
            std::filesystem::path saved_directory;
            if (!save_qlora_checkpoint(save_state.output_root, output_refs, arch, lora_alpha, params.model.path,
                                       qparams, params, ctx, dataset.get(), schedule, n_ctx, dataset_windows,
                                       final_next_epoch, final_next_window, adapter_source,
                                       save_state.save_checkpoint_limit, &saved_directory)) {
                LOG_ERR("%s: final resumable checkpoint could not be saved\n", __func__);
                return false;
            }
            save_state.last_checkpoint_step = schedule.step;
            save_state.last_checkpoint_next_epoch = final_next_epoch;
            save_state.last_checkpoint_next_window = final_next_window;
            save_state.cached_adapter_step = schedule.step;
            save_state.cached_adapter_path = saved_directory / "adapter.gguf";
        }

        const std::filesystem::path final_checkpoint = save_state.output_root / step_name("checkpoint-", schedule.step);
        if (!is_checkpoint_directory(final_checkpoint) ||
            !install_file_alias(final_checkpoint / "adapter.gguf", save_state.output_root / "adapter.gguf")) {
            LOG_ERR("%s: final adapter could not be installed in '%s'\n", __func__,
                    (save_state.output_root / "adapter.gguf").string().c_str());
            return false;
        }
    } else {
        const std::filesystem::path adapter_source = save_state.cached_adapter_step == schedule.step
            ? save_state.cached_adapter_path : std::filesystem::path();
        const std::filesystem::path final_adapter = save_state.output_root / "adapter.gguf";
        const bool saved = adapter_source.empty()
            ? save_adapter(output_refs, final_adapter.string(), arch, lora_alpha,
                           params.model.path, false)
            : install_file_alias(adapter_source, final_adapter);
        if (!saved) {
            LOG_ERR("%s: final adapter could not be saved to '%s'\n", __func__, final_adapter.string().c_str());
            return false;
        }
    }
    LOG_INF("final adapter saved (optimizer step %ld)\n", static_cast<long>(schedule.step));
    return true;
}

}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    qlora_params qparams;
    common_params params;
    params.cache_type_k = GGML_TYPE_F32;
    params.cache_type_v = GGML_TYPE_F32;
    {
        std::vector<std::string> common_args_storage;
        std::string parse_error;
        if (!qlora_parse_args(argc, argv, qparams, common_args_storage, parse_error)) {
            LOG_ERR("%s: %s\n", __func__, parse_error.c_str());
            qlora_print_usage(argc, argv);
            return 1;
        }

        std::vector<char *> common_argv;
        common_argv.reserve(common_args_storage.size());
        for (std::string & arg : common_args_storage) {
            common_argv.push_back(arg.data());
        }

        params.escape = false;
        if (!common_params_parse(static_cast<int>(common_argv.size()), common_argv.data(), params,
                                 LLAMA_EXAMPLE_FINETUNE, qlora_print_usage)) {
            return 1;
        }
    }

    qparams.dataset.threads = params.cpuparams.n_threads > 0
        ? static_cast<size_t>(params.cpuparams.n_threads)
        : 0;

    if (params.lora_adapters.size() > 1) {
        LOG_ERR("%s: QLoRA accepts only one input adapter; comma-separated and repeated --lora inputs are unsupported\n", __func__);
        return 1;
    }

    std::error_code output_ec;
    std::filesystem::create_directories(qparams.output_dir, output_ec);
    if (output_ec || !std::filesystem::is_directory(qparams.output_dir, output_ec) || output_ec) {
        LOG_ERR("%s: cannot create output directory '%s'\n", __func__, qparams.output_dir.c_str());
        return 1;
    }
    if (!recover_checkpoint_directories(qparams.output_dir)) {
        LOG_ERR("%s: failed to recover checkpoint output directory '%s'\n", __func__, qparams.output_dir.c_str());
        return 1;
    }

    const bool input_adapter_cli = !params.lora_adapters.empty();
    if (!qparams.resume_from.empty() && input_adapter_cli) {
        LOG_ERR("%s: --resume-from cannot be combined with --lora; the checkpoint supplies its adapter\n", __func__);
        return 1;
    }
    qlora_checkpoint resume_checkpoint;
    qlora_checkpoint * resume = nullptr;
    if (!qparams.resume_from.empty()) {
        if (!load_qlora_checkpoint(qparams.resume_from, resume_checkpoint)) {
            return 1;
        }
        resume = &resume_checkpoint;
    }
    const bool adapter_supplied = input_adapter_cli || resume != nullptr;
    const std::string input_adapter_path = resume
        ? (resume_checkpoint.directory / "adapter.gguf").string()
        : (input_adapter_cli ? params.lora_adapters.front().path : std::string());
    const float input_adapter_scale = input_adapter_cli ? params.lora_adapters.front().scale : 1.0f;
    params.lora_adapters.clear();

    if (qparams.dataset.input_paths.empty()) {
        LOG_ERR("%s: --train-file is required\n", __func__);
        return 1;
    }
    if (params.optimizer != GGML_OPT_OPTIMIZER_TYPE_ADAMW && qparams.optimizer_state_type != "f32") {
        LOG_ERR("%s: --optimizer-state-type applies only to AdamW; use f32 with SGD\n", __func__);
        return 1;
    }
    if (params.optimizer != GGML_OPT_OPTIMIZER_TYPE_ADAMW && qparams.lora_type != "f32") {
        LOG_ERR("%s: non-F32 LoRA storage currently requires AdamW; use --optimizer adamw\n", __func__);
        return 1;
    }

    // Settings required for training.
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.warmup = false;
    params.no_alloc_kv_cache = true;

    const float lora_alpha = qparams.lora_alpha > 0.0f ? qparams.lora_alpha : static_cast<float>(qparams.lora_rank);

    // Step 1: read architecture metadata before loading the model.
    std::string arch;
    {
        ggml_context * meta_ctx_raw = nullptr;
        ggml_context_ptr meta_ctx;
        gguf_init_params gguf_params = { true, &meta_ctx_raw };
        gguf_context_ptr gguf(gguf_init_from_file(params.model.path.c_str(), gguf_params));
        meta_ctx.reset(meta_ctx_raw);
        if (!gguf) {
            LOG_ERR("%s: failed to open model GGUF: %s\n", __func__, params.model.path.c_str());
            return 1;
        }
        const int key_id = gguf_find_key(gguf.get(), "general.architecture");
        if (key_id >= 0) {
            arch = gguf_get_val_str(gguf.get(), key_id);
        }
    }

    if (adapter_supplied) {
        if (!std::isfinite(input_adapter_scale)) {
            LOG_ERR("%s: input adapter scale must be finite\n", __func__);
            return 1;
        }
        if (!arch.empty() && !validate_lora_adapter_rank(input_adapter_path, arch, qparams.lora_rank)) {
            return 1;
        }
    }

    // Load the base model without an adapter. Both new and existing adapters are
    // prepared as a temporary GGUF and then loaded through the upstream loader.
    lora_tensor_refs output_refs;
    lora_tensor_refs training_refs;

    if (resume) {
        LOG_INF("%s: preparing LoRA adapter from checkpoint: %s\n",
                __func__, input_adapter_path.c_str());
    } else if (input_adapter_cli) {
        LOG_INF("%s: preparing existing LoRA adapter: %s (optimizer state starts fresh)\n",
                __func__, input_adapter_path.c_str());
    }

    common_init();
    llama_backend_init();
    llama_backend_scope backend_scope;
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("%s: failed to load model/context\n", __func__);
        return 1;
    }

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    if (arch.empty()) {
        char description[256] = {};
        llama_model_desc(model, description, sizeof(description));
        arch = std::string(description);
        arch = arch.substr(0, arch.find_first_of(" /"));
    }

    {
        remove_file_scope temporary_adapter_file;
        temporary_adapter_file.path = unique_init_adapter_path(qparams.output_dir);
        if (temporary_adapter_file.path.empty()) {
            return 1;
        }
        const std::vector<std::string> targets = string_split(qparams.lora_targets, ",");
        const enum ggml_type lora_type = qlora_type_from_name(qparams.lora_type);
        std::mt19937 rng(42);
        lora_tensors prepared;
        if (adapter_supplied) {
            prepared = transform_lora_tensors(
                input_adapter_path, arch, model->tensors_by_name, targets, qparams.lora_rank,
                lora_alpha, input_adapter_scale, rng, llama_model_n_layer(model), qparams.freeze_layers, lora_type);
        } else {
            prepared = alloc_lora_tensors(
                model->tensors_by_name, targets, qparams.lora_rank, rng,
                llama_model_n_layer(model), qparams.freeze_layers, lora_type);
        }
        if (prepared.refs.empty()) {
            return 1;
        }
        if (!save_adapter(prepared.refs, temporary_adapter_file.path, arch, lora_alpha, params.model.path)) {
            LOG_ERR("%s: cannot write temporary adapter; aborting before training\n", __func__);
            return 1;
        }

        llama_adapter_lora_ptr initial_lora(
            llama_adapter_lora_init(model, temporary_adapter_file.path.c_str()));
        if (!initial_lora) {
            LOG_ERR("%s: failed to load prepared adapter '%s' into the loaded model\n",
                    __func__, temporary_adapter_file.path.c_str());
            return 1;
        }

        llama_init->lora().emplace_back(std::move(initial_lora));
        common_adapter_lora_info adapter_info = {};
        adapter_info.path = temporary_adapter_file.path;
        adapter_info.scale = 1.0f;
        adapter_info.ptr = llama_init->lora().back().get();
        params.lora_adapters.push_back(adapter_info);
        common_set_adapter_lora(ctx, params.lora_adapters);

        llama_adapter_lora * loaded = adapter_info.ptr;
        if (!loaded) {
            LOG_ERR("%s: prepared adapter was not loaded into the model\n", __func__);
            return 1;
        }
        for (auto & kv : loaded->ab_map) {
            output_refs[kv.first] = { kv.second.a, kv.second.b };
            if (lora_tensor_matches_targets(
                    kv.first, targets, qparams.freeze_layers, llama_model_n_layer(model))) {
                ggml_set_param(kv.second.a);
                ggml_set_param(kv.second.b);
                training_refs[kv.first] = { kv.second.a, kv.second.b };
            }
        }
        if (training_refs.empty()) {
            LOG_ERR("%s: no adapter tensors matched --lora-targets; nothing can be trained\n", __func__);
            return 1;
        }
        LOG_INF("%s: adapter has %zu tensor pairs; training %zu and keeping %zu frozen\n",
                __func__, output_refs.size(), training_refs.size(), output_refs.size() - training_refs.size());
        temporary_adapter_file.remove_now();
    }

    if (!train_adapter(qparams, params, model, ctx, output_refs, arch, lora_alpha, resume)) {
        return 1;
    }
    return 0;
}
