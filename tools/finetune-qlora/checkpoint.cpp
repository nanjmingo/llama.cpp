#include "checkpoint.h"

#include "adapter.h"
#include "common.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-opt.h"
#include "gguf.h"
#include "log.h"
#include "training-internal.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

static std::string step_name(const char * prefix, int64_t step, const char * suffix = "") {
    std::ostringstream name;
    name << prefix << std::setw(6) << std::setfill('0') << step << suffix;
    return name.str();
}

static bool parse_step_name(const std::string & name,
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

std::string qlora_canonical_path_string(const std::filesystem::path & path) {
    std::error_code ec;
    std::filesystem::path canonical = std::filesystem::weakly_canonical(path, ec);
    if (ec) {
        ec.clear();
        canonical = std::filesystem::absolute(path, ec);
        if (ec) canonical = path;
    }
    return canonical.lexically_normal().string();
}

bool qlora_get_file_identity(const std::string & path, qlora_file_identity & result) {
    std::error_code ec;
    const std::filesystem::path file_path(path);
    const uintmax_t size = std::filesystem::file_size(file_path, ec);
    if (ec || size > static_cast<uintmax_t>(INT64_MAX)) return false;
    result.path = qlora_canonical_path_string(file_path);
    result.size = static_cast<int64_t>(size);
    return true;
}

common_json qlora_make_dataset_file_metadata(const qlora_params & qparams) {
    common_json result = common_json::array();
    for (const std::string & path : qparams.dataset.input_paths) {
        qlora_file_identity identity;
        if (!qlora_get_file_identity(path, identity)) {
            throw std::runtime_error("cannot inspect training dataset file: " + path);
        }
        common_json item = common_json::object();
        item["path"] = identity.path;
        item["size"] = identity.size;
        result.push_back(item);
    }
    return result;
}

static bool write_text_file(const std::filesystem::path & path, const std::string & contents) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) return false;
    stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    stream.flush();
    const bool write_ok = static_cast<bool>(stream);
    stream.close();
    return write_ok && !stream.fail();
}

static bool read_text_file(const std::filesystem::path & path, std::string & contents) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return false;
    contents.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    return stream.eof() || static_cast<bool>(stream);
}

bool is_checkpoint_directory(const std::filesystem::path & directory, int64_t * optimizer_step) {
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
            (state.at("format_version").get<int>() != 3 &&
             state.at("format_version").get<int>() != 4)) {
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

static bool collect_checkpoints(const std::filesystem::path & root,
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

static bool prune_checkpoints(const std::filesystem::path & root, int32_t limit) {
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

static bool prune_snapshots(const std::filesystem::path & root, int32_t limit) {
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

static bool link_or_copy_file(const std::filesystem::path & source, const std::filesystem::path & destination) {
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

static std::filesystem::path resolve_checkpoint_directory(const std::filesystem::path & input) {
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

static bool read_gguf_tensor_bytes(const std::filesystem::path & path,
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

static bool write_optimizer_file(const std::filesystem::path & path,
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
        if (format_version != 3 && format_version != 4) {
            LOG_ERR("%s: unsupported checkpoint format version\n", __func__);
            return false;
        }
        result.format_version = format_version;
        result.next_epoch = result.metadata.at("next_epoch").get<int64_t>();
        result.next_window = result.metadata.at("next_window").get<int64_t>();
        result.scheduler_step = result.metadata.at("scheduler_step").get<int64_t>();
        if (format_version >= 4) {
            result.training_statistics_complete = result.metadata.at("training_statistics_complete").get<bool>();
            result.training_loss_sum = result.metadata.at("training_loss_sum").get<double>();
            result.training_data_count = result.metadata.at("training_data_count").get<int64_t>();
            result.training_weight_sum = result.metadata.at("training_weight_sum").get<double>();
        } else {
            result.training_statistics_complete = result.next_window == 0;
        }
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
    if (!ggml_opt_state_data_is_finite(result.optimizer_state.data(), result.optimizer_state.size())) {
        LOG_ERR("%s: optimizer checkpoint contains non-finite values or is invalid\n", __func__);
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
            LOG_ERR("%s: resume configuration mismatch for '%s'; this setting must match the checkpoint\n", __func__, key);
            return false;
        };
        if (state.at("base_model_path").get<std::string>() != qlora_canonical_path_string(params.model.path)) return fail("base_model_path");
        qlora_file_identity base_model_identity;
        if (!qlora_get_file_identity(params.model.path, base_model_identity)) return fail("base_model_file");
        const common_json & saved_model_file = state.at("base_model_file");
        if (saved_model_file.at("path").get<std::string>() != base_model_identity.path ||
            saved_model_file.at("size").get<int64_t>() != base_model_identity.size) return fail("base_model_file");
        if (state.at("architecture").get<std::string>() != arch) return fail("architecture");
        if (state.at("lora_rank").get<int32_t>() != qparams.lora_rank) return fail("lora_rank");
        if (std::fabs(state.at("lora_alpha").get<double>() - lora_alpha) > 1e-6) return fail("lora_alpha");
        if (state.at("lora_targets").get<std::string>() != qparams.lora_targets) return fail("lora_targets");
        if (state.at("freeze_layers").get<int32_t>() != qparams.freeze_layers) return fail("freeze_layers");
        if (state.at("gradient_accumulation_steps").get<int32_t>() != qparams.gradient_accumulation_steps) return fail("gradient_accumulation_steps");
        if (state.at("context_length").get<int32_t>() != n_ctx) return fail("context_length");
        if (state.at("dataset_windows").get<int64_t>() != dataset_windows) return fail("dataset_windows");
        if (state.at("skip_invalid_records").get<bool>() != qparams.dataset.skip_invalid) return fail("skip_invalid_records");
        if (state.at("optimizer").get<std::string>() != ggml_opt_optimizer_name(params.optimizer)) return fail("optimizer");
        if (state.at("optimizer_state_type").get<std::string>() != qparams.optimizer_state_type) return fail("optimizer_state_type");
        const ggml_opt_optimizer_params optimizer_defaults = ggml_opt_get_default_optimizer_params(nullptr);
        if (std::fabs(state.at("adamw_beta1").get<float>() - optimizer_defaults.adamw.beta1) > 1e-7f ||
            std::fabs(state.at("adamw_beta2").get<float>() - optimizer_defaults.adamw.beta2) > 1e-7f ||
            std::fabs(state.at("adamw_eps").get<float>() - optimizer_defaults.adamw.eps) > 1e-12f) return fail("optimizer_hyperparameters");

        const common_json current_files = qlora_make_dataset_file_metadata(qparams);
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
        if (checkpoint.format_version >= 4) {
            if (!std::isfinite(checkpoint.training_loss_sum) || checkpoint.training_loss_sum < 0.0 ||
                checkpoint.training_data_count < 0 || !std::isfinite(checkpoint.training_weight_sum) ||
                checkpoint.training_weight_sum < 0.0) {
                return fail("training_statistics");
            }
            if (checkpoint.training_statistics_complete && checkpoint.next_window > 0 &&
                (n_ctx <= 0 || checkpoint.next_window > INT64_MAX / n_ctx ||
                 checkpoint.training_data_count != checkpoint.next_window * n_ctx)) {
                return fail("training_statistics");
            }
        }
        const common_json & permutation = state.at("dataset_permutation");
        if (permutation.size() != static_cast<size_t>(dataset_windows)) return fail("dataset_permutation");
    } catch (const std::exception & ex) {
        LOG_ERR("%s: failed to validate resume metadata: %s\n", __func__, ex.what());
        return false;
    }
    return true;
}

std::string qlora_serialize_loss_log(const std::vector<qlora_loss_log_entry> & history) {
    std::ostringstream stream;
    stream << "phase\tepoch\twindow\tloss\taverage_loss\n";
    stream << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const qlora_loss_log_entry & entry : history) {
        stream << (entry.training ? "train" : "validation") << '\t'
               << entry.epoch << '\t' << entry.window << '\t'
               << entry.loss << '\t' << entry.average_loss << '\n';
    }
    return stream.str();
}

std::string qlora_serialize_grad_stats_log(const std::vector<qlora_grad_stats_log_entry> & history) {
    std::ostringstream stream;
    stream << "step\ttensors\telements\tscale\tnorm\trms\tmax_abs\tzero_percent\tnonfinite\n";
    stream << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const qlora_grad_stats_log_entry & entry : history) {
        const ggml_opt_grad_stats & stats = entry.stats;
        const double zero_percent = stats.n_elements > 0
            ? 100.0 * static_cast<double>(stats.n_zeros) / static_cast<double>(stats.n_elements) : 0.0;
        stream << entry.step << '\t' << stats.n_tensors << '\t' << stats.n_elements << '\t'
               << stats.scale << '\t' << stats.norm << '\t' << stats.rms << '\t'
               << stats.max_abs << '\t' << zero_percent << '\t' << stats.n_nonfinite << '\n';
    }
    return stream.str();
}

bool initialize_training_logs(const std::filesystem::path & output_root, bool write_grad_stats_log) {
    if (!write_text_file(output_root / "loss.log", qlora_serialize_loss_log({}))) {
        return false;
    }

    const std::filesystem::path grad_stats_path = output_root / "grad-stats.log";
    if (write_grad_stats_log) {
        return write_text_file(grad_stats_path, qlora_serialize_grad_stats_log({}));
    }

    std::error_code ec;
    std::filesystem::remove(grad_stats_path, ec);
    return !ec;
}

static bool write_checkpoint_from_snapshot(const qlora_save_job & job, std::string & error) {
    const std::filesystem::path root = job.output_root;
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    if (ec) {
        error = "cannot create checkpoint directory '" + root.string() + "': " + ec.message();
        return false;
    }

    const std::filesystem::path final_dir = root / step_name("checkpoint-", job.step);
    const std::filesystem::path backup_dir = final_dir.string() + ".previous";
    const auto temp_stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path temp_dir = root /
        ("." + step_name("checkpoint-", job.step) + ".tmp-" + std::to_string(temp_stamp));
    std::filesystem::create_directory(temp_dir, ec);
    if (ec) {
        error = "cannot create temporary checkpoint directory '" + temp_dir.string() + "': " + ec.message();
        return false;
    }

    bool adapter_ok = false;
    if (job.write_snapshot && job.snapshot_saved) {
        adapter_ok = link_or_copy_file(job.snapshot_path, temp_dir / "adapter.gguf");
    }
    if (!adapter_ok) {
        adapter_ok = save_lora_adapter_snapshot(job.adapter, (temp_dir / "adapter.gguf").string(), false);
    }
    const bool optimizer_ok = adapter_ok && write_optimizer_file(
        temp_dir / "optimizer.gguf", job.optimizer_state, job.optimizer_name.c_str(), job.optimizer_state_type);
    const bool loss_log_ok = optimizer_ok && write_text_file(temp_dir / "loss.log", job.loss_log);
    const bool grad_stats_log_ok = loss_log_ok && (!job.write_grad_stats_log ||
        write_text_file(temp_dir / "grad-stats.log", job.grad_stats_log));
    const bool state_ok = grad_stats_log_ok && write_text_file(temp_dir / "training-state.json", job.training_state_json);
    const bool manifest_ok = state_ok && write_text_file(temp_dir / "manifest.json", job.manifest_json);
    bool ok = manifest_ok;

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
        error = "failed to write checkpoint under '" + root.string() + "'";
        return false;
    }

    ec.clear();
    if (std::filesystem::exists(backup_dir, ec) && !ec) {
        std::filesystem::remove_all(backup_dir, ec);
    }
    if (!prune_checkpoints(root, job.checkpoint_limit)) {
        LOG_WRN("checkpoint was saved but old checkpoint pruning failed under '%s'\n", root.string().c_str());
    }
    return true;
}

bool write_qlora_save_job(const std::shared_ptr<qlora_save_job> & job, std::string & error) {
    bool all_ok = true;
    if (job->write_snapshot) {
        job->snapshot_saved = save_lora_adapter_snapshot(
            job->adapter, job->snapshot_path.string(), false);
        if (job->snapshot_saved) {
            if (!prune_snapshots(job->output_root, job->snapshot_limit)) {
                LOG_WRN("snapshot was saved but old snapshot pruning failed\n");
            }
        } else {
            all_ok = false;
            error = "failed to write adapter snapshot '" + job->snapshot_path.string() + "'";
        }
    }
    if (job->write_checkpoint) {
        std::string checkpoint_error;
        job->checkpoint_saved = write_checkpoint_from_snapshot(*job, checkpoint_error);
        if (!job->checkpoint_saved) {
            all_ok = false;
            if (!error.empty()) error += "; ";
            error += checkpoint_error;
        }
    }
    if (job->write_final_adapter) {
        job->final_adapter_saved = save_lora_adapter_snapshot(
            job->adapter, job->final_adapter_path.string(), false);
        if (!job->final_adapter_saved) {
            all_ok = false;
            if (!error.empty()) error += "; ";
            error += "failed to write final adapter '" + job->final_adapter_path.string() + "'";
        }
    }
    if ((job->snapshot_saved || job->checkpoint_saved) &&
        !write_text_file(job->output_root / "loss.log", job->loss_log)) {
        all_ok = false;
        if (!error.empty()) error += "; ";
        error += "failed to update loss log";
    }
    if ((job->snapshot_saved || job->checkpoint_saved) && job->write_grad_stats_log &&
        !write_text_file(job->output_root / "grad-stats.log", job->grad_stats_log)) {
        all_ok = false;
        if (!error.empty()) error += "; ";
        error += "failed to update gradient statistics log";
    }
    return all_ok;
}


std::filesystem::path qlora_checkpoint_path(const std::filesystem::path & output_root, int64_t step) {
    return output_root / step_name("checkpoint-", step);
}

std::filesystem::path qlora_snapshot_path(const std::filesystem::path & output_root, int64_t step) {
    return output_root / step_name("snapshot-", step, ".gguf");
}
