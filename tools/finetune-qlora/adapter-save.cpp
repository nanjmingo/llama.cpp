#include "adapter.h"
#include "adapter-internal.h"

#include "common.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string basename_from_path(const std::string & path) {
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

}

bool capture_lora_adapter_snapshot(const lora_tensor_refs & tensors,
                                   const std::string & arch,
                                   float alpha,
                                   const std::string & base_model_path,
                                   lora_adapter_snapshot & snapshot,
                                   std::string * error) {
    try {
        if (error) error->clear();
        if (tensors.empty()) {
            if (error) *error = "no LoRA tensors are available";
            return false;
        }

        lora_adapter_snapshot captured;
        captured.arch = arch;
        captured.base_model = basename_from_path(base_model_path);
        captured.alpha = alpha;
        captured.tensors.reserve(tensors.size());

        for (const auto & entry : tensors) {
            lora_tensor_snapshot item;
            item.name = entry.first;
            const std::pair<ggml_tensor *, const char *> components[] = {
                { entry.second.first, "lora_a" },
                { entry.second.second, "lora_b" },
            };
            std::vector<uint8_t> * outputs[] = { &item.data_a, &item.data_b };
            enum ggml_type * types[] = { &item.type_a, &item.type_b };
            int32_t * n_dims[] = { &item.n_dims_a, &item.n_dims_b };
            int64_t * shapes[] = { item.ne_a, item.ne_b };

            for (size_t i = 0; i < 2; ++i) {
                const ggml_tensor * tensor = components[i].first;
                const std::string tensor_name = item.name + "." + components[i].second;
                if (!tensor || !qlora_adapter_detail::lora_type_is_supported(tensor->type, tensor_name) ||
                    ggml_n_dims(tensor) <= 0 || ggml_n_dims(tensor) > GGML_MAX_DIMS) {
                    if (error) *error = "invalid LoRA tensor " + tensor_name;
                    return false;
                }

                const size_t nbytes = ggml_nbytes(tensor);
                outputs[i]->resize(nbytes);
                ggml_backend_tensor_get(tensor, outputs[i]->data(), 0, nbytes);

                std::vector<float> values;
                if (!qlora_adapter_detail::tensor_read_f32_values(outputs[i]->data(), tensor->type, ggml_nelements(tensor), values)) {
                    if (error) *error = "cannot inspect LoRA tensor " + tensor_name;
                    return false;
                }
                if (!std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); })) {
                    if (error) *error = "non-finite value in LoRA tensor " + tensor_name;
                    return false;
                }

                *types[i] = tensor->type;
                *n_dims[i] = ggml_n_dims(tensor);
                std::copy_n(tensor->ne, GGML_MAX_DIMS, shapes[i]);
            }
            captured.tensors.push_back(std::move(item));
        }

        snapshot = std::move(captured);
        return true;
    } catch (const std::exception & ex) {
        if (error) *error = ex.what();
        return false;
    } catch (...) {
        if (error) *error = "unknown error while capturing LoRA tensors";
        return false;
    }
}

bool save_lora_adapter_snapshot(const lora_adapter_snapshot & snapshot,
                                const std::string & out_path,
                                bool log_success) {
    if (snapshot.tensors.empty()) {
        LOG_ERR("%s: refusing to save an adapter with no LoRA tensors\n", __func__);
        return false;
    }

    size_t data_size = 0;
    for (const lora_tensor_snapshot & item : snapshot.tensors) {
        if (item.data_a.size() > SIZE_MAX - data_size) return false;
        data_size += item.data_a.size();
        if (item.data_b.size() > SIZE_MAX - data_size) return false;
        data_size += item.data_b.size();
    }
    if (snapshot.tensors.size() > SIZE_MAX / 4096) return false;
    const size_t metadata_capacity = std::max<size_t>(4 * 1024 * 1024, snapshot.tensors.size() * 4096);
    if (data_size > SIZE_MAX - metadata_capacity) return false;
    ggml_init_params params = {};
    params.mem_size = metadata_capacity;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context_ptr tensor_ctx(ggml_init(params));
    if (!tensor_ctx) {
        LOG_ERR("%s: failed to allocate adapter snapshot metadata\n", __func__);
        return false;
    }

    gguf_context_ptr gctx(gguf_init_empty());
    if (!gctx) {
        LOG_ERR("%s: failed to allocate GGUF context\n", __func__);
        return false;
    }
    gguf_set_val_str(gctx.get(), "general.type", "adapter");
    gguf_set_val_str(gctx.get(), "general.architecture", snapshot.arch.c_str());
    gguf_set_val_str(gctx.get(), "adapter.type", "lora");
    gguf_set_val_f32(gctx.get(), "adapter.lora.alpha", snapshot.alpha);
    gguf_set_val_str(gctx.get(), "adapter.base_model", snapshot.base_model.c_str());

    for (const lora_tensor_snapshot & item : snapshot.tensors) {
        const struct {
            const char * suffix;
            enum ggml_type type;
            int32_t n_dims;
            const int64_t * ne;
            const std::vector<uint8_t> * data;
        } components[] = {
            { ".lora_a", item.type_a, item.n_dims_a, item.ne_a, &item.data_a },
            { ".lora_b", item.type_b, item.n_dims_b, item.ne_b, &item.data_b },
        };

        for (const auto & component : components) {
            if (!qlora_adapter_detail::lora_type_is_supported(component.type, item.name + component.suffix) ||
                component.n_dims <= 0 || component.n_dims > GGML_MAX_DIMS) {
                return false;
            }
            ggml_tensor * tensor = ggml_new_tensor(tensor_ctx.get(), component.type,
                                                   component.n_dims, component.ne);
            if (!tensor || ggml_nbytes(tensor) != component.data->size()) return false;
            const std::string tensor_name = item.name + component.suffix;
            ggml_set_name(tensor, tensor_name.c_str());
            gguf_add_tensor(gctx.get(), tensor);
        }
    }

    const std::string temp_path = out_path + ".tmp";
    std::remove(temp_path.c_str());
    std::ofstream fout(temp_path, std::ios::binary | std::ios::trunc);
    if (!fout.is_open()) {
        LOG_ERR("%s: cannot open %s for writing\n", __func__, temp_path.c_str());
        return false;
    }

    const size_t meta_size = gguf_get_meta_size(gctx.get());
    std::vector<uint8_t> metadata(meta_size, 0);
    fout.write(reinterpret_cast<const char *>(metadata.data()), static_cast<std::streamsize>(metadata.size()));
    const size_t alignment = gguf_get_alignment(gctx.get());
    const std::vector<char> padding(alignment, 0);
    if (fout.good()) {
        for (const lora_tensor_snapshot & item : snapshot.tensors) {
            for (const std::vector<uint8_t> * data : { &item.data_a, &item.data_b }) {
                fout.write(reinterpret_cast<const char *>(data->data()), static_cast<std::streamsize>(data->size()));
                const size_t pad_size = GGML_PAD(data->size(), alignment) - data->size();
                if (pad_size > 0) {
                    fout.write(padding.data(), static_cast<std::streamsize>(pad_size));
                }
                if (!fout.good()) break;
            }
            if (!fout.good()) break;
        }
    }
    if (fout.good()) {
        gguf_get_meta_data(gctx.get(), metadata.data());
        fout.seekp(0);
        fout.write(reinterpret_cast<const char *>(metadata.data()), static_cast<std::streamsize>(metadata.size()));
    }
    fout.close();
    const bool write_ok = fout.good();
    if (!write_ok) {
        LOG_ERR("%s: failed while writing adapter %s\n", __func__, out_path.c_str());
        std::remove(temp_path.c_str());
        return false;
    }

    if (std::rename(temp_path.c_str(), out_path.c_str()) != 0) {
        const std::string backup_path = out_path + ".previous";
        if (std::rename(out_path.c_str(), backup_path.c_str()) != 0) {
            LOG_ERR("%s: cannot replace %s with completed adapter file\n", __func__, out_path.c_str());
            std::remove(temp_path.c_str());
            return false;
        }
        if (std::rename(temp_path.c_str(), out_path.c_str()) != 0) {
            const int restore_result = std::rename(backup_path.c_str(), out_path.c_str());
            LOG_ERR("%s: cannot install completed adapter at %s\n", __func__, out_path.c_str());
            if (restore_result != 0) {
                LOG_ERR("%s: previous adapter remains at %s and could not be restored automatically\n",
                        __func__, backup_path.c_str());
            }
            std::remove(temp_path.c_str());
            return false;
        }
        std::remove(backup_path.c_str());
    }

    if (log_success) {
        LOG_INF("%s: adapter saved to %s\n", __func__, out_path.c_str());
    }
    return true;
}

bool lora_tensors_are_finite(const lora_tensor_refs & tensors, std::string * error) {
    lora_adapter_snapshot snapshot;
    return capture_lora_adapter_snapshot(tensors, "", 0.0f, "", snapshot, error);
}

bool save_adapter(const lora_tensor_refs & tensors,
                  const std::string & out_path,
                  const std::string & arch,
                  float alpha,
                  const std::string & base_model_path,
                  bool log_success,
                  bool * validation_failed) {
    if (validation_failed) *validation_failed = false;
    lora_adapter_snapshot snapshot;
    std::string validation_error;
    if (!capture_lora_adapter_snapshot(tensors, arch, alpha, base_model_path, snapshot, &validation_error)) {
        if (validation_failed) *validation_failed = true;
        LOG_ERR("%s: refusing to save adapter: %s\n", __func__, validation_error.c_str());
        return false;
    }
    return save_lora_adapter_snapshot(snapshot, out_path, log_success);
}
