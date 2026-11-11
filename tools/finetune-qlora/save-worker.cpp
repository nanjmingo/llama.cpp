#include "save-worker.h"

#include <exception>
#include <utility>

qlora_save_worker::qlora_save_worker() {
    // Start the worker only after all members have been initialized.
    thread = std::thread(&qlora_save_worker::run, this);
}

qlora_save_worker::~qlora_save_worker() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
    }
    condition.notify_one();
    if (thread.joinable()) {
        thread.join();
    }
}

bool qlora_save_worker::submit(std::function<bool(std::string &)> next_task) {
    std::lock_guard<std::mutex> lock(mutex);
    if (stopping || busy || result_ready || !next_task) {
        return false;
    }
    task = std::move(next_task);
    busy = true;
    condition.notify_one();
    return true;
}

bool qlora_save_worker::poll(qlora_save_result & completed) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!result_ready) {
        return false;
    }
    completed = std::move(result);
    result = {};
    result_ready = false;
    return true;
}

bool qlora_save_worker::wait(qlora_save_result & completed) {
    std::unique_lock<std::mutex> lock(mutex);
    completed_condition.wait(lock, [this] { return result_ready || (!busy && stopping); });
    if (!result_ready) {
        return false;
    }
    completed = std::move(result);
    result = {};
    result_ready = false;
    return true;
}

void qlora_save_worker::run() {
    for (;;) {
        std::function<bool(std::string &)> current_task;
        {
            std::unique_lock<std::mutex> lock(mutex);
            condition.wait(lock, [this] { return stopping || static_cast<bool>(task); });
            if (stopping && !task) {
                return;
            }
            current_task = std::move(task);
        }

        qlora_save_result completed;
        try {
            completed.success = current_task(completed.error);
        } catch (const std::exception & ex) {
            completed.error = ex.what();
        } catch (...) {
            completed.error = "unknown exception in save worker";
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            result = std::move(completed);
            result_ready = true;
            busy = false;
        }
        completed_condition.notify_all();
    }
}
