#include "dataset-thread-pool.h"

#include <algorithm>
#include <utility>

size_t dataset_thread_count(size_t requested, size_t task_count) {
    if (task_count == 0) {
        return 0;
    }
    if (requested == 0) {
        requested = std::min<size_t>(16, std::max(1u, std::thread::hardware_concurrency()));
    }
    return std::clamp<size_t>(requested, 1, std::min<size_t>(64, task_count));
}

dataset_thread_pool::dataset_thread_pool(size_t thread_count) {
    const size_t worker_count = thread_count > 0 ? thread_count - 1 : 0;
    workers.reserve(worker_count);
    try {
        for (size_t i = 0; i < worker_count; ++i) {
            workers.emplace_back([this] { worker_loop(); });
        }
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        work_ready.notify_all();
        for (std::thread & worker : workers) {
            worker.join();
        }
        throw;
    }
}

dataset_thread_pool::~dataset_thread_pool() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
    }
    work_ready.notify_all();
    for (std::thread & worker : workers) {
        worker.join();
    }
}

void dataset_thread_pool::parallel_for(size_t task_count, std::function<void(size_t)> function) {
    if (task_count == 0) {
        return;
    }
    if (workers.empty() || task_count == 1) {
        for (size_t i = 0; i < task_count; ++i) {
            function(i);
        }
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        task = std::move(function);
        this->task_count = task_count;
        next.store(0, std::memory_order_relaxed);
        cancelled.store(false, std::memory_order_relaxed);
        completed = 0;
        failure = nullptr;
        ++generation;
    }
    work_ready.notify_all();
    process_tasks();
    std::exception_ptr failure_to_rethrow;
    {
        std::unique_lock<std::mutex> lock(mutex);
        batch_done.wait(lock, [this] { return completed == workers.size(); });
        failure_to_rethrow = failure;
        task = {};
        this->task_count = 0;
    }
    if (failure_to_rethrow) {
        std::rethrow_exception(failure_to_rethrow);
    }
}

void dataset_thread_pool::process_tasks() {
    while (!cancelled.load(std::memory_order_relaxed)) {
        const size_t index = next.fetch_add(1, std::memory_order_relaxed);
        if (index >= task_count) {
            break;
        }
        try {
            task(index);
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (!failure) {
                    failure = std::current_exception();
                }
            }
            cancelled.store(true, std::memory_order_relaxed);
            break;
        }
    }
}

void dataset_thread_pool::worker_loop() {
    size_t seen_generation = 0;
    while (true) {
        std::unique_lock<std::mutex> lock(mutex);
        work_ready.wait(lock, [this, seen_generation] {
            return stopping || generation != seen_generation;
        });
        if (stopping) {
            return;
        }
        seen_generation = generation;
        lock.unlock();
        process_tasks();
        lock.lock();
        ++completed;
        if (completed == workers.size()) {
            batch_done.notify_one();
        }
    }
}
