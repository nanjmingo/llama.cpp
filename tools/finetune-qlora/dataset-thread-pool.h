#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

size_t dataset_thread_count(size_t requested, size_t task_count);

class dataset_thread_pool {
public:
    explicit dataset_thread_pool(size_t thread_count);
    ~dataset_thread_pool();

    dataset_thread_pool(const dataset_thread_pool &) = delete;
    dataset_thread_pool & operator=(const dataset_thread_pool &) = delete;

    void parallel_for(size_t task_count, std::function<void(size_t)> function);

private:
    void process_tasks();
    void worker_loop();

    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable work_ready;
    std::condition_variable batch_done;
    std::function<void(size_t)> task;
    std::atomic<size_t> next{0};
    std::atomic<bool> cancelled{false};
    size_t task_count = 0;
    size_t generation = 0;
    size_t completed = 0;
    std::exception_ptr failure;
    bool stopping = false;
};
