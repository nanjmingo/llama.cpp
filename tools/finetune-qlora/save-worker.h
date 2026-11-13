#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

struct qlora_save_result {
    bool success = false;
    std::string error;
};

class qlora_save_worker {
public:
    qlora_save_worker();
    ~qlora_save_worker();

    qlora_save_worker(const qlora_save_worker &) = delete;
    qlora_save_worker & operator=(const qlora_save_worker &) = delete;

    bool submit(std::function<bool(std::string &)> task);
    bool poll(qlora_save_result & result);
    bool wait(qlora_save_result & result);

private:
    void run();

    std::mutex mutex;
    std::condition_variable condition;
    std::condition_variable completed_condition;
    std::thread thread;
    std::function<bool(std::string &)> task;
    qlora_save_result result;
    bool stopping = false;
    bool busy = false;
    bool result_ready = false;
};
