#include "save-worker.h"

#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_task_runs_off_caller_thread_and_returns_result() {
    qlora_save_worker worker;
    const std::thread::id caller_thread = std::this_thread::get_id();
    std::promise<std::thread::id> started_promise;
    std::future<std::thread::id> started = started_promise.get_future();
    std::promise<void> release_promise;
    std::shared_future<void> release = release_promise.get_future().share();

    require(worker.submit([&](std::string &) {
        started_promise.set_value(std::this_thread::get_id());
        release.wait();
        return true;
    }), "failed to submit save task");
    require(started.wait_for(std::chrono::seconds(2)) == std::future_status::ready,
            "save task did not start on worker thread");
    require(started.get() != caller_thread, "save task ran on the caller thread");

    qlora_save_result result;
    require(!worker.poll(result), "save task completed before the worker was released");
    require(!worker.submit([](std::string &) { return true; }),
            "worker accepted a second task while the first task was active");

    release_promise.set_value();
    require(worker.wait(result), "worker did not report task completion");
    require(result.success, "successful save task was reported as failed");
}

void test_worker_reports_failure_and_can_be_reused() {
    qlora_save_worker worker;
    require(worker.submit([](std::string & error) {
        error = "expected test failure";
        return false;
    }), "failed to submit failing save task");

    qlora_save_result result;
    require(worker.wait(result), "worker did not report failed task completion");
    require(!result.success && result.error == "expected test failure",
            "worker did not preserve task failure details");

    require(worker.submit([](std::string &) { return true; }),
            "worker could not accept a task after previous completion was consumed");
    require(worker.wait(result) && result.success,
            "worker did not complete the task submitted after a failure");
}

}

int main() {
    try {
        test_task_runs_off_caller_thread_and_returns_result();
        test_worker_reports_failure_and_can_be_reused();
        std::cout << "save worker tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
