#include "dataset-thread-pool.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

}

int main() {
    require(dataset_thread_count(0, 0) == 0, "empty workloads should not request workers");
    require(dataset_thread_count(1, 16) == 1, "one requested thread should remain serial");
    require(dataset_thread_count(64, 16) == 16, "worker count should not exceed task count");
    require(dataset_thread_count(100, 1000) == 64, "explicit worker count should be capped");

    dataset_thread_pool pool(4);
    std::vector<size_t> output(2048);
    pool.parallel_for(output.size(), [&](size_t index) {
        output[index] = index + 1;
    });
    for (size_t i = 0; i < output.size(); ++i) {
        require(output[i] == i + 1, "parallel tasks should complete exactly once and retain their indexes");
    }

    pool.parallel_for(1, [&](size_t index) {
        output[index] = 7;
    });
    require(output[0] == 7, "single tasks should complete on the calling thread");
    pool.parallel_for(0, [&](size_t) {
        std::exit(2);
    });

    bool caught = false;
    try {
        pool.parallel_for(32, [](size_t index) {
            if (index == 3) {
                throw std::runtime_error("expected test failure");
            }
        });
    } catch (const std::runtime_error &) {
        caught = true;
    }
    require(caught, "task exceptions should be rethrown on the calling thread");

    std::vector<size_t> after_error(64);
    pool.parallel_for(after_error.size(), [&](size_t index) {
        after_error[index] = index;
    });
    for (size_t i = 0; i < after_error.size(); ++i) {
        require(after_error[i] == i, "the pool should remain usable after a task throws");
    }
    return 0;
}
