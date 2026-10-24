set(QLORA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

add_executable(test-qlora-adapter
    test-adapter.cpp
    "${QLORA_DIR}/adapter.cpp"
)
target_include_directories(test-qlora-adapter PRIVATE "${QLORA_DIR}")
target_link_libraries(test-qlora-adapter PRIVATE llama-common llama)
target_compile_features(test-qlora-adapter PRIVATE cxx_std_17)
add_test(NAME test-qlora-adapter COMMAND test-qlora-adapter)

add_executable(test-qlora-dataset-format
    test-dataset-format.cpp
    "${QLORA_DIR}/dataset-format.cpp"
)
target_include_directories(test-qlora-dataset-format PRIVATE
    "${QLORA_DIR}"
    "${CMAKE_SOURCE_DIR}/vendor"
)
target_link_libraries(test-qlora-dataset-format PRIVATE llama-common)
target_compile_features(test-qlora-dataset-format PRIVATE cxx_std_17)
add_test(NAME test-qlora-dataset-format COMMAND test-qlora-dataset-format)

add_executable(test-qlora-dataset-thread-pool
    test-dataset-thread-pool.cpp
    "${QLORA_DIR}/dataset-thread-pool.cpp"
)
target_include_directories(test-qlora-dataset-thread-pool PRIVATE "${QLORA_DIR}")
target_link_libraries(test-qlora-dataset-thread-pool PRIVATE ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(test-qlora-dataset-thread-pool PRIVATE cxx_std_17)
add_test(NAME test-qlora-dataset-thread-pool COMMAND test-qlora-dataset-thread-pool)

add_executable(test-qlora-dataset-template
    test-dataset-template.cpp
    "${QLORA_DIR}/dataset-template.cpp"
    "${QLORA_DIR}/dataset-format.cpp")
target_include_directories(test-qlora-dataset-template PRIVATE
    "${QLORA_DIR}"
    "${CMAKE_SOURCE_DIR}/vendor"
)
target_compile_definitions(test-qlora-dataset-template PRIVATE
    QLORA_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test-qlora-dataset-template PRIVATE llama-common)
target_compile_features(test-qlora-dataset-template PRIVATE cxx_std_17)
add_test(NAME test-qlora-dataset-template COMMAND test-qlora-dataset-template)

add_executable(test-qlora-dataset-template-hf
    test-dataset-template-hf.cpp
    "${QLORA_DIR}/dataset-template.cpp"
    "${QLORA_DIR}/dataset-format.cpp")
target_include_directories(test-qlora-dataset-template-hf PRIVATE
    "${QLORA_DIR}"
    "${CMAKE_SOURCE_DIR}/vendor"
)
target_compile_definitions(test-qlora-dataset-template-hf PRIVATE
    QLORA_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test-qlora-dataset-template-hf PRIVATE llama-common)
target_compile_features(test-qlora-dataset-template-hf PRIVATE cxx_std_17)
add_test(NAME test-qlora-dataset-template-hf COMMAND test-qlora-dataset-template-hf)

add_executable(test-qlora-dataset-template-model
    test-dataset-template-model.cpp
    "${QLORA_DIR}/dataset-template.cpp"
    "${QLORA_DIR}/dataset-format.cpp"
    "${QLORA_DIR}/dataset-tokenize.cpp"
)
target_include_directories(test-qlora-dataset-template-model PRIVATE
    "${QLORA_DIR}"
    "${CMAKE_SOURCE_DIR}/vendor"
)
target_compile_definitions(test-qlora-dataset-template-model PRIVATE
    QLORA_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
    QLORA_TEMPLATE_TEST_WITH_VOCAB
)
target_link_libraries(test-qlora-dataset-template-model PRIVATE llama-common llama)
target_compile_features(test-qlora-dataset-template-model PRIVATE cxx_std_17)

add_executable(test-qlora-dataset-reader
    test-dataset-reader.cpp
    "${QLORA_DIR}/dataset-reader.cpp"
    "${QLORA_DIR}/dataset-reader-jsonl.cpp"
    "${QLORA_DIR}/dataset-thread-pool.cpp"
    "${QLORA_DIR}/dataset-format.cpp"
)
target_include_directories(test-qlora-dataset-reader PRIVATE
    "${QLORA_DIR}"
    "${CMAKE_SOURCE_DIR}/vendor"
)
target_link_libraries(test-qlora-dataset-reader PRIVATE llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(test-qlora-dataset-reader PRIVATE cxx_std_17)
add_test(NAME test-qlora-dataset-reader COMMAND test-qlora-dataset-reader)

add_executable(test-qlora-dataset-window
    test-dataset-window.cpp
    "${QLORA_DIR}/dataset-window.cpp"
)
target_include_directories(test-qlora-dataset-window PRIVATE "${QLORA_DIR}")
target_link_libraries(test-qlora-dataset-window PRIVATE llama-common)
target_compile_features(test-qlora-dataset-window PRIVATE cxx_std_17)
add_test(NAME test-qlora-dataset-window COMMAND test-qlora-dataset-window)

add_executable(test-qlora-training-math
    test-training-math.cpp
)
target_link_libraries(test-qlora-training-math PRIVATE llama-common llama)
target_compile_features(test-qlora-training-math PRIVATE cxx_std_17)
add_test(NAME test-qlora-training-math COMMAND test-qlora-training-math)

add_executable(test-qlora-model-gradient
    test-training-model-gradient.cpp
    "${QLORA_DIR}/adapter.cpp"
    "${QLORA_DIR}/dataset-window.cpp"
)
target_include_directories(test-qlora-model-gradient PRIVATE "${QLORA_DIR}")
target_link_libraries(test-qlora-model-gradient PRIVATE llama-common llama)
target_compile_features(test-qlora-model-gradient PRIVATE cxx_std_17)

add_executable(test-qlora-schedule
    test-schedule.cpp
    "${QLORA_DIR}/schedule.cpp"
)
target_include_directories(test-qlora-schedule PRIVATE "${QLORA_DIR}")
target_link_libraries(test-qlora-schedule PRIVATE llama-common llama)
target_compile_features(test-qlora-schedule PRIVATE cxx_std_17)
add_test(NAME test-qlora-schedule COMMAND test-qlora-schedule)

add_custom_target(qlora-tests
    DEPENDS
        test-qlora-adapter
        test-qlora-dataset-format
        test-qlora-dataset-thread-pool
        test-qlora-dataset-template
        test-qlora-dataset-template-hf
        test-qlora-dataset-template-model
        test-qlora-dataset-reader
        test-qlora-dataset-window
        test-qlora-training-math
        test-qlora-model-gradient
        test-qlora-schedule
)

set(QLORA_TEST_MODEL "" CACHE FILEPATH "Optional GGUF model for the CPU QLoRA training integration test")
if (QLORA_TEST_MODEL)
    if (NOT EXISTS "${QLORA_TEST_MODEL}")
        message(FATAL_ERROR "QLORA_TEST_MODEL does not exist: ${QLORA_TEST_MODEL}")
    endif()
    if (NOT TARGET llama-finetune-qlora OR NOT TARGET llama-perplexity OR NOT TARGET test-qlora-model-gradient)
        message(FATAL_ERROR "The model-backed QLoRA test requires llama-finetune-qlora, llama-perplexity, and test-qlora-model-gradient targets")
    endif()

    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(
        NAME test-qlora-training-integration
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/test-training-integration.py"
            --trainer "$<TARGET_FILE:llama-finetune-qlora>"
            --perplexity "$<TARGET_FILE:llama-perplexity>"
            --gradient-checker "$<TARGET_FILE:test-qlora-model-gradient>"
            --model "${QLORA_TEST_MODEL}"
    )
    set_tests_properties(test-qlora-training-integration PROPERTIES
        LABELS "qlora;integration;cpu"
        TIMEOUT 1800
    )
    add_dependencies(qlora-tests llama-finetune-qlora llama-perplexity)
endif()
