set(TARGET llama-finetune-qlora)
set(QLORA_SOURCES
    finetune-qlora.cpp
    args.cpp
    adapter.cpp
    schedule.cpp
    dataset.cpp
    dataset-reader.cpp
    dataset-reader-jsonl.cpp
    dataset-thread-pool.cpp
    dataset-format.cpp
    dataset-template.cpp
    dataset-tokenize.cpp
    dataset-window.cpp
)

add_executable(${TARGET} ${QLORA_SOURCES})
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/vendor)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

add_library(qlora-model-experiment-support OBJECT EXCLUDE_FROM_ALL
    model-experiment/model-experiment.cpp
    model-experiment/model-experiment-forward.cpp
    model-experiment/model-experiment-step.cpp
    model-experiment/model-experiment-finite-difference.cpp
    adapter.cpp
    dataset-window.cpp
)
target_include_directories(qlora-model-experiment-support
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/model-experiment"
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_SOURCE_DIR}/ggml/include"
        "${CMAKE_SOURCE_DIR}/src"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/vendor"
)
target_link_libraries(qlora-model-experiment-support PRIVATE llama-common llama)
target_compile_features(qlora-model-experiment-support PRIVATE cxx_std_17)

if (LLAMA_BUILD_TESTS)
    add_subdirectory(tests)
endif()

add_subdirectory(eval)

install(TARGETS ${TARGET} RUNTIME)
