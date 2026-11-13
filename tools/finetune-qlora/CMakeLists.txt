set(TARGET llama-finetune-qlora)
set(QLORA_SOURCES
    finetune-qlora.cpp
    training.cpp
    training-callback.cpp
    checkpoint.cpp
    save-worker.cpp
    args.cpp
    adapter.cpp
    adapter-save.cpp
    schedule.cpp
    dataset-pipeline.cpp
    dataset-reader.cpp
    dataset-reader-jsonl.cpp
    dataset-thread-pool.cpp
    dataset-parser.cpp
    dataset-renderer.cpp
    dataset-tokenize.cpp
    dataset-window.cpp
)

add_executable(${TARGET} ${QLORA_SOURCES})
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/vendor)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

add_subdirectory(eval)

install(TARGETS ${TARGET} RUNTIME)
