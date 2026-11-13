add_executable(eval-qlora-model-precision EXCLUDE_FROM_ALL
    eval-model-precision.cpp
    options.cpp
    model.cpp
    run.cpp
    ../adapter.cpp
    ../adapter-save.cpp
    ../dataset-window.cpp
)
target_include_directories(eval-qlora-model-precision PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/ggml/include"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/vendor"
)
target_link_libraries(eval-qlora-model-precision PRIVATE llama-common llama)
target_compile_features(eval-qlora-model-precision PRIVATE cxx_std_17)

add_executable(eval-qlora-dataset EXCLUDE_FROM_ALL
    eval-dataset.cpp
    ../dataset-reader.cpp
    ../dataset-reader-jsonl.cpp
    ../dataset-thread-pool.cpp
    ../dataset-parser.cpp
    ../dataset-renderer.cpp
    ../dataset-tokenize.cpp
)
target_include_directories(eval-qlora-dataset PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${CMAKE_SOURCE_DIR}/vendor"
)
target_link_libraries(eval-qlora-dataset PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(eval-qlora-dataset PRIVATE cxx_std_17)

add_custom_target(llamango-eval-tools
    DEPENDS eval-qlora-model-precision eval-qlora-dataset
)
