add_executable(eval-qlora-model-precision EXCLUDE_FROM_ALL
    eval-model-precision.cpp
    eval-model-precision-options.cpp
)
target_link_libraries(eval-qlora-model-precision PRIVATE qlora-model-experiment-support llama-common llama)
target_compile_features(eval-qlora-model-precision PRIVATE cxx_std_17)

add_executable(eval-qlora-dataset EXCLUDE_FROM_ALL
    eval-dataset.cpp
    ../dataset-reader.cpp
    ../dataset-reader-jsonl.cpp
    ../dataset-thread-pool.cpp
    ../dataset-format.cpp
    ../dataset-template.cpp
    ../dataset-tokenize.cpp
)
target_include_directories(eval-qlora-dataset PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${CMAKE_SOURCE_DIR}/vendor"
)
target_link_libraries(eval-qlora-dataset PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(eval-qlora-dataset PRIVATE cxx_std_17)

add_custom_target(qlora-eval-tools
    DEPENDS eval-qlora-model-precision eval-qlora-dataset
)
