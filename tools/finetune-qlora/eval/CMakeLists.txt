add_executable(eval-qlora-model-precision EXCLUDE_FROM_ALL
    eval-model-precision.cpp
    eval-model-precision-options.cpp
)
target_link_libraries(eval-qlora-model-precision PRIVATE qlora-model-experiment-support llama-common llama)
target_compile_features(eval-qlora-model-precision PRIVATE cxx_std_17)

add_custom_target(qlora-eval-tools
    DEPENDS eval-qlora-model-precision
)
