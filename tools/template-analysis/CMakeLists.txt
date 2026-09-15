set(TARGET llama-template-analysis)
add_executable(${TARGET} ${PROJECT_SOURCE_DIR}/tests/test-chat-analysis.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
