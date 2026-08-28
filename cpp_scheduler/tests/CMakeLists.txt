find_package(Threads REQUIRED)

set_target_properties(async_scheduler PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

if(MSVC)
    target_compile_options(async_scheduler PRIVATE /utf-8 /W4)
else()
    target_compile_options(async_scheduler PRIVATE -Wall -Wextra)
endif()
