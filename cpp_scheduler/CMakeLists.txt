cmake_minimum_required(VERSION 3.14)
set(CMAKE_TLS_VERIFY OFF)
set(FETCHCONTENT_QUIET OFF) # 开启下载进度显示
project(cpp_scheduler VERSION 1.0.0 LANGUAGES CXX)

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 编译选项
if(MSVC)
    add_compile_options(/utf-8 /W4 /permissive- /O2 /Oi /Ot /Oy /GL)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native -fno-finite-math-only)
endif()

# 查找依赖
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# libuv: 使用系统安装的版本
pkg_check_modules(LIBUV REQUIRED libuv)
if(NOT LIBUV_FOUND)
    message(FATAL_ERROR "请安装libuv: sudo apt install libuv1-dev")
endif()

# llama.cpp 依赖 (使用本地源码)
set(LLAMA_CPP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/llama.cpp-master")
message(STATUS "Using local llama.cpp from: ${LLAMA_CPP_SOURCE_DIR}")
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
option(ENABLE_LLAMA_CUDA "Enable llama.cpp CUDA backend" OFF)
if(ENABLE_LLAMA_CUDA)
    set(LLAMA_CUDA ON CACHE BOOL "" FORCE)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
else()
    set(LLAMA_CUDA OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
endif()
add_subdirectory(${LLAMA_CPP_SOURCE_DIR} ${CMAKE_BINARY_DIR}/llama.cpp)

# Pybind11: 使用系统安装的版本
find_package(pybind11 REQUIRED)
if(NOT pybind11_FOUND)
    message(FATAL_ERROR "请安装pybind11: sudo apt install pybind11-dev")
endif()

# 源文件
file(GLOB_RECURSE CORE_SOURCES "core/*.cpp")
file(GLOB_RECURSE WORKER_SOURCES "workers/*.cpp")
file(GLOB_RECURSE QUEUE_SOURCES "queue/*.cpp")
file(GLOB_RECURSE UTILS_SOURCES "utils/*.cpp")

# 创建库
add_library(async_scheduler STATIC
    ${CORE_SOURCES}
    ${WORKER_SOURCES}
    ${QUEUE_SOURCES}
    ${UTILS_SOURCES}
)

# Python 扩展模块
pybind11_add_module(scheduler_py bindings/python_bindings.cpp)
target_link_libraries(scheduler_py PRIVATE async_scheduler)
target_include_directories(scheduler_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 链接依赖
target_link_libraries(async_scheduler PRIVATE
    Threads::Threads
    ${LIBUV_LIBRARIES}
    llama
    common # llama.cpp common library
)

if(MSVC)
    target_compile_options(async_scheduler PRIVATE /utf-8 /Zc:char8_t- /W4 /permissive- /O2 /Oi /Ot /Oy /GL)
    target_compile_options(scheduler_py PRIVATE /utf-8 /Zc:char8_t- /W4 /permissive- /O2 /Oi /Ot /Oy /GL)
else()
    target_compile_options(async_scheduler PRIVATE -Wall -Wextra -O3 -march=native -fno-finite-math-only)
    target_compile_options(scheduler_py PRIVATE -Wall -Wextra -O3 -march=native -fno-finite-math-only)
endif()

# 包含目录
target_include_directories(async_scheduler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LLAMA_CPP_SOURCE_DIR}
    ${LLAMA_CPP_SOURCE_DIR}/common
    ${LIBUV_INCLUDE_DIRS}
)

# 启用测试选项
option(CPP_SCHEDULER_BUILD_TESTS "Build cpp_scheduler tests" OFF)

# 如果启用测试，添加测试子目录
if(CPP_SCHEDULER_BUILD_TESTS)
    message(STATUS "Building tests enabled")
    enable_testing()
    add_subdirectory(tests)
endif()

# 安装配置
install(TARGETS async_scheduler
    ARCHIVE DESTINATION lib
)
