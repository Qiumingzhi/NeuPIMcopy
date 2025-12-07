cmake_minimum_required(VERSION 3.15.0)
set(project_name "AiFrameworkSim")
project(${project_name})

# 强制开启位置无关代码 (Fix: creating DT_TEXTREL in a PIE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==========================================
# Conan 2.x 依赖包查找
# ==========================================
find_package(Boost REQUIRED CONFIG)
find_package(robin_hood REQUIRED CONFIG)
find_package(spdlog REQUIRED CONFIG)
find_package(nlohmann_json REQUIRED CONFIG)
# zlib and libbacktrace are usually dependencies of others or system libs, 
# but if needed explicitly:
# find_package(ZLIB REQUIRED CONFIG)
# find_package(libbacktrace REQUIRED CONFIG)

# ==========================================
# C++ 设置
# ==========================================
set(CMAKE_CXX_STANDARD 17)
set(ONNX_ML 1)
set(JSON_BuildTests OFF CACHE INTERNAL "")

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/build/bin")
set(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/build/lib")

message("BINARY DIR ${CMAKE_BINARY_DIR}")

# 调试选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra --debug-cpp --trace -rdynamic")

# ==========================================
# 添加外部库子目录
# ==========================================
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/NewtonSim")
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/booksim")

# 强制让 booksim2 链接 Conan 的 Boost
target_link_libraries(booksim2 PUBLIC Boost::boost)

# ==========================================
# 统一在根目录构建 Simulator
# ==========================================

# 1. 搜集源码
file(GLOB_RECURSE SOURCES "src/*.cc")
file(GLOB_RECURSE HEADERS "src/*.h")

# 从库源码中排除 main.cc
list(FILTER SOURCES EXCLUDE REGEX ".*/main.cc$")

# 2. 定义 Simulator_lib 库目标
add_library(Simulator_lib ${SOURCES} ${HEADERS})

target_include_directories(Simulator_lib PUBLIC 
    "${PROJECT_SOURCE_DIR}/src"
)

target_link_libraries(Simulator_lib
    PUBLIC
    dramsim3
    booksim2
    Boost::boost
    Boost::program_options
    Boost::filesystem
    Boost::system
    robin_hood::robin_hood
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    stdc++fs
)

# 3. 定义 Simulator 可执行文件目标
add_executable(Simulator "${PROJECT_SOURCE_DIR}/src/main.cc")

target_include_directories(Simulator PUBLIC 
    "${PROJECT_SOURCE_DIR}/src"
)

target_link_libraries(Simulator
    PUBLIC
    Simulator_lib
    dramsim3
    booksim2
    Boost::boost
    Boost::program_options
    Boost::filesystem
    Boost::system
    robin_hood::robin_hood
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    stdc++fs
)

enable_testing()
# add_subdirectory("${PROJECT_SOURCE_DIR}/tests")