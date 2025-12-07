CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

# project setting
set(LIB_NAME "Simulator")

# set source and headers
file(GLOB_RECURSE SRC_FILES
  "${CMAKE_SOURCE_DIR}/src/*.h"
  "${CMAKE_SOURCE_DIR}/src/*.cc"
)

# Exclude main.cc and debug_graph.cc from the library sources
list(FILTER SRC_FILES EXCLUDE REGEX ".*main\\.cc$")
list(FILTER SRC_FILES EXCLUDE REGEX ".*debug_graph\\.cc$")

# build library (without main)
add_library(${LIB_NAME}_lib ${SRC_FILES})

# build Simulator executable
add_executable(${LIB_NAME} "${CMAKE_SOURCE_DIR}/src/main.cc" ${SRC_FILES})

# build debug_graph executable
add_executable(debug_graph "${CMAKE_SOURCE_DIR}/src/debug_graph.cc")
target_link_libraries(debug_graph ${LIB_NAME}_lib)
