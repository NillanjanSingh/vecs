cmake_minimum_required(VERSION 4.3)
project(vecs)

set(CMAKE_CXX_STANDARD 23)

add_subdirectory(llama.cpp)

add_executable(vecs src/main.cpp src/hnsw.cpp src/chunker.cpp)

# 1. Add llama.cpp header locations alongside your own include folder
target_include_directories(vecs PRIVATE
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include
)

# 2. Link the built llama.cpp libraries to your executable
target_link_libraries(vecs PRIVATE llama ggml)
