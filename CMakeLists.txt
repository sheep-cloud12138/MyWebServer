cmake_minimum_required(VERSION 3.10)
project(MyWebServer)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -O2")

# 头文件路径
include_directories(include)

# 明确指定源文件
set(SRC_FILES 
    src/sqlconnpool.cpp
    src/buffer.cpp
    src/epoller.cpp
    src/httpconn.cpp
    src/webserver.cpp
)

# 生成可执行文件
add_executable(server main.cpp ${SRC_FILES})

# 链接库 (MySQL库)
target_link_libraries(server mysqlclient)