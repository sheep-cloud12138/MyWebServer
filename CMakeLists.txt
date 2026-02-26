cmake_minimum_required(VERSION 3.10)
project(MyWebServer)

set(CMAKE_CXX_STANDARD 14)
# 加上 -O3 优化，这对 AI 推理速度很重要
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -O3")

# 【新增】定义 ONNX Runtime 的根目录变量
set(ONNX_PATH "${CMAKE_SOURCE_DIR}/onnxruntime")

# 头文件路径
include_directories(
    include
    ${ONNX_PATH}/include  # 【新增】让编译器能找到 onnxruntime_cxx_api.h
)

# 【新增】库文件路径 (让链接器能找到 libonnxruntime.so)
link_directories(${ONNX_PATH}/lib)

# 明确指定源文件
set(SRC_FILES 
    src/sqlconnpool.cpp
    src/buffer.cpp
    src/epoller.cpp
    src/httpconn.cpp
    src/webserver.cpp
    src/ai_engine.cpp     # 【新增】别忘了把 AI 引擎的源文件加进去！
)

# 生成可执行文件
add_executable(server main.cpp ${SRC_FILES})

# 链接库 
# 【新增】链接 onnxruntime 库
target_link_libraries(server mysqlclient onnxruntime)

# 【可选】设置 RPATH，这样运行的时候不用设置 LD_LIBRARY_PATH 也能找到库
# set_target_properties(server PROPERTIES INSTALL_RPATH "${ONNX_PATH}/lib")