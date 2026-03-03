# MyWebServer

一个**生产级 C++ 高性能 WebServer**，支持静态文件服务和 AI 推理网关：

- 页面：`GET /predict.html`
- 预测接口：`POST /api/predict`（Body 为数字字符串，例如 `50`）

## 🚀 核心技术亮点

| 特性 | 实现 | 优势 |
|------|------|------|
| **IO 多路复用** | Epoll (ET+ONESHOT) | 吞吐高，避免竞态 |
| **并发模型** | Reactor + 线程池 | 主线程不阻塞，可支持 10K+ 连接 |
| **零拷贝传输** | mmap + writev | 响应头+文件一次 syscall，减少 2 倍拷贝 |
| **连接管理** | 堆定时器 (O(log n)) | 自动清除超时连接，心跳延长 |
| **AI 集成** | ONNX Runtime | 端到端 PyTorch → C++ 推理网关 |
| **异步日志** | 双缓冲队列 + 后台线程 | 日志 IO 不阻塞业务线程，按日期自动分割 |
| **配置管理** | 环境变量 + .env | 符合 12-Factor，避免硬编码密码 |

## 📊 性能指标（测试环境：WSL2 4核 8G）

> 测试工具：[wrk](https://github.com/wg/wrk) | 测试时长：每组 10s

### 静态文件服务 `GET /predict.html`

| 并发模型 | QPS | 平均延迟 | 最大延迟 | 吞吐量 | 异常 |
|----------|-----|----------|----------|--------|------|
| 1 线程 / 10 连接 | **2,138** | 310µs | 1.40ms | 3.09 MB/s | 0 |
| 4 线程 / 100 连接 | **1,933** | 568µs | 7.93ms | 2.80 MB/s | 1 timeout |
| 4 线程 / 500 连接 | **4,019** | 5.57ms | 51.36ms | 5.81 MB/s | 0 |
| 8 线程 / 1000 连接 | **2,823** | 20.98ms | 64.81ms | 4.08 MB/s | 169 connect |

> 峰值 QPS **4,019**（500 并发），1000 并发下 169 个 connect 错误来自系统 fd 软限制

### AI 推理接口 `POST /api/predict`

| 并发模型 | QPS | 平均延迟 | 最大延迟 | 吞吐量 |
|----------|-----|----------|----------|--------|
| 1 线程 / 10 连接 | **1,904** | 366µs | 3.14ms | 198 KB/s |
| 4 线程 / 100 连接 | **2,501** | 636µs | 6.32ms | 261 KB/s |

> 推理 QPS 达 **2,500**，ONNX Runtime C++ 推理单次延迟 < 1ms

## 1. 环境准备

- Linux / WSL
- CMake + g++
- MySQL（本地可连）
- ONNX Runtime 动态库（本项目默认用本地路径）

## 🔒 安全特性（生产级）

✅ **已实现的安全加固**：
- SIGPIPE / SIGTERM / SIGINT 优雅处理（防崩溃）
- 请求体大小限制（防 OOM 攻击，默认 1MB）
- 并发连接数限制（防资源耗尽）
- HTTP Keep-Alive 支持（防 Slowloris）
- 路径规范化检查（防路径遍历）

## 2. 安全配置（避免明文密码）

服务启动时会读取以下环境变量：

- `MYSQL_USER`（默认 `root`）
- `MYSQL_PASSWORD`（默认空）
- `MYSQL_DB`（默认 `test`）
- `SERVER_PORT`（默认 `8080`）
- `SERVER_SRC_DIR`（默认 `/home/wjh/MyWebServer`）
- `MODEL_PATH`（默认 `/home/wjh/MyWebServer/test_model.onnx`）

示例：

```bash
export MYSQL_USER=wjh
export MYSQL_PASSWORD='你的密码'
export MYSQL_DB=test
export SERVER_PORT=8080
export SERVER_SRC_DIR=/home/wjh/MyWebServer
export MODEL_PATH=/home/wjh/MyWebServer/test_model.onnx
```

## 3. 构建与启动

推荐一条命令启动（自动读取 `.env`）：

```bash
cd ~/MyWebServer
./run.sh
```

手动方式：

```bash
cd ~/MyWebServer/build
cmake ..
make
./server
```

启动后访问：

- 页面：`http://127.0.0.1:8080/predict.html`
- **优雅关闭**：按 `Ctrl+C` 或 `kill -TERM <pid>`（自动完成连接生命周期）

## 4. 接口测试

```bash
# 静态文件服务
curl http://127.0.0.1:8080/predict.html

# AI 推理接口
curl -X POST --data "50" http://127.0.0.1:8080/api/predict
```

## 5. 压测（性能验证）

```bash
# 安装 wrk
sudo apt-get install wrk

# 静态文件压测 (GET)
wrk -t4 -c500 -d10s http://127.0.0.1:8080/predict.html

# AI 推理压测 (POST)
wrk -t4 -c100 -d10s -s post.lua http://127.0.0.1:8080/api/predict

# post.lua 内容：
# wrk.method = "POST"
# wrk.headers["Content-Type"] = "application/x-www-form-urlencoded"
# wrk.body = "feature1=1.0&feature2=2.0&feature3=3.0&feature4=4.0"
```

## 6. 常用运维命令

```bash
# 查看进程
pgrep -af server

# 查看端口监听
ss -ltnp | grep 8080

# 杀死服务
pkill -f "MyWebServer/build/server"
```

## 7. 仓库规范

已通过 `.gitignore` 忽略以下本地/敏感内容：

- `.venv/`
- `.env` / `.env.*`
- `build/`
- `log/` / `*.log`
- `onnxruntime/`
- `test_model.onnx*`

## 📚 架构详解

### 并发模型（Reactor 架构）

```
         主线程 (Main Reactor)
              |
         Epoll(ET) -- 监听 listenFd
              |
      +-------+--------+
      |                |
   新连接            活跃连接 (EPOLLIN/EPOLLOUT/EPOLLERR)
      |                |
   加入定时器         丢到线程池执行
                       |
              +--------+--------+--------+
              |        |        |        |
           Worker   Worker   Worker   Worker (线程池)
              |        |        |        |
           读HTTP  解析请求  生成响应  writev 零拷贝
                       |
                   mmap + iov
                   (response header + file)

    所有线程 --LOG_INFO/WARN/ERROR--> [有界阻塞队列] --> 后台写盘线程 --> log/YYYY_MM_DD.log
```

### 核心模块职责

| 模块 | 职责 | 关键设计 |
|------|------|---------|
| **WebServer** | 主框架，事件分派 | Epoll + ONESHOT + 非阻塞 |
| **HttpConn** | HTTP 解析和响应 | 简易状态机，mmap，writev |
| **HeapTimer** | 连接超时管理 | 堆+hash，O(log n) 调整 |
| **ThreadPool** | 异步任务执行 | 生产者-消费者，完美转发 |
| **Buffer** | 数据缓冲 | readPos/writePos，readv |
| **SqlConnPool** | DB 连接复用 | 单例+信号量+互斥，线程安全 |
| **AIEngine** | ONNX 推理 | 单例，全局互斥保护（可优化） |
| **Log** | 异步日志 | 有界阻塞队列 + 后台写盘线程，四级日志 |

### 关键路径性能优化

1. **零拷贝**：文件通过 mmap 映射，不进用户空间；writev 一次 syscall 完成响应头+文件
2. **ET 模式**：减少 epoll_wait 通知次数；+ONESHOT 防并发竞态
3. **定时器**：堆+hash 结构，O(log n) 调整；自动清除僵尸连接
4. **线程池**：避免"一线程一连接"爆炸；完美转发减少拷贝5. **异步日志**：有界阻塞队列背压，后台单线程顺序写盘；日志 IO 完全不阻塞业务路径
## 🎓 面试亮点速记

**30秒版本**：
> 这是一个高性能 WebServer，用 Epoll(ET) + 线程池 Reactor 架构，wrk 压测峰值 **4000+ QPS**（500 并发零错误）。核心亮点是零拷贝（mmap+writev）、线程安全堆定时器（mutex 保护）、异步日志（有界阻塞队列+后台写盘）、SIGTERM 优雅关闭。集成 ONNX Runtime，推理接口吞吐 **2500 QPS**。

**扩展版**：
> 项目从网络编程（Epoll）、并发（线程池）、数据结构（堆定时器）、系统优化（零拷贝）、异步日志（有界队列+后台写盘）多个维度展示工程能力。特别是 ET 模式、ONESHOT、mmap+writev 等细节，体现了对性能的执着。安全方面补了 SIGPIPE/SIGTERM、请求体限制等生产级特性。经 wrk 压测验证，500 并发峰值 4000+ QPS，AI 推理 2500+ QPS，1000 并发无崩溃。
