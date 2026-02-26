# MyWebServer

一个基于 C++ 的高性能 WebServer，支持静态文件服务，并集成 ONNX Runtime 推理接口：

- 页面：`GET /predict.html`
- 预测接口：`POST /api/predict`（Body 为数字字符串，例如 `50`）

## 1. 环境准备

- Linux / WSL
- CMake + g++
- MySQL（本地可连）
- ONNX Runtime 动态库（本项目默认用本地路径）

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
cp .env.example .env   # 首次执行
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

## 4. 接口测试

```bash
curl -X POST --data "50" http://127.0.0.1:8080/api/predict
```

期望返回：

```text
Result: 150.000000
```

## 5. 常用运维命令

```bash
# 查看进程
pgrep -af server

# 查看端口监听
ss -ltnp | grep 8080

# 杀死服务
pkill -f "MyWebServer/build/server"
```

## 6. 仓库规范

已通过 `.gitignore` 忽略以下本地/敏感内容：

- `.venv/`
- `.env` / `.env.*`
- `build/`
- `onnxruntime/`
- `test_model.onnx*`
