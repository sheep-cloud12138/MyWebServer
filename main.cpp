#include "webserver.h"
#include "log.h"
#include <iostream>
#include <signal.h>
#include <stdlib.h> // system
#include "ai_engine.h" // 【新增】
#include <vector>
#include <fstream>
#include <string>

std::string Trim(const std::string& text) {
    size_t start = 0;
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) {
        ++start;
    }
    size_t end = text.size();
    while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return text.substr(start, end - start);
}

void LoadDotEnvIfPresent() {
    const char* candidates[] = {".env", "../.env"};
    for (const char* path : candidates) {
        std::ifstream file(path);
        if (!file.is_open()) {
            continue;
        }

        std::string line;
        while (std::getline(file, line)) {
            std::string trimmed = Trim(line);
            if (trimmed.empty() || trimmed[0] == '#') {
                continue;
            }

            size_t pos = trimmed.find('=');
            if (pos == std::string::npos) {
                continue;
            }

            std::string key = Trim(trimmed.substr(0, pos));
            std::string value = Trim(trimmed.substr(pos + 1));
            if (key.empty()) {
                continue;
            }

            if (value.size() >= 2 &&
                ((value.front() == '"' && value.back() == '"') ||
                 (value.front() == '\'' && value.back() == '\''))) {
                value = value.substr(1, value.size() - 2);
            }

            if (!getenv(key.c_str())) {
                setenv(key.c_str(), value.c_str(), 0);
            }
        }
        return;
    }
}

const char* GetEnvOrDefault(const char* key, const char* fallback) {
    const char* value = getenv(key);
    return (value && value[0] != '\0') ? value : fallback;
}

// 【新增】AI 引擎测试函数
void testAIEngine() {
    std::cout << "\n[Test AI] Testing ONNX Runtime Engine..." << std::endl;

    // 1. 获取单例并加载模型
    // 注意：路径必须正确，这里假设 test_model.onnx 在 build 目录的上一级
    bool loaded = AIEngine::Instance()->LoadModel("../test_model.onnx");
    if (!loaded) {
        std::cerr << "  [Error] Failed to load model. Check path!" << std::endl;
        return;
    }

    // 2. 准备输入数据 (输入 1.0)
    std::vector<float> inputData = { 1.0f };
    
    // 3. 执行推理
    std::cout << "  -> Input: 1.0" << std::endl;
    std::vector<float> outputData = AIEngine::Instance()->Predict(inputData);

    // 4. 验证结果
    if (!outputData.empty()) {
        std::cout << "  -> Output: " << outputData[0] << std::endl;
        if (abs(outputData[0] - 101.0f) < 0.001) {
            std::cout << "  -> Prediction Correct! (1.0 + 100 = 101.0)" << std::endl;
        } else {
            std::cout << "  -> Prediction Wrong!" << std::endl;
        }
    } else {
        std::cerr << "  -> Inference Failed!" << std::endl;
    }
}

// 全局标志位，用于优雅关闭
volatile int g_shutdown = 0;
void SignalHandler(int sig) { 
    g_shutdown = 1;
    LOG_INFO("Received signal %d, shutting down gracefully...", sig);
}

int main() {
        // 忽略 SIGPIPE 信号，防止客户端异常断开导致服务器崩溃 (面试防挂细节)
        signal(SIGPIPE, SIG_IGN);
        // 优雅关闭处理
        signal(SIGTERM, SignalHandler);
        signal(SIGINT, SignalHandler);
    LoadDotEnvIfPresent();

        // 初始化异步日志系统（后台线程写盘，队列容量 8192）
        Log::Instance()->Init("./log", ".log", 8192, Log::INFO);

        // 配置参数：
    int port = atoi(GetEnvOrDefault("SERVER_PORT", "8080"));
    const char* srcDir = GetEnvOrDefault("SERVER_SRC_DIR", "/home/wjh/MyWebServer"); // 静态文件统一从项目目录读取
        
    // MySQL 连接参数（从环境变量读取，避免明文写进代码）
    const char* sqlUser = GetEnvOrDefault("MYSQL_USER", "root");
    const char* sqlPwd = GetEnvOrDefault("MYSQL_PASSWORD", "");
    const char* dbName = GetEnvOrDefault("MYSQL_DB", "test");
        
        int sqlPoolNum = 4;
        int threadNum = 8; // 八核动力

        LOG_INFO("Starting WebServer on port %d...", port);
        LOG_INFO("Connecting to MySQL: user=%s, db=%s", sqlUser, dbName);

        // 【新增】启动服务器前先加载 AI 模型
        const char* modelPath = GetEnvOrDefault("MODEL_PATH", "/home/wjh/MyWebServer/test_model.onnx");
        if (!AIEngine::Instance()->LoadModel(modelPath)) {
            LOG_ERROR("Failed to load AI model from %s!", modelPath);
            return 1;
        }

        // 自动打开前端页面（WSL 下使用 Windows cmd.exe 打开默认浏览器）
        if (system("(sleep 1; cmd.exe /C start \"\" \"http://127.0.0.1:8080/predict.html\" >/dev/null 2>&1) &")) {}

        // 实例化服务器大管家并启动
        WebServer server(
            port, 
            srcDir, 
            sqlUser, sqlPwd, dbName, 
            sqlPoolNum, threadNum
        );

        server.Start(); // 死循环，永不退出
        //testAIEngine();
    return 0;
}