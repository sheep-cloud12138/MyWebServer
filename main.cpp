#include "webserver.h"
#include <iostream>
#include <signal.h>
#include <stdlib.h> // system
#include "ai_engine.h" // 【新增】
#include <vector>

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

int main() {
        // 忽略 SIGPIPE 信号，防止客户端异常断开导致服务器崩溃 (面试防挂细节)
        signal(SIGPIPE, SIG_IGN); 

        // 配置参数：
        int port = 8080;
    const char* srcDir = "/home/wjh/MyWebServer"; // 静态文件统一从项目目录读取
        
        // MySQL 连接参数 - 修改这里为你的实际配置
        const char* sqlUser = "wjh";      // 修改为你的 MySQL 用户名
        const char* sqlPwd = "031126";           // 修改为你的 MySQL 密码 (默认为空)
        const char* dbName = "test";    // 确保这个数据库存在
        
        int sqlPoolNum = 4;
        int threadNum = 8; // 八核动力

        std::cout << "Starting WebServer on port " << port << "..." << std::endl;
        std::cout << "Connecting to MySQL: user=" << sqlUser << ", db=" << dbName << std::endl;

        // 【新增】启动服务器前先加载 AI 模型
        if (!AIEngine::Instance()->LoadModel("/home/wjh/MyWebServer/test_model.onnx")) {
            std::cerr << "[Error] Failed to load AI model!" << std::endl;
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