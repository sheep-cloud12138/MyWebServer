#ifndef AI_ENGINE_H
#define AI_ENGINE_H

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <mutex>

class AIEngine {
public:
    // 单例模式，保证全局只有一个推理引擎和模型驻留内存
    static AIEngine* Instance();

    // 加载 ONNX 模型
    bool LoadModel(const std::string& modelPath);

    // 执行推理 (假设我们的模型输入是一维 float 数组，输出也是一维 float 数组)
    std::vector<float> Predict(const std::vector<float>& inputData);

private:
    AIEngine();
    ~AIEngine();

    // ONNX Runtime 环境与会话
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memoryInfo_;

    // 模型的输入输出节点信息
    std::vector<const char*> inputNodeNames_;
    std::vector<const char*> outputNodeNames_;
    
    std::mutex mtx_; // 保证推理过程的线程安全
};

#endif // AI_ENGINE_H