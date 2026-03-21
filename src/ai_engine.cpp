#include "ai_engine.h"
#include "log.h"
#include <thread>

AIEngine::AIEngine() : memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    // 初始化 ONNX Runtime 环境，设置日志级别为 WARNING
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "AIEngine");
    sessionCount_.store(0);
}

AIEngine::~AIEngine() {}

AIEngine* AIEngine::Instance() {
    static AIEngine engine;
    return &engine;
}

bool AIEngine::LoadModel(const std::string& modelPath) {
    try {
        {
            std::lock_guard<std::mutex> lock(poolMtx_);
            sessions_.clear();
            std::queue<size_t> empty;
            std::swap(availableSessions_, empty);
        }

        Ort::SessionOptions sessionOptions;
        // 每个会话内部单线程，整体靠多会话并发拉吞吐
        sessionOptions.SetIntraOpNumThreads(1); 
        // 开启所有的图优化 (面试亮点：图优化机制)
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        unsigned int hc = std::thread::hardware_concurrency();
        size_t poolSize = hc == 0 ? 4 : static_cast<size_t>(hc);
        if (poolSize > 16) {
            poolSize = 16;
        }

        sessions_.reserve(poolSize);
        for (size_t i = 0; i < poolSize; ++i) {
            sessions_.emplace_back(std::make_unique<Ort::Session>(*env_, modelPath.c_str(), sessionOptions));
        }

        {
            std::lock_guard<std::mutex> lock(poolMtx_);
            for (size_t i = 0; i < sessions_.size(); ++i) {
                availableSessions_.push(i);
            }
        }
        sessionCount_.store(sessions_.size());

        // 为了简化，这里硬编码输入输出节点的名字。
        // 实际工业界会通过 session_->GetInputNameAllocated 动态获取
        inputNodeNames_ = {"input"};   // 必须与你 ONNX 模型的输入名一致
        outputNodeNames_ = {"output"}; // 必须与你 ONNX 模型的输出名一致

        LOG_INFO("[AIEngine] Model loaded successfully from %s, session_pool=%zu", modelPath.c_str(), sessions_.size());
        return true;
    } catch (const Ort::Exception& e) {
        LOG_ERROR("[AIEngine] Failed to load model: %s", e.what());
        return false;
    }
}

// 核心前向传播逻辑
std::vector<float> AIEngine::Predict(const std::vector<float>& inputData) {
    if (sessions_.empty()) {
        LOG_ERROR("[AIEngine] Model not loaded!");
        return {};
    }

    size_t sessionIdx = 0;
    {
        std::unique_lock<std::mutex> lock(poolMtx_);
        poolCv_.wait(lock, [this]() { return !availableSessions_.empty(); });
        sessionIdx = availableSessions_.front();
        availableSessions_.pop();
    }

    // 1. 定义输入的 Tensor 形状 (假设模型要求输入形状是 [1, 特征数量])
    std::vector<int64_t> inputDims = {1, static_cast<int64_t>(inputData.size())};

    // 2. 将 C++ 的 std::vector 包装成 ONNX 认识的 Tensor
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo_, 
        const_cast<float*>(inputData.data()), inputData.size(), 
        inputDims.data(), inputDims.size()
    );

    // 3. 执行推理 (Run)
    try {
        auto outputTensors = sessions_[sessionIdx]->Run(
            Ort::RunOptions{nullptr},
            inputNodeNames_.data(), &inputTensor, 1,
            outputNodeNames_.data(), 1
        );

        // 4. 解析输出结果
        float* floatArray = outputTensors.front().GetTensorMutableData<float>();
        size_t outputCount = outputTensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

        std::vector<float> result(floatArray, floatArray + outputCount);
        {
            std::lock_guard<std::mutex> lock(poolMtx_);
            availableSessions_.push(sessionIdx);
        }
        poolCv_.notify_one();
        return result;
    } catch (const Ort::Exception& e) {
        LOG_ERROR("[AIEngine] Inference error: %s", e.what());
        {
            std::lock_guard<std::mutex> lock(poolMtx_);
            availableSessions_.push(sessionIdx);
        }
        poolCv_.notify_one();
        return {};
    }
}