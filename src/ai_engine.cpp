#include "ai_engine.h"

AIEngine::AIEngine() : memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    // 初始化 ONNX Runtime 环境，设置日志级别为 WARNING
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "AIEngine");
}

AIEngine::~AIEngine() {}

AIEngine* AIEngine::Instance() {
    static AIEngine engine;
    return &engine;
}

bool AIEngine::LoadModel(const std::string& modelPath) {
    try {
        Ort::SessionOptions sessionOptions;
        // 设置为单线程执行图，或者根据你的 CPU 核心数调整
        sessionOptions.SetIntraOpNumThreads(1); 
        // 开启所有的图优化 (面试亮点：图优化机制)
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // 加载模型到内存
        session_ = std::make_unique<Ort::Session>(*env_, modelPath.c_str(), sessionOptions);

        // 为了简化，这里硬编码输入输出节点的名字。
        // 实际工业界会通过 session_->GetInputNameAllocated 动态获取
        inputNodeNames_ = {"input"};   // 必须与你 ONNX 模型的输入名一致
        outputNodeNames_ = {"output"}; // 必须与你 ONNX 模型的输出名一致

        std::cout << "[AIEngine] Model loaded successfully from " << modelPath << std::endl;
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[AIEngine] Failed to load model: " << e.what() << std::endl;
        return false;
    }
}

// 核心前向传播逻辑
std::vector<float> AIEngine::Predict(const std::vector<float>& inputData) {
    if (!session_) {
        std::cerr << "[AIEngine] Model not loaded!" << std::endl;
        return {};
    }

    // 加锁，确保多个 HTTP 请求打过来时，推理图的状态不乱
    std::lock_guard<std::mutex> lock(mtx_);

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
        auto outputTensors = session_->Run(
            Ort::RunOptions{nullptr},
            inputNodeNames_.data(), &inputTensor, 1,
            outputNodeNames_.data(), 1
        );

        // 4. 解析输出结果
        float* floatArray = outputTensors.front().GetTensorMutableData<float>();
        size_t outputCount = outputTensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

        std::vector<float> result(floatArray, floatArray + outputCount);
        return result;
    } catch (const Ort::Exception& e) {
        std::cerr << "[AIEngine] Inference error: " << e.what() << std::endl;
        return {};
    }
}