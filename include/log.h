#ifndef LOG_H
#define LOG_H

#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <memory>
#include <fstream>
#include <cstdarg>

/**
 * 异步并发日志系统
 * 
 * 设计亮点：
 * 1. 双缓冲队列 + 后台线程异步写盘，避免 IO 阻塞业务线程
 * 2. 支持按日期自动分割日志文件
 * 3. 四级日志：DEBUG / INFO / WARN / ERROR
 * 4. 线程安全：前端 mutex 保护入队，后端单线程顺序写盘
 * 5. 单例模式，全局可用
 */

class Log {
public:
    // 日志级别
    enum Level {
        DEBUG = 0,
        INFO,
        WARN,
        ERROR
    };

    // 初始化：日志目录、日志文件后缀、最大队列容量、日志级别
    void Init(const char* path = "./log", const char* suffix = ".log",
              int maxQueueCapacity = 8192, Level level = INFO);

    // 单例
    static Log* Instance();

    // 写日志（格式化）
    void Write(Level level, const char* format, ...);

    // 刷新缓冲区
    void Flush();

    // 获取/设置日志级别
    Level GetLevel() const;
    void SetLevel(Level level);

    // 日志是否已初始化
    bool IsOpen() const { return isOpen_; }

private:
    Log();
    ~Log();

    // 后台写线程函数
    void AsyncWriteThread_();

    // 追加日志等级标签
    void AppendLogLevelTitle_(Level level, char* buf, int& idx);

    // 检测日期变化，自动创建新日志文件
    void CheckDateAndOpenFile_();

    // 私有成员
    static const int LOG_BUF_SIZE = 8192;   // 单条日志最大长度
    static const int LOG_NAME_LEN = 256;    // 日志文件名最大长度

    const char* path_;
    const char* suffix_;

    Level level_;

    int today_;                     // 当前日期 (day of month)
    bool isOpen_;
    bool isAsync_;                  // 是否异步模式

    std::ofstream fp_;              // 日志文件流
    std::mutex mtx_;                // 前端入队锁
    std::mutex fileMtx_;            // 文件操作锁

    // 阻塞队列（有界）
    struct BlockQueue {
        std::mutex mtx;
        std::condition_variable condProducer;
        std::condition_variable condConsumer;
        std::queue<std::string> queue;
        size_t capacity;
        bool isClosed = false;

        void Push(const std::string& item);
        bool Pop(std::string& item);
        void Close();
        void Flush();
        bool Empty();
    };

    std::unique_ptr<BlockQueue> blockQueue_;
    std::unique_ptr<std::thread> writeThread_;
};

// ===================== 便捷宏（核心面试亮点：零开销条件编译 + 格式化）=====================
// 使用 do-while(0) 包裹保证宏在任何上下文中都是安全的

#define LOG_BASE(level, format, ...) \
    do { \
        Log* log = Log::Instance(); \
        if (log->IsOpen() && log->GetLevel() <= level) { \
            log->Write(level, format, ##__VA_ARGS__); \
        } \
    } while(0)

#define LOG_DEBUG(format, ...) LOG_BASE(Log::DEBUG, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...)  LOG_BASE(Log::INFO,  format, ##__VA_ARGS__)
#define LOG_WARN(format, ...)  LOG_BASE(Log::WARN,  format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) LOG_BASE(Log::ERROR, format, ##__VA_ARGS__)

#endif // LOG_H
