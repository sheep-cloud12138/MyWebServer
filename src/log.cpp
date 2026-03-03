#include "log.h"
#include <cstring>
#include <ctime>
#include <sys/stat.h>
#include <sys/time.h>
#include <stdarg.h>

// ==================== BlockQueue 实现 ====================

void Log::BlockQueue::Push(const std::string& item) {
    std::unique_lock<std::mutex> lock(mtx);
    // 队列满时阻塞生产者（背压机制）
    condProducer.wait(lock, [this]{ return queue.size() < capacity || isClosed; });
    if (isClosed) return;
    queue.push(item);
    condConsumer.notify_one();
}

bool Log::BlockQueue::Pop(std::string& item) {
    std::unique_lock<std::mutex> lock(mtx);
    condConsumer.wait(lock, [this]{ return !queue.empty() || isClosed; });
    if (isClosed && queue.empty()) return false;
    item = std::move(queue.front());
    queue.pop();
    condProducer.notify_one();
    return true;
}

void Log::BlockQueue::Close() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        isClosed = true;
    }
    condConsumer.notify_all();
    condProducer.notify_all();
}

void Log::BlockQueue::Flush() {
    condConsumer.notify_one();
}

bool Log::BlockQueue::Empty() {
    std::lock_guard<std::mutex> lock(mtx);
    return queue.empty();
}

// ==================== Log 实现 ====================

Log::Log() : path_(nullptr), suffix_(nullptr),
             level_(INFO), today_(0), isOpen_(false), isAsync_(false) {
}

Log::~Log() {
    // 关闭异步写线程
    if (writeThread_ && writeThread_->joinable()) {
        blockQueue_->Close();
        writeThread_->join();
    }
    std::lock_guard<std::mutex> lock(fileMtx_);
    if (fp_.is_open()) {
        fp_.flush();
        fp_.close();
    }
}

Log* Log::Instance() {
    static Log inst;
    return &inst;
}

void Log::Init(const char* path, const char* suffix,
               int maxQueueCapacity, Level level) {
    level_ = level;
    path_ = path;
    suffix_ = suffix;
    isOpen_ = true;

    // 创建日志目录
    mkdir(path, 0755);

    // 设置异步模式
    if (maxQueueCapacity > 0) {
        isAsync_ = true;
        if (!blockQueue_) {
            blockQueue_ = std::unique_ptr<BlockQueue>(new BlockQueue());
            blockQueue_->capacity = maxQueueCapacity;
            // 启动后台写线程
            writeThread_ = std::unique_ptr<std::thread>(
                new std::thread(&Log::AsyncWriteThread_, this));
        }
    } else {
        isAsync_ = false;
    }

    // 打开当天的日志文件
    CheckDateAndOpenFile_();

    // 写一条启动日志
    Write(INFO, "========== Log system initialized [%s] ==========",
          isAsync_ ? "ASYNC" : "SYNC");
}

void Log::CheckDateAndOpenFile_() {
    time_t now = time(nullptr);
    struct tm t;
    localtime_r(&now, &t);

    // 日期变了或文件未打开 → 新建日志文件
    if (today_ != t.tm_mday || !fp_.is_open()) {
        today_ = t.tm_mday;

        char filename[LOG_NAME_LEN] = {0};
        snprintf(filename, LOG_NAME_LEN - 1, "%s/%04d_%02d_%02d%s",
                 path_, t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, suffix_);

        std::lock_guard<std::mutex> lock(fileMtx_);
        if (fp_.is_open()) {
            fp_.flush();
            fp_.close();
        }
        fp_.open(filename, std::ios::app);
    }
}

void Log::AppendLogLevelTitle_(Level level, char* buf, int& idx) {
    switch (level) {
        case DEBUG: idx += snprintf(buf + idx, LOG_BUF_SIZE - idx, "[DEBUG] "); break;
        case INFO:  idx += snprintf(buf + idx, LOG_BUF_SIZE - idx, "[INFO]  "); break;
        case WARN:  idx += snprintf(buf + idx, LOG_BUF_SIZE - idx, "[WARN]  "); break;
        case ERROR: idx += snprintf(buf + idx, LOG_BUF_SIZE - idx, "[ERROR] "); break;
        default:    idx += snprintf(buf + idx, LOG_BUF_SIZE - idx, "[????]  "); break;
    }
}

void Log::Write(Level level, const char* format, ...) {
    // 时间戳
    struct timeval now;
    gettimeofday(&now, nullptr);
    struct tm t;
    localtime_r(&now.tv_sec, &t);

    // 检查日期是否变化
    if (today_ != t.tm_mday) {
        CheckDateAndOpenFile_();
    }

    // 格式化日志
    char buf[LOG_BUF_SIZE] = {0};
    int idx = 0;

    // 时间戳：2026-03-03 14:30:45.123456
    idx += snprintf(buf + idx, LOG_BUF_SIZE - idx,
                    "%04d-%02d-%02d %02d:%02d:%02d.%06ld ",
                    t.tm_year + 1900, t.tm_mon + 1, t.tm_mday,
                    t.tm_hour, t.tm_min, t.tm_sec, now.tv_usec);

    // 日志级别
    AppendLogLevelTitle_(level, buf, idx);

    // 用户消息
    va_list args;
    va_start(args, format);
    idx += vsnprintf(buf + idx, LOG_BUF_SIZE - idx, format, args);
    va_end(args);

    // 换行
    if (idx < LOG_BUF_SIZE - 1) {
        buf[idx++] = '\n';
        buf[idx] = '\0';
    }

    std::string logLine(buf, idx);

    if (isAsync_ && blockQueue_) {
        // 异步：入队，由后台线程写盘
        blockQueue_->Push(logLine);
    } else {
        // 同步：直接写盘
        std::lock_guard<std::mutex> lock(fileMtx_);
        if (fp_.is_open()) {
            fp_ << logLine;
            fp_.flush();
        }
    }
}

void Log::Flush() {
    if (isAsync_ && blockQueue_) {
        blockQueue_->Flush();
    }
    std::lock_guard<std::mutex> lock(fileMtx_);
    if (fp_.is_open()) {
        fp_.flush();
    }
}

Log::Level Log::GetLevel() const {
    return level_;
}

void Log::SetLevel(Level level) {
    level_ = level;
}

// 后台异步写线程：不断从队列取日志写入文件
void Log::AsyncWriteThread_() {
    std::string logLine;
    while (blockQueue_->Pop(logLine)) {
        std::lock_guard<std::mutex> lock(fileMtx_);
        if (fp_.is_open()) {
            fp_ << logLine;
        }
    }
    // 队列关闭后，刷盘
    std::lock_guard<std::mutex> lock(fileMtx_);
    if (fp_.is_open()) {
        fp_.flush();
    }
}
