#include "httpconn.h"
#include "log.h"
#include <sys/mman.h>
#include <iostream>
#include <cstring>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <fcntl.h>   // open
#include <unistd.h>  // close
#include <sys/stat.h>
#include <sys/mman.h>
#include "ai_engine.h" // 【新增】引入大脑

using namespace std;

const char *HttpConn::srcDir_ = nullptr;
std::atomic<int> HttpConn::userCount_(0);
bool HttpConn::isET = true;

HttpConn::HttpConn()
{
    fd_ = -1;
    addr_ = {0};
    isClose_ = true;
    file_ = nullptr;
    iovCnt_ = 0;
    contentLength_ = 0;
    memset(iov_, 0, sizeof(iov_));
}

void HttpConn::ResetRequestState_()
{
    isKeepAlive_ = false;
    method_.clear();
    path_.clear();
    version_.clear();
    body_.clear();
    contentLength_ = 0;
}

// 初始化连接 (记录 fd 和 客户端 IP 地址)
void HttpConn::Init(int fd, const sockaddr_in &addr)
{
    assert(fd > 0);
    userCount_++;
    fd_ = fd;
    addr_ = addr;
    writeBuff_.RetrieveAll(); // 清空写缓冲区
    readBuff_.RetrieveAll();  // 清空读缓冲区
    isClose_ = false;

    // 初始化 HTTP 状态变量
    ResetRequestState_();
    file_ = nullptr;
    fileStat_ = {0};
}
// 关闭连接：非常重要，要清理 Socket 和 mmap 内存
void HttpConn::Close()
{
    // 释放 mmap 映射的内存（零拷贝的清理工作）
    if (file_)
    {
        munmap(file_, fileStat_.st_size);
        file_ = nullptr;
    }
    // 关闭 Socket
    if (!isClose_)
    {
        isClose_ = true;
        close(fd_);
        fd_ = -1;  // 标记已关闭，防止双重 close
        userCount_--;
    }
}

// 读接口：封装了 Buffer 的 ReadFd
ssize_t HttpConn::Read(int *saveErrno)
{
    ssize_t bytes_read = 0; // 【修正：用于累加真正读到的字节数】
    while (true)
    {
        ssize_t len = readBuff_.ReadFd(fd_, saveErrno);
        if (len <= 0)
        {
            if (*saveErrno == EAGAIN || *saveErrno == EWOULDBLOCK)
            {
                break; // 读干净了，跳出循环
            }
            if (*saveErrno == EINTR)
            {
                continue; // 被信号打断，重试
            }
            if (bytes_read == 0) return -1; // 真的出错了且一个字节都没读到
            break;
        }
        bytes_read += len; // 累加读到的数据
        if (!isET) break;  // 如果不是 ET 模式，读一次就走
    }
    return bytes_read; // 返回总共读到的字节数
}

// 写接口：封装了 Buffer 的 WriteFd
ssize_t HttpConn::Write(int *saveErrno)
{
    ssize_t bytes_write = 0; // 【修正：用于累加真正写出的字节数】
    while (true)
    {
        ssize_t len = writev(fd_, iov_, iovCnt_);
        if (len < 0)
        {
            *saveErrno = errno;
            if (*saveErrno == EAGAIN || *saveErrno == EWOULDBLOCK)
            {
                break; // 缓冲区满了，等下次 EPOLLOUT
            }
            if (*saveErrno == EINTR)
            {
                continue;
            }
            if (bytes_write == 0) return -1;
            break;
        }
        if (len == 0) break;
        
        bytes_write += len; // 累加发送的字节数

        // 调整 iov_ 指针
        if (static_cast<size_t>(len) > iov_[0].iov_len)
        {
            iov_[1].iov_base = (uint8_t *)iov_[1].iov_base + (len - iov_[0].iov_len);
            iov_[1].iov_len -= (len - iov_[0].iov_len);
            if (iov_[0].iov_len)
            {
                writeBuff_.RetrieveAll();
                iov_[0].iov_len = 0;
            }
        }
        else
        {
            iov_[0].iov_base = (uint8_t *)iov_[0].iov_base + len;
            iov_[0].iov_len -= len;
            writeBuff_.Retrieve(len);
        }

        if (ToWriteBytes() == 0)
        {
            if (file_)
            {
                munmap(file_, fileStat_.st_size);
                file_ = nullptr;
            }
            break;
        }

        if (!isET && ToWriteBytes() < 10240) break; // 退出条件
    }
    return bytes_write; // 返回总共发出的字节数
}

// 处理请求 (解析 HTTP 请求，生成响应)
bool HttpConn::Process() {
    if (readBuff_.ReadAbleBytes() <= 0) {
        return false;
    }

    std::string requestData(readBuff_.Peek(), readBuff_.ReadAbleBytes());
    if (requestData.size() > MAX_REQUEST_HEADER_SIZE + MAX_REQUEST_BODY_SIZE) {
        writeBuff_.Append("HTTP/1.1 413 Payload Too Large\r\nContent-Length: 0\r\n\r\n");
        iov_[0].iov_base = const_cast<char*>(writeBuff_.Peek());
        iov_[0].iov_len = writeBuff_.ReadAbleBytes();
        iovCnt_ = 1;
        readBuff_.RetrieveAll();
        return true;
    }

    const std::string headerEndMark = "\r\n\r\n";
    size_t headerEnd = requestData.find(headerEndMark);
    if (headerEnd == std::string::npos) {
        return false;
    }
    if (headerEnd + headerEndMark.size() > MAX_REQUEST_HEADER_SIZE) {
        writeBuff_.Append("HTTP/1.1 431 Request Header Fields Too Large\r\nContent-Length: 0\r\n\r\n");
        iov_[0].iov_base = const_cast<char*>(writeBuff_.Peek());
        iov_[0].iov_len = writeBuff_.ReadAbleBytes();
        iovCnt_ = 1;
        readBuff_.RetrieveAll();
        return true;
    }

    ResetRequestState_();

    // 1) 解析请求行
    size_t lineEnd = requestData.find("\r\n");
    if (lineEnd == std::string::npos || lineEnd > headerEnd) {
        return false;
    }
    std::string requestLine = requestData.substr(0, lineEnd);
    if (!ParseRequestLine_(requestLine)) {
        return false;
    }

    // 2) 解析请求头
    size_t headerStart = lineEnd + 2;
    while (headerStart < headerEnd) {
        size_t curLineEnd = requestData.find("\r\n", headerStart);
        if (curLineEnd == std::string::npos || curLineEnd > headerEnd) {
            return false;
        }
        std::string headerLine = requestData.substr(headerStart, curLineEnd - headerStart);
        if (!ParseHeader_(headerLine)) {
            writeBuff_.Append("HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            iov_[0].iov_base = const_cast<char*>(writeBuff_.Peek());
            iov_[0].iov_len = writeBuff_.ReadAbleBytes();
            iovCnt_ = 1;
            isKeepAlive_ = false;
            readBuff_.RetrieveAll();
            return true;
        }
        headerStart = curLineEnd + 2;
    }

    // 3) 请求体按 Content-Length 读取，支持半包/粘包
    size_t bodyStart = headerEnd + headerEndMark.size();
    if (contentLength_ > MAX_REQUEST_BODY_SIZE) {
        writeBuff_.Append("HTTP/1.1 413 Payload Too Large\r\nContent-Length: 0\r\n\r\n");
        iov_[0].iov_base = const_cast<char*>(writeBuff_.Peek());
        iov_[0].iov_len = writeBuff_.ReadAbleBytes();
        iovCnt_ = 1;
        isKeepAlive_ = false;
        readBuff_.RetrieveAll();
        return true;
    }
    if (contentLength_ > (static_cast<size_t>(-1) - bodyStart)) {
        writeBuff_.Append("HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        iov_[0].iov_base = const_cast<char*>(writeBuff_.Peek());
        iov_[0].iov_len = writeBuff_.ReadAbleBytes();
        iovCnt_ = 1;
        isKeepAlive_ = false;
        readBuff_.RetrieveAll();
        return true;
    }
    size_t totalNeed = bodyStart + contentLength_;
    if (requestData.size() < totalNeed) {
        return false;
    }
    if (contentLength_ > 0) {
        if (!ParseBody_(requestData.substr(bodyStart, contentLength_))) {
            iov_[0].iov_base = const_cast<char*>(writeBuff_.Peek());
            iov_[0].iov_len = writeBuff_.ReadAbleBytes();
            iovCnt_ = 1;
            isKeepAlive_ = false;
            readBuff_.Retrieve(totalNeed);
            return true;
        }
    }

    // 消费当前请求，剩余数据留给下一次处理（支持 keep-alive 连续请求）
    readBuff_.Retrieve(totalNeed);

    writeBuff_.RetrieveAll();
    if (file_) {
        munmap(file_, fileStat_.st_size);
        file_ = nullptr;
    }
    
    // 🌟【新增】AI 智能接口拦截逻辑
    // ==========================================================
    if (method_ == "POST" && path_ == "/api/predict") {
        // 1. 解析用户输入的数字 (这里假设 Body 里就是一个纯数字字符串)
        float inputVal = 0.0f;
        const char* begin = body_.c_str();
        char* end = nullptr;
        inputVal = std::strtof(begin, &end);
        while (end && *end && std::isspace(static_cast<unsigned char>(*end))) {
            ++end;
        }
        if (end == begin || (end && *end != '\0')) {
            std::string badReq = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n";
            badReq += isKeepAlive_ ? "Connection: keep-alive\r\n\r\n" : "Connection: close\r\n\r\n";
            writeBuff_.Append(badReq);
            iov_[0].iov_base = const_cast<char*>(writeBuff_.Peek());
            iov_[0].iov_len = writeBuff_.ReadAbleBytes();
            iovCnt_ = 1;
            return true;
        }

        // 2. 调用 AI 引擎进行推理
        std::vector<float> inputVec = { inputVal };
        std::vector<float> outputVec = AIEngine::Instance()->Predict(inputVec);
        
        // 3. 构造响应内容 (这里为了简单，直接返回计算结果的字符串)
        std::string responseBody = "Result: " + std::to_string(outputVec.empty() ? 0.0f : outputVec[0]);

        // 4. 组装 HTTP 响应报文
        writeBuff_.Append("HTTP/1.1 200 OK\r\n");
        writeBuff_.Append("Content-Type: text/plain\r\n");
        writeBuff_.Append("Content-Length: " + std::to_string(responseBody.size()) + "\r\n");
        if (isKeepAlive_) {
            writeBuff_.Append("Connection: keep-alive\r\n\r\n");
        } else {
            writeBuff_.Append("Connection: close\r\n\r\n");
        }
        writeBuff_.Append(responseBody);

        // 5. 设置 writev 的指针
        iov_[0].iov_base = const_cast<char*>(writeBuff_.Peek());
        iov_[0].iov_len = writeBuff_.ReadAbleBytes();
        iovCnt_ = 1; // 只需要发送 writeBuff_ 里的数据，没有文件映射
        
        return true; // 处理完毕，直接返回，不走后面的静态文件逻辑
    }
    // ==========================================================
    // 3. 根据解析结果生成 HTTP 响应 (设置 iov_ 指向响应头和文件内容)
    MakeResponse_();
    return true;
}
bool HttpConn::ParseRequestLine_(const std::string& line){
    size_t firstSpace = line.find(' ');
    if (firstSpace == std::string::npos) {
        return false;
    }
    size_t secondSpace = line.find(' ', firstSpace + 1);
    if (secondSpace == std::string::npos) {
        return false;
    }

    method_ = line.substr(0, firstSpace);
    path_ = line.substr(firstSpace + 1, secondSpace - firstSpace - 1);
    const std::string prefix = "HTTP/";
    if (line.compare(secondSpace + 1, prefix.size(), prefix) != 0) {
        return false;
    }
    version_ = line.substr(secondSpace + 1 + prefix.size());
    isKeepAlive_ = (version_ == "1.1");
    if (path_ == "/") {
        path_ = "/index.html";
    }
    return !method_.empty() && !path_.empty() && !version_.empty();
}

bool HttpConn::ParseHeader_(const std::string& line){
    size_t colonPos = line.find(':');
    if (colonPos == std::string::npos) {
        return true;
    }
    std::string headerName = line.substr(0, colonPos);
    size_t valueStart = colonPos + 1;
    while (valueStart < line.size() && std::isspace(static_cast<unsigned char>(line[valueStart]))) {
        ++valueStart;
    }
    std::string headerValue = line.substr(valueStart);
    std::string headerNameLower = headerName;
    std::string headerValueLower = headerValue;
    for (char& ch : headerNameLower) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    for (char& ch : headerValueLower) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

    if (headerNameLower == "connection") {
        if (headerValueLower == "close") {
            isKeepAlive_ = false;
        } else if (headerValueLower == "keep-alive") {
            isKeepAlive_ = true;
        }
    } else if (headerNameLower == "content-length") {
        if (headerValueLower.empty()) {
            return false;
        }
        for (char ch : headerValueLower) {
            if (!std::isdigit(static_cast<unsigned char>(ch))) {
                return false;
            }
        }
        errno = 0;
        unsigned long long parsed = std::strtoull(headerValueLower.c_str(), nullptr, 10);
        if (errno == ERANGE) {
            return false;
        }
        if (parsed > static_cast<unsigned long long>(MAX_REQUEST_BODY_SIZE)) {
            contentLength_ = MAX_REQUEST_BODY_SIZE + 1;
            return true;
        }
        contentLength_ = static_cast<size_t>(parsed);
    }
    return true;
}
// 解析请求体：如果是 POST 登录，这里会调用 SqlConnPool 查数据库
bool HttpConn::ParseBody_(const std::string& line)
{
    // 【新增】请求体大小检查（防止 OOM 攻击）
    if(line.size() > MAX_REQUEST_BODY_SIZE) {
        LOG_WARN("Request body too large: %zu bytes", line.size());
        writeBuff_.Append("HTTP/1.1 413 Payload Too Large\r\n");
        writeBuff_.Append("Content-Length: 0\r\nConnection: close\r\n\r\n");
        return false;
    }
    
    body_ = line; // 【修正】先保存请求体到成员变量

    // 假设这是一个登录请求，路径是 /login
    if(method_ == "POST" && path_ == "/login")
    {
        // 面试亮点：从线程池里取出一个 Worker 线程正在执行这行代码
        // 我们利用 RAII 自动从连接池拿一个 MySQL 连接
        MYSQL* sql;
        SqlConnRAII(&sql, SqlConnPool::Instance());

        // （此处省略具体的 SQL 账号密码校验逻辑，为了保持代码精简）
        // 真实业务中，会解析 line (如 user=admin&pwd=123)，然后查库
        LOG_DEBUG("[DB] Executing Login check using pooled connection.");
    }
    return true;
}

// 阶段三：生成 HTTP 响应 (零拷贝核心)
// ==========================================
void HttpConn::MakeResponse_()
{
    // 1. 拼接目标文件的绝对路径
    // srcDir_ 是在 WebServer 启动时设置的，比如 /var/www/html
    std::string targetPath = std::string(srcDir_) + path_;
    
    // 2. 检查文件是否存在且可读 (stat 系统调用)
    if(stat(targetPath.c_str(), &fileStat_) < 0 || S_ISDIR(fileStat_.st_mode)) {
        // 文件不存在或是目录，返回 404
        writeBuff_.Append("HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
        // 【修正：统一使用 const_cast 和 Peek()，保持代码风格一致，避免去除 const 带来的隐患】
        iov_[0].iov_base = const_cast<char*>(writeBuff_.Peek());
        iov_[0].iov_len = writeBuff_.ReadAbleBytes();
        iovCnt_ = 1;
        return;
    }

    // 3. 文件存在，生成 200 OK 的 HTTP 响应头
    writeBuff_.Append("HTTP/1.1 200 OK\r\n");
    if(isKeepAlive_) {
        writeBuff_.Append("Connection: keep-alive\r\n");
    } else {
        writeBuff_.Append("Connection: close\r\n");
    }
    writeBuff_.Append("Content-Length: " + std::to_string(fileStat_.st_size) + "\r\n\r\n");

    // 4. 使用 mmap 将文件映射到内存，file_ 指向文件内容的首地址
    int srcFd = open(targetPath.c_str(), O_RDONLY);
    if(srcFd < 0) {
        // 打开文件失败
        writeBuff_.RetrieveAll();
        writeBuff_.Append("HTTP/1.1 403 Forbidden\r\n\r\n");
        iov_[0].iov_base = const_cast<char*>(writeBuff_.Peek());
        iov_[0].iov_len = writeBuff_.ReadAbleBytes();
        iovCnt_ = 1;
        return;
    }

    // MAP_PRIVATE 表示内存映射私有，不影响原文件
    // PROT_READ 表示只读
    file_ = static_cast<char*>(mmap(nullptr, fileStat_.st_size, PROT_READ, MAP_PRIVATE, srcFd, 0));
    close(srcFd); // 映射后就可以关闭文件描述符了

    // 5. 设置 iovec 分散写数组
    // 第一块：HTTP 响应头 (存放在写缓冲区)
    iov_[0].iov_base = const_cast<char*>(writeBuff_.Peek());
    iov_[0].iov_len = writeBuff_.ReadAbleBytes();   

    // 第二块：文件内容 (存放在 mmap 映射的内存中)
    iov_[1].iov_base = file_;
    iov_[1].iov_len = fileStat_.st_size;

    iovCnt_ = 2; // 需要发送两块数据
}
// 一些get方法
int HttpConn::GetFd() const { return fd_; }
int HttpConn::GetPort() const { return ntohs(addr_.sin_port); }
const char *HttpConn::GetIP() const { return inet_ntoa(addr_.sin_addr); }
sockaddr_in HttpConn::GetAddr() const { return addr_; }

HttpConn::~HttpConn()
{
    Close();
}