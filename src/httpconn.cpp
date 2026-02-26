#include "httpconn.h"
#include <sys/mman.h>
#include <iostream>
#include <cstring>
#include <cassert>
#include <regex>
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
    memset(iov_, 0, sizeof(iov_));
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
    isKeepAlive_ = false;
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
        close(fd_);
        isClose_ = true;
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

        if (iov_[0].iov_len + iov_[1].iov_len == 0) { break; } // 数据全发完了

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

        if (!isET && ToWriteBytes() < 10240) break; // 退出条件
    }
    return bytes_write; // 返回总共发出的字节数
}

// 处理请求 (解析 HTTP 请求，生成响应)
bool HttpConn::Process() {
    // 1. 从 readBuff_ 中取出所有收到的数据
    if(readBuff_.ReadAbleBytes() <= 0) {
        return false;
    }
    std::string requestData = readBuff_.RetrieveAllToStr();

    // 2. 按行解析 (以 \r\n 为分隔符)
    size_t lineEnd = requestData.find("\r\n");
    if(lineEnd == std::string::npos) {
        return false; // 请求行不完整
    }

    // 2.1 解析请求行 (如: GET /index.html HTTP/1.1)
    std::string requestLine = requestData.substr(0, lineEnd);
    if(!ParseRequestLine_(requestLine)) {
        return false; // 请求行解析失败 
    }

    // 2.2 解析请求头 (循环查找 \r\n)
    size_t headerStart = lineEnd + 2; // 跳过 \r\n
    // 【修正：增加括号，保证先赋值再比较】
    while((lineEnd = requestData.find("\r\n", headerStart)) != std::string::npos) {
        std::string headerLine = requestData.substr(headerStart, lineEnd - headerStart);
        if(headerLine.empty()) {
            // 遇到空行，说明请求头结束了，后面紧跟着的是 Body
            headerStart = lineEnd + 2; 
            break;
        }
        ParseHeader_(headerLine);
        headerStart = lineEnd + 2; // 继续查找下一行
    }

    // 2.3 解析请求体 (比如 POST 提交的表单数据)
    // 注意：这段代码必须在 while 循环的外面！
    if(headerStart < requestData.size()) {
        std::string body = requestData.substr(headerStart);
        ParseBody_(body);
    }
    
    // 🌟【新增】AI 智能接口拦截逻辑
    // ==========================================================
    if (method_ == "POST" && path_ == "/api/predict") {
        // 1. 解析用户输入的数字 (这里假设 Body 里就是一个纯数字字符串)
        float inputVal = 0.0f;
        try {
            inputVal = std::stof(body_); // string -> float
        } catch (...) {
            inputVal = 0.0f; // 解析失败给个默认值
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
        writeBuff_.Append("Connection: keep-alive\r\n\r\n"); // 保持长连接
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
// 解析请求行：使用正则表达式提取 Method, Path, Version
bool HttpConn::ParseRequestLine_(const std::string& line){
    std::regex patten("^([^ ]*) ([^ ]*) HTTP/([^ ]*)$");
    std::smatch subMatch;
    if(std::regex_match(line, subMatch, patten)) {
        method_ = subMatch[1];
        path_ = subMatch[2];
        version_ = subMatch[3];
        if(path_ == "/") {
            path_ = "/index.html";
        }
        return true;
    }
    return false;
}

// 解析请求头：重点关注 Connection 字段，判断是否是长连接
void HttpConn::ParseHeader_(const std::string& line){
    std::regex patten("^([^:]*): ?(.*)$");
    std::smatch subMatch;
    if(std::regex_match(line, subMatch, patten)) {
        std::string headerName = subMatch[1];
        std::string headerValue = subMatch[2];
        if(headerName == "Connection" && headerValue == "keep-alive") {
            isKeepAlive_ = true;
        }
    }
        
}
// 解析请求体：如果是 POST 登录，这里会调用 SqlConnPool 查数据库
void HttpConn::ParseBody_(const std::string& line)
{
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
        std::cout << "  [DB] Executing Login check using pooled connection." << std::endl;
    }
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