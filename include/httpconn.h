#ifndef HTTP_CONN_H
#define HTTP_CONN_H

#include <sys/types.h>
#include <sys/uio.h>   // iovec, readv, writev
#include <sys/stat.h>  // stat (获取文件状态)
#include <arpa/inet.h> // sockaddr_in
#include <stdlib.h>
#include <errno.h>
#include <string>
#include <atomic>

#include "buffer.h"
#include "sqlconnpool.h"

// 每一个连上来的客户端，都会分配一个 HttpConn 对象
class HttpConn
{
private:
    int fd_;
    struct sockaddr_in addr_;
    bool isClose_;

    // 🌟 面试亮点：分散写 (Scatter/Gather IO)
    int iovCnt_;
    struct iovec iov_[2]; // iov_[0]存HTTP响应头，iov_[1]存具体的网页文件

    Buffer readBuff_;  // 读缓冲区
    Buffer writeBuff_; // 写缓冲区

    // -- HTTP 请求相关的状态变量 --
    bool isKeepAlive_;
    std::string method_;  // GET / POST
    std::string path_;    // 请求路径，如 /index.html
    std::string version_; // 协议版本，如 HTTP/1.1
    std::string body_;    // POST 的请求体

    // 内部处理函数：解析 HTTP 请求 (简易状态机)
    bool ParseRequestLine_(const std::string &line); // 解析请求行
    void ParseHeader_(const std::string &line);      // 解析请求头
    void ParseBody_(const std::string &line);        // 解析请求体

    // 内部处理函数：生成 HTTP 响应
    void MakeResponse_();

    // 🌟 面试亮点：零拷贝之 mmap 内存映射
    char *file_;           // 指向 mmap 映射到内存中的文件首地址
    struct stat fileStat_; // 请求文件的状态信息 (大小、是否存在等)
public:
    HttpConn() ;
    ~HttpConn() ;

    // 初始化连接 (记录 fd 和 客户端 IP 地址)
    void Init(int fd, const sockaddr_in &addr);
    // 关闭连接
    void Close();

    // 读接口：封装了 Buffer 的 ReadFd
    ssize_t Read(int *saveErrno);
    // 写接口：封装了 Buffer 的 WriteFd
    ssize_t Write(int *saveErrno);

    // 处理请求 (解析 HTTP 请求，生成响应)
    bool Process();

    // 一些get方法
    int GetFd() const;
    int GetPort() const;
    const char *GetIP() const;
    sockaddr_in GetAddr() const;

    //// 还需要写多少字节的数据
    int ToWriteBytes() { return iov_[0].iov_len + iov_[1].iov_len; }
    //是否保持长连接
    bool IsKeepAlive() const { return isKeepAlive_; }

    //静态全局变量
    static std::atomic<int> userCount_; // 统计当前在线用户数量
    static const char *srcDir_;          // 网站根目录
    static bool isET;                 // 是否是边缘触发模式
};
#endif // HTTP_CONN_H