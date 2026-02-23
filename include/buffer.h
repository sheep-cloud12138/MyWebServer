#ifndef BUFFER_H
#define BUFFER_H

#include <vector>
#include <string>
#include <atomic>
#include <iostream>
#include <unistd.h>  // 包含 read, write 系统调用
#include <sys/uio.h> // 包含 readv (这是面试亮点)
using namespace std;
// ... 这里我们将写入类定义 ...

class Buffer
{
private:
    char *BeginPtr_(); // 获取缓冲区起始位置的指针(// 返回 buffer 内部 vector 的起始指针 (这是一个辅助私有函数)

    const char *BeginPtr_() const; // const 版本，供只读操作使用

    void MakeSpace_(size_t len); // 确保缓冲区有足够的空间存储 len 字节的数据(核心函数：当空间不够时，用来扩展空间或整理内存)

    // 成员变量
    vector<char> buffer_;// 用于存储数据的缓冲区
    atomic<size_t> readPos_;// 读指针，表示下一个可读字节的位置
    atomic<size_t> writePos_;// 写指针，表示下一个可写字节的位置
   
public:

    // 构造函数，允许指定初始缓冲区大小，默认为1024字节
    Buffer(int initBufSize = 1024);

    // 返回可读字节数：writePos_ - readPos_
    size_t ReadAbleBytes() const; 

    // 返回可写字节数：buffer总大小 - writePos_
    size_t WriteAbleBytes() const;

    // 返回头部预留字节数：readPos_ (这些是已处理的数据，可以回收)
    size_t PrePendBytes() const;

    // 返回当前读取位置的指针 (解析 HTTP 时非常需要这个)
    const char *Peek() const;

    // 核心：读取数据后，移动 readPos_
    void Retrieve(size_t len);

    // 读取直到某个指针位置 (用于解析 HTTP 行)
    void RetrieveUntil(const char *end);

    // 清空缓冲区 (重置 readPos_ = writePos_ = 0)
    void RetrieveAll();

    // 获取所有可读数据，并转换为字符串
    string RetrieveAllToStr();

    // 核心：写入数据前，确保空间足够
    void EnsureWriteable(size_t len);

    // 写入数据后，移动 writePos_
    void HasWritten(size_t len);

    // 返回写入位置的指针
    char *BeginWrite();
    const char *BeginWriteConst() const;

    //写入string
    void Append(const string &str);

    //写入字符串
    void Append(const char *str, size_t len);

    // 写入 void* (通用数据)
    void Append(const void *data, size_t len);

    // 写入另一个 Buffer 的数据
    void Append(const Buffer &buff);

    // 从 fd 读取数据 (利用 readv)
    ssize_t ReadFd(int fd, int *Errno);

    // 向 fd 写入数据
    ssize_t WriteFd(int fd, int *Errno);

    // 析构函数
    ~Buffer() = default;
};
#endif // BUFFER_H