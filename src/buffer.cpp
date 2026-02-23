#include "buffer.h"
using namespace std;

Buffer::Buffer(int initBufSize)
    : buffer_(initBufSize), readPos_(0), writePos_(0)
{
}
// 获取缓冲区起始位置的指针(// 返回 buffer 内部 vector 的起始指针 (这是一个辅助私有函数)
char * Buffer::BeginPtr_() {
    return &*buffer_.begin();
}

// const 版本，供只读操作使用
const char *Buffer::BeginPtr_() const {
    return &*buffer_.begin();
}

// 确保缓冲区有足够的空间存储 len 字节的数据(核心函数：当空间不够时，用来扩展空间或整理内存)
void Buffer::MakeSpace_(size_t len) {
    if(WriteAbleBytes() + PrePendBytes() < len) {
        buffer_.resize(writePos_ + len);
    } else {
       size_t readable = ReadAbleBytes();
       // 把 [readPos_, writePos_] 的数据拷贝到 [0, readable]
       copy(BeginPtr_() + readPos_, BeginPtr_() + writePos_, BeginPtr_());

        readPos_ = 0;
        writePos_ = readable + readPos_;
    }
}

// 返回可读字节数：writePos_ - readPos_
size_t Buffer::ReadAbleBytes() const
{
    return writePos_ - readPos_;
}

// 返回可写字节数：buffer总大小 - writePos_
size_t Buffer::WriteAbleBytes() const
{
    return buffer_.size() - writePos_;
}

// 返回头部预留字节数：readPos_ (这些是已处理的数据，可以回收)
size_t Buffer::PrePendBytes() const
{
    return readPos_;
}

// 返回当前读取位置的指针 (解析 HTTP 时非常需要这个)
const char *Buffer::Peek() const
{
    return BeginPtr_() + readPos_;
}

// 核心：读取数据后，移动 readPos_
void Buffer::Retrieve(size_t len)
{
    if(len < ReadAbleBytes())
        {readPos_ += len;// 还有剩余数据，只移动指针
        }
    else{
        RetrieveAll();// 全部数据读取完，重置指针
    }
    
}

// 读取直到某个指针位置 (用于解析 HTTP 行)
void Buffer::RetrieveUntil(const char *end)
{
    if(end >= Peek()){
        Retrieve(end - Peek());
    }

}

// 清空缓冲区 (重置 readPos_ = writePos_ = 0)
void Buffer::RetrieveAll() {
    //bzero(&buffer_[0], buffer_.size());
    readPos_ = 0;
    writePos_ = 0;
}

// 获取所有可读数据，并转换为字符串
string Buffer::RetrieveAllToStr() {
    string str(Peek(), ReadAbleBytes());
    RetrieveAll();
    return str;
}

// 核心：写入数据前，确保空间足够
void Buffer::EnsureWriteable(size_t len) {
    if(WriteAbleBytes() < len) {
        MakeSpace_(len);
    }
}

// 写入数据后，移动 writePos_
void Buffer::HasWritten(size_t len) {
    writePos_ += len ;
}

// 返回写入位置的指针
char *Buffer::BeginWrite() {
    return BeginPtr_() + writePos_;
}
const char *Buffer::BeginWriteConst() const {
    return BeginPtr_() + writePos_;
}

// 写入string
void Buffer::Append(const string& str) {
    Append(str.data(), str.size());
}

// 写入字符串
void Buffer::Append(const char *str, size_t len) {
    EnsureWriteable(len);
    copy(str ,str+len ,BeginWrite());
    HasWritten(len);
}

// 写入 void* (通用数据)
void Buffer::Append(const void *data, size_t len) {
    Append(static_cast<const char*>(data), len);
}

// 写入另一个 Buffer 的数据
void Buffer::Append(const Buffer &buff) {
    Append(buff.Peek(), buff.ReadAbleBytes());
}

// 从 fd 读取数据 (利用 readv)
ssize_t Buffer::ReadFd(int fd, int *Errno) {
    char buff[4096]; // 栈上的大缓冲区
    struct iovec iov[2];
    const size_t writeable = WriteAbleBytes();

    // 第一块：Buffer 内部剩余空间
    iov[0].iov_base = BeginPtr_() + writePos_;
    iov[0].iov_len = writeable;

    // 第二块：栈的临时空间
    iov[1].iov_base = buff ;
    iov[1].iov_len = sizeof(buff);

    //readv 自动按顺序填充这两块内存
    const ssize_t len = readv(fd, iov, 2);

    if(len < 0) {
        *Errno = errno;
    } else if(static_cast<size_t>(len) <= writeable) {
        // 数据少，Buffer 够用，直接移动指针
        writePos_ += len;
    } else {
        // 数据多，Buffer 不够用，先写满 Buffer ，再把多余的数据写入栈空间
        writePos_ = buffer_.size();// 先写满 buffer
        Append(buff, len - writeable); // 把多余的数据写入 buffer
    }
    return len;
}

// 向 fd 写入数据
ssize_t Buffer::WriteFd(int fd, int *Errno) {
    size_t readSize = ReadAbleBytes();
    ssize_t len = write(fd, Peek(), readSize);
    if(len < 0) {
        *Errno = errno;
        return len;
    } 
    readPos_ += len;
    return len;
}

