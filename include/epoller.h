#ifndef EPOLLER_H
#define EPOLLER_H

#include <sys/epoll.h> // epoll_event
#include <fcntl.h>     // fcntl
#include <unistd.h>    // close
#include <assert.h>
#include <vector>
#include <errno.h>

class Epoller
{
private:
    int epollFd_;                        // epoll 实例的文件描述符
    std::vector<struct epoll_event> events_; // 用于存储就绪事件的数组
public:
    // 构造函数：maxEvent是 epoll_wait 返回的最大事件数
    explicit Epoller(int maxEvent = 1024);
    
    // 添加 fd 到 epoll 实例，默认水平触发
    bool AddFd(int fd, uint32_t events  );

    // 修改 fd 监控的事件
    bool ModFd (int fd, uint32_t events );

    // 删除 fd 从 epoll 实例
    bool DelFd (int fd );

    // 等待事件发生，返回就绪事件数
    // 返回事件数量，结果存放在 events_ 数组里
    int Wait(int timeoutMs = -1);

    // 获取第i个就绪事件的 fd
    int GetEventFd (size_t i ) const;
    
    // 获取第i个就绪事件的事件类型
    uint32_t GetEvents (size_t i ) const;

    //析构函数：关闭 epoll fd
    ~Epoller();
};
#endif // EPOLLER_H