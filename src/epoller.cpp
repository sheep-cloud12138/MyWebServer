#include "epoller.h"

// 构造函数：maxEvent是 epoll_wait 返回的最大事件数
 Epoller::Epoller(int maxEvent):epollFd_(epoll_create(512)), events_(maxEvent) {
    assert(epollFd_ >= 0 && events_.size() > 0);
}

// 添加 fd 到 epoll 实例，默认水平触发
bool Epoller::AddFd(int fd, uint32_t events) {
    if(fd<0)return false;
    epoll_event ev={0} ;   
    ev.data.fd = fd;
    ev.events = events;
    return epoll_ctl(epollFd_,EPOLL_CTL_ADD,fd,&ev)==0;  
}

// 修改 fd 监控的事件
bool Epoller::ModFd(int fd, uint32_t events) {
    if(fd<0)return false;
    epoll_event ev={0} ;   
    ev.data.fd = fd;
    ev.events = events;
    return epoll_ctl(epollFd_,EPOLL_CTL_MOD,fd,&ev)==0;
}

// 删除 fd 从 epoll 实例
bool Epoller::DelFd(int fd) {
    if(fd<0)return false;
    epoll_event ev={0} ;
    return epoll_ctl(epollFd_,EPOLL_CTL_DEL,fd,nullptr)==0;
}

// 等待事件发生，返回就绪事件数
// 返回事件数量，结果存放在 events_ 数组里
int Epoller::Wait(int timeoutMs) {
    // &*events_.begin() 获取 vector 底层数组的首地址
    // 这里的 events_.size() 只是告诉内核我也能接收这么多事件
    return epoll_wait(epollFd_, &*events_.begin(), static_cast<int>(events_.size()), timeoutMs);
}

// 获取第i个就绪事件的 fd
int Epoller::GetEventFd(size_t i) const {
    assert(i < events_.size() && i >= 0);
    return events_[i].data.fd;
}

// 获取第i个就绪事件的事件类型
uint32_t Epoller::GetEvents(size_t i) const {
    assert(i < events_.size() && i >= 0);
    return events_[i].events;
}

// 析构函数：关闭 epoll fd
Epoller::~Epoller() {
    close(epollFd_);
}