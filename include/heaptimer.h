#ifndef HEAP_TIMER_H
#define HEAP_TIMER_H

#include <queue>
#include <unordered_map>
#include <time.h>
#include <algorithm>
#include <arpa/inet.h> 
#include <functional> 
#include <assert.h> 
#include <chrono>
#include <mutex>

typedef std::function<void()> TimeoutCallBack;
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds MS;
typedef Clock::time_point TimeStamp;

// 定时器节点
struct TimerNode {
    int id;             // 用来标记定时器，通常是客户端的 fd
    TimeStamp expires;  // 具体的过期绝对时间
    TimeoutCallBack cb; // 超时后的回调函数 (比如关闭该 fd)
    
    // 重载比较运算符，用于堆的排序
    bool operator<(const TimerNode& t) {
        return expires < t.expires;
    }
};

class HeapTimer {
public:
    HeapTimer() { heap_.reserve(64); }
    ~HeapTimer() { clear(); }
    
    // 调整指定 id 的定时器 (客户端发来心跳，延长超时时间)
    void adjust(int id, int timeout);
    
    // 添加一个新的定时器
    void add(int id, int timeout, const TimeoutCallBack& cb);
    
    // 提前触发并删除指定的定时器
    void doWork(int id);
    
    // 只删除定时器，不执行回调（用于主动关闭连接时避免递归）
    void cancel(int id);
    
    // 清除所有定时器
    void clear();
    
    // 核心逻辑：清除超时的节点，并执行回调函数
    void tick();
    
    void pop();
    
    // 获取下一个定时器的超时时间 (给 epoll_wait 用)
    int GetNextTick();

private:
    void del_(size_t i);
    void siftup_(size_t i);
    bool siftdown_(size_t index, size_t n);
    void SwapNode_(size_t i, size_t j);

    std::vector<TimerNode> heap_;             // 底层用 vector 实现完全二叉树
    std::unordered_map<int, size_t> ref_;     // 映射：fd -> 在 heap_ 中的索引 (用于 O(1) 查找)
    std::mutex mtx_;                          // 保护堆和映射的线程安全锁
};

#endif // HEAP_TIMER_H