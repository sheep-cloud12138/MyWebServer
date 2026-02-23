#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <functional>
#include <vector>
#include <memory> // shared_ptr 需要
#include <cassert> // assert 需要

class ThreadPool {
public:
    // 构造函数：默认创建 8 个线程
    // 删除了 ThreadPool() = default 以避免歧义
    explicit ThreadPool(size_t threadCount = 8) : pool_(std::make_shared<Pool>()) {
        assert(threadCount > 0);

        // 创建 threadCount 个线程
        for(size_t i = 0; i < threadCount; i++) {
            std::thread([pool = pool_] {
                std::unique_lock<std::mutex> locker(pool->mtx);
                while(true) {
                    if(!pool->tasks.empty()) {
                        // 1. 取出任务
                        auto task = std::move(pool->tasks.front());
                        pool->tasks.pop();
                        
                        // 2. 核心：任务执行时解锁，允许其他线程并发取任务
                        locker.unlock();
                        
                        task(); // 执行任务
                        
                        locker.lock(); // 重新加锁准备取下一个
                    } 
                    else if(pool->isClosed) {
                        break; // 队列空了且已关闭，退出线程
                    } 
                    else {
                        pool->cond.wait(locker); // 等待唤醒
                    }
                }
            }).detach(); // 线程分离
        }
    }

    // 移动构造函数
    ThreadPool(ThreadPool&&) = default;
    
    // 析构函数
    ~ThreadPool() {
        if(static_cast<bool>(pool_)) {
            {
                std::lock_guard<std::mutex> locker(pool_->mtx);
                pool_->isClosed = true;
            }
            pool_->cond.notify_all(); // 唤醒所有线程让它们退出
        }
    }

    // 添加任务
    template<class F>
    void AddTask(F&& task) {
        {
            std::lock_guard<std::mutex> locker(pool_->mtx);
            pool_->tasks.emplace(std::forward<F>(task));
        }
        pool_->cond.notify_one();
    }

private:
    struct Pool {
        std::mutex mtx;
        std::condition_variable cond;
        bool isClosed = false; // 默认初始化为 false
        std::queue<std::function<void()>> tasks;
    };
    std::shared_ptr<Pool> pool_;
};

#endif // THREADPOOL_H