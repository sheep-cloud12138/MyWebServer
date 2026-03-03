#include "heaptimer.h"

// 向上调整堆 (当新加入节点或者时间提早时)
void HeapTimer::siftup_(size_t i) {
    if(i >= heap_.size()) return;
    size_t j = (i - 1) / 2;
    while(j >= 0) {
        if(heap_[j] < heap_[i]) { break; }
        SwapNode_(i, j);
        i = j;
        j = (i - 1) / 2;
    }
}

// 交换堆中两个节点
void HeapTimer::SwapNode_(size_t i, size_t j) {
    if(i >= heap_.size() || j >= heap_.size()) return;
    std::swap(heap_[i], heap_[j]);
    ref_[heap_[i].id] = i;
    ref_[heap_[j].id] = j;
}

bool HeapTimer::siftdown_(size_t index, size_t n) {
    if(index >= heap_.size() || n > heap_.size()) return false;
    size_t i = index;
    size_t j = i * 2 + 1;
    while(j < n) {
        if(j + 1 < n && heap_[j + 1] < heap_[j]) j++;
        if(heap_[i] < heap_[j]) break;
        SwapNode_(i, j);
        i = j;
        j = i * 2 + 1;
    }
    return i > index;
}

// 新增定时器：如果 fd 已经存在就更新，不存在就放到堆底然后向上调整
void HeapTimer::add(int id, int timeout, const TimeoutCallBack& cb) {
    std::lock_guard<std::mutex> lock(mtx_);
    if(id < 0) return;
    size_t i;
    if(ref_.count(id) == 0) {
        i = heap_.size();
        ref_[id] = i;
        heap_.push_back({id, Clock::now() + MS(timeout), cb});
        siftup_(i);
    } 
    else {
        i = ref_[id];
        heap_[i].expires = Clock::now() + MS(timeout);
        heap_[i].cb = cb;
        if(!siftdown_(i, heap_.size())) {
            siftup_(i);
        }
    }
}

// 客户端活跃，延长超时时间
void HeapTimer::adjust(int id, int timeout) {
    std::lock_guard<std::mutex> lock(mtx_);
    if(heap_.empty() || ref_.count(id) == 0) return;
    heap_[ref_[id]].expires = Clock::now() + MS(timeout);
    siftdown_(ref_[id], heap_.size());
}

// 删除指定位置的节点（内部方法，无锁，由调用方持锁）
void HeapTimer::del_(size_t i) {
    if(heap_.empty() || i >= heap_.size()) return;
    size_t n = heap_.size() - 1;
    if(i < n) {
        SwapNode_(i, n);
        if(!siftdown_(i, n)) {
            siftup_(i);
        }
    }
    ref_.erase(heap_.back().id);
    heap_.pop_back();
}

// 主动触发某个定时器
void HeapTimer::doWork(int id) {
    std::unique_lock<std::mutex> lock(mtx_);
    if(heap_.empty() || ref_.count(id) == 0) return;
    size_t i = ref_[id];
    TimerNode node = heap_[i];
    del_(i);
    lock.unlock();
    node.cb();
}

// 只删除定时器，不执行回调
void HeapTimer::cancel(int id) {
    std::lock_guard<std::mutex> lock(mtx_);
    if(heap_.empty() || ref_.count(id) == 0) return;
    del_(ref_[id]);
}

// 清理所有超时的定时器
void HeapTimer::tick() {
    std::unique_lock<std::mutex> lock(mtx_);
    while(!heap_.empty()) {
        TimerNode node = heap_.front();
        if(std::chrono::duration_cast<MS>(node.expires - Clock::now()).count() > 0) { 
            break; 
        }
        del_(0);
        lock.unlock();
        node.cb();
        lock.lock();
    }
}

// 弹出堆顶
void HeapTimer::pop() {
    std::lock_guard<std::mutex> lock(mtx_);
    if(!heap_.empty()) {
        del_(0);
    }
}

void HeapTimer::clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    ref_.clear();
    heap_.clear();
}

// 给 Epoll_wait 用的：看看还需要等多久才有连接超时
int HeapTimer::GetNextTick() {
    tick();
    std::lock_guard<std::mutex> lock(mtx_);
    size_t res = -1;
    if(!heap_.empty()) {
        res = std::chrono::duration_cast<MS>(heap_.front().expires - Clock::now()).count();
        if(res < 0) { res = 0; }
    }
    return res;
}
