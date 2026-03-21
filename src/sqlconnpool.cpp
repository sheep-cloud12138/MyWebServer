#include "sqlconnpool.h"
#include "log.h"

using namespace std;

SqlConnPool::SqlConnPool()
{
    useCount_ = 0;
    freeCount_ = 0;
    MAX_CONN_ = 0;
    isInitialized_ = false;
    semInitialized_ = false;
    poolDestroyed_ = true;
} // 构造函数私有化，防止外部创建

SqlConnPool *SqlConnPool::Instance()
{
    static SqlConnPool connPool;
    return &connPool;
} // 单例模式，获取唯一实例

//// 初始化：主机、端口、用户名、密码、库名、池大小
bool SqlConnPool::Init(const char *host, int port,
                       const char *user, const char *pwd,
                       const char *dbName, int connSize)
{
    assert(connSize > 0);
    DestroyPool();

    queue<MYSQL*> newQueue;
    // 循环创建连接
    for (int i = 0; i < connSize; i++)
    {
        MYSQL *sql = nullptr;
        sql = mysql_init(sql);
        if (!sql)
        {
            LOG_ERROR("MySQL init failed at connection %d", i);
            while (!newQueue.empty()) {
                mysql_close(newQueue.front());
                newQueue.pop();
            }
            return false;
        }
        sql = mysql_real_connect(sql, host, user, pwd, dbName, port, nullptr, 0);
        if (!sql)
        {
            LOG_ERROR("MySQL connect failed at connection %d", i);
            mysql_close(sql);
            while (!newQueue.empty()) {
                mysql_close(newQueue.front());
                newQueue.pop();
            }
            return false;
        }
        newQueue.push(sql);
    }

    {
        lock_guard<mutex> locker(mtx_);
        connQue_ = std::move(newQueue);
    }

    MAX_CONN_ = connSize;
    freeCount_ = connSize;
    useCount_ = 0;
    if (semInitialized_) {
        sem_destroy(&semId_);
        semInitialized_ = false;
    }
    // sem_init(信号量指针, 0表示线程间共享, 初始值)
    if (sem_init(&semId_, 0, MAX_CONN_) != 0) {
        LOG_ERROR("sem_init failed for sql conn pool");
        DestroyPool();
        return false;
    }
    semInitialized_ = true;
    isInitialized_ = true;
    poolDestroyed_ = false;
    return true;
}

// 从池中取出一个连接
MYSQL *SqlConnPool::GetConn()
{
    if (!isInitialized_ || !semInitialized_) {
        return nullptr;
    }

    // 等待信号量 (资源 -1)，如果没有资源则阻塞
    sem_wait(&semId_);
    MYSQL *sql = nullptr;
    // 加锁保护队列
    {
        lock_guard<mutex> locker(mtx_);
        if (connQue_.empty()) {
            sem_post(&semId_);
            return nullptr;
        }
        sql = connQue_.front();
        connQue_.pop();
        --freeCount_;
        ++useCount_;
    }
    return sql;
}

// 释放连接，放回连接池
void SqlConnPool::FreeConn(MYSQL *sql)
{
    if(!sql || !isInitialized_ || !semInitialized_) return;
    { // 加锁保护队列
        lock_guard<mutex> locker(mtx_);
        connQue_.push(sql);
        ++freeCount_;
        --useCount_;
    }
    // 资源 +1，唤醒等待的线程
    sem_post(&semId_);
}

// 获取当前空闲的连接数
int SqlConnPool::GetFreeConnCount()
{
    lock_guard<mutex> locker(mtx_);
    return connQue_.size();
}

// 销毁所有连接
void SqlConnPool::DestroyPool()
{
    lock_guard<mutex> locker(mtx_);
    if (poolDestroyed_) {
        return;
    }
    while (!connQue_.empty())
    {
        auto item = connQue_.front();
        connQue_.pop();
        mysql_close(item); // 销毁 MySQL 连接
    }
    useCount_ = 0;
    freeCount_ = 0;
    MAX_CONN_ = 0;
    isInitialized_ = false;
    poolDestroyed_ = true;
    if (semInitialized_) {
        sem_destroy(&semId_);
        semInitialized_ = false;
    }
}

SqlConnPool::~SqlConnPool() {
    DestroyPool();
} // 析构函数私有化，防止外部删除
