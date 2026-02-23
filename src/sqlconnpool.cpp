#include "sqlconnpool.h"

using namespace std;

SqlConnPool::SqlConnPool()
{
    useCount_ = 0;
    freeCount_ = 0;
} // 构造函数私有化，防止外部创建

SqlConnPool *SqlConnPool::Instance()
{
    static SqlConnPool connPool;
    return &connPool;
} // 单例模式，获取唯一实例

//// 初始化：主机、端口、用户名、密码、库名、池大小
void SqlConnPool::Init(const char *host, int port,
                       const char *user, const char *pwd,
                       const char *dbName, int connSize)
{
    assert(connSize > 0);
    // 循环创建连接
    for (int i = 0; i < connSize; i++)
    {
        MYSQL *sql = nullptr;
        sql = mysql_init(sql);
        assert(sql);
        if (!sql)
        {
            cerr << "MySQL Error: " << mysql_error(sql) << endl;
            assert(false);
        }
        sql = mysql_real_connect(sql, host, user, pwd, dbName, port, nullptr, 0);
        if (!sql)
        {
            cerr << "MySQL Error: " << mysql_error(sql) << endl;
            assert(false);
        }
        connQue_.push(sql);
    }
    MAX_CONN_ = connSize;
    freeCount_ = connSize;
    useCount_ = 0;
    // sem_init(信号量指针, 0表示线程间共享, 初始值)
    sem_init(&semId_, 0, MAX_CONN_); // 初始化信号量
}

// 从池中取出一个连接
MYSQL *SqlConnPool::GetConn()
{

    // 等待信号量 (资源 -1)，如果没有资源则阻塞
    sem_wait(&semId_);
    MYSQL *sql = nullptr;
    // 加锁保护队列
    {
        lock_guard<mutex> locker(mtx_);
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
    assert(sql);
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
    while (!connQue_.empty())
    {
        auto item = connQue_.front();
        connQue_.pop();
        mysql_close(item); // 销毁 MySQL 连接
    }
    mysql_library_end();
}

SqlConnPool::~SqlConnPool() {
    DestroyPool();
} // 析构函数私有化，防止外部删除
