#ifndef SQLCONNPOOL_H
#define SQLCONNPOOL_H

#include <mysql/mysql.h>
#include <string>
#include <queue>
#include <mutex>
#include <semaphore.h> // 信号量，用来控制连接数量
#include <thread>
#include <memory>
#include <iostream>
#include <cassert>

/*
 * 数据库连接池
 * 职责：负责创建、回收数据库连接
 */
class SqlConnPool
{
public:
    static SqlConnPool *Instance(); // 单例模式，获取唯一实例

    //// 初始化：主机、端口、用户名、密码、库名、池大小
    void Init(const char *host, int port,
              const char *user, const char *pwd,
              const char *dbName, int connSize = 10);

    // 从池中取出一个连接
    MYSQL *GetConn();

    // 释放连接，放回连接池
    void FreeConn(MYSQL *conn);

    // 获取当前空闲的连接数
    int GetFreeConnCount();

    // 销毁所有连接
    void DestroyPool();

private:
    SqlConnPool();  // 构造函数私有化，防止外部创建
    ~SqlConnPool(); // 析构函数私有化，防止外部删除

    int MAX_CONN_;  // 连接池最大连接数
    int useCount_;  // 已使用的连接数
    int freeCount_; // 空闲的连接数

    std::queue<MYSQL *> connQue_; // 连接池队列
    std::mutex mtx_;              // 互斥锁，保护连接池队列
    sem_t semId_;                 // 信号量，控制连接数量
};

/* RAII 机制封装类
 * 职责：对象创建时自动从池中取连接，对象销毁时自动归还
 *避免忘记 FreeConn 导致连接泄漏
 */
class SqlConnRAII
{
public:
//构造函数：传入一个 MYSQL指针的引用，会自动赋值
    SqlConnRAII(MYSQL **SQL, SqlConnPool *connPool){
        assert(connPool);
        *SQL = connPool->GetConn();//取出连接
        sql_ = *SQL;
        pool_ = connPool;
    }
    //析构函数：自动归还连接
    ~SqlConnRAII(){
        if(sql_){
            pool_->FreeConn(sql_);
        }
    }
private:
    MYSQL *sql_;               // 当前连接
    SqlConnPool *pool_;        // 连接池
};
#endif // SQLCONNPOOL_H