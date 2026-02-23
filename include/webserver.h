#ifndef WEBSERVER_H
#define WEBSERVER_H

#include <unordered_map>
#include <fcntl.h>  // fcntl()
#include <unistd.h> // close()
#include <assert.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "epoller.h"
#include "threadpool.h"
#include "sqlconnpool.h"
#include "httpconn.h"

class WebServer
{
public:
    // 构造函数：传入端口号、网站根目录、数据库账号密码等
    WebServer(int port, const char *srcDir,
              const char *sqlUser, const char *sqlPwd, const char *dbName,
              int connPoolNum, int threadNum);
    ~WebServer();

    void Start();

private:
    // --- 核心网络初始化 ---
    bool InitSocket_(); // 初始化监听 socket
    void InitEventMode_();// 初始化事件模式（LT/ET）
    void AddClient_(int fd, sockaddr_in addr); // 添加新客户端连接
    void DealListen_(); // 处理监听事件
    void CloseConn_(HttpConn* client);

    // --- 事件处理派发 ---
    void DealRead_(HttpConn *client); // 处理读事件(丢进线程池)
    void DealWrite_(HttpConn *client); // 处理写事件（丢进线程池）

    // --- 线程池实际执行的函数 ---
    void OnRead_(HttpConn* client);
    void OnWrite_(HttpConn* client);
    void OnProcess_(HttpConn* client);

    // --- 服务器基础配置 ---
    int port_;
    bool isClose_;
    int listenFd_;
    char* srcDir_;

    // --- Epoll 事件模式 (LT/ET) ---
    uint32_t listenEvent_; // 监听 Socket 的事件模式
    uint32_t connEvent_;   // 客户端 Socket 的事件模式

    // --- 三大核心组件 ---
    std::unique_ptr<Epoller> epoller_;         // IO多路复用
    std::unique_ptr<ThreadPool> threadpool_;   // 线程池
    std::unordered_map<int, HttpConn> users_;  // 保存所有客户端连接映射: fd -> HttpConn

};
#endif // WEBSERVER_H