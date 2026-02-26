#include "webserver.h"
#include <iostream>
using namespace std;

// 构造函数：装配所有零件
WebServer::WebServer(int port, const char *srcDir,
                     const char *sqlUser, const char *sqlPwd, const char *dbName,
                     int connPoolNum, int threadNum)
    : port_(port), isClose_(false), srcDir_(const_cast<char *>(srcDir)),
      epoller_(new Epoller()), threadpool_(new ThreadPool(threadNum))
{
    // 1. 初始化数据库连接池
    SqlConnPool::Instance()->Init("localhost", 3306, sqlUser, sqlPwd, dbName, connPoolNum);

    // 2. 初始化 HttpConn 的全局静态变量
    HttpConn::srcDir_ = srcDir_;
    HttpConn::userCount_ = 0;

    // 3. 设置 Epoll 的事件模式 (监听用 LT，连接用 ET)
    InitEventMode_();

    // 4. 初始化监听 Socket
    if (!InitSocket_())
    {
        isClose_ = true;
    }
}

WebServer::~WebServer()
{
    close(listenFd_);
    isClose_ = true;
    SqlConnPool::Instance()->DestroyPool(); // 关闭数据库池
}
void WebServer::Start(){
    int timeMS = -1; // -1 表示无事件时永远阻塞休眠
    if(!isClose_) { cout << "========== Server start at port " << port_ << " ==========" << endl; }
    while(!isClose_){
        // 等待事件发生
        int eventCnt = epoller_->Wait(timeMS);
        for(int i = 0; i < eventCnt; i++) {
            // 获取发生事件的 fd 和事件类型
            int fd = epoller_->GetEventFd(i);
            uint32_t events = epoller_->GetEvents(i);
            
            if(fd == listenFd_) {
                // 如果是监听 fd 发生事件，说明有新用户连上来了！
                DealListen_();
            }else if(events & (EPOLLRDHUP | EPOLLHUP | EPOLLERR)) {
                // 发生错误或者客户端断开
                assert(users_.count(fd) > 0);
                CloseConn_(&users_[fd]);
            }else if(events & EPOLLIN) {
                // 有数据发过来了 (浏览器发了 HTTP 请求)
                assert(users_.count(fd) > 0);
                DealRead_(&users_[fd]);
            }
            else if(events & EPOLLOUT) {
                // 缓冲区空了，可以继续发数据了 (响应头/网页还没发完)
                assert(users_.count(fd) > 0);
                DealWrite_(&users_[fd]);
            }
    }
}
}
// --- 核心网络初始化 ---
bool WebServer::InitSocket_()
{
    if(port_ > 65535 || port_ < 1024) return false;

    // 1. 创建监听 Socket
    listenFd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listenFd_ < 0)    {
        perror("socket");
        return false;
    }

    // 2. 设置 Socket 选项：端口复用
    int optval = 1;
    setsockopt(listenFd_, SOL_SOCKET, SO_REUSEADDR, (const void*)&optval, sizeof(int));
    
    // 绑定端口
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port_);
    if (bind(listenFd_, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        return false;
    }

    // 监听
    if(listen(listenFd_, 6) < 0) {
        return false;
    }
    // 将 listenFd 挂载到 Epoll 上
    epoller_->AddFd(listenFd_, listenEvent_ | EPOLLIN);
    
    // 设置非阻塞 (配合 ET 模式必须)
    fcntl(listenFd_, F_SETFL, fcntl(listenFd_, F_GETFL, 0) | O_NONBLOCK);
    return true;
} // 初始化监听 socket
void WebServer::InitEventMode_()
{
    listenEvent_ = EPOLLRDHUP;
    // EPOLLONESHOT: 保证一个 socket 连接在任一时刻只被一个线程处理

    connEvent_ = EPOLLONESHOT | EPOLLRDHUP;

    // 使用边缘触发模式 (ET)
    listenEvent_ |= EPOLLET;
    connEvent_ |= EPOLLET;
    HttpConn::isET = true;
} // 初始化事件模式（LT/ET）
void WebServer::AddClient_(int fd, sockaddr_in addr) {
    users_[fd].Init(fd, addr);
    // 把新连接设为非阻塞
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
    // 加入 Epoll 监听读事件
    epoller_->AddFd(fd, EPOLLIN | connEvent_);
} // 添加新客户端连接
void WebServer::DealListen_() {
    struct sockaddr_in addr;
    socklen_t len = sizeof(addr);
    
    // 因为是 ET 模式，必须把队列里所有的新连接一次性 accept 完
    while(true) {
        int fd = accept(listenFd_, (struct sockaddr *)&addr, &len);
        if(fd <= 0) break; // 没有新连接了
        
        // 分配 HttpConn，加入 Epoll 监控
        AddClient_(fd, addr);
    }
}                        // 处理监听事件
// 关闭客户端
void WebServer::CloseConn_(HttpConn* client) {
    assert(client);
    epoller_->DelFd(client->GetFd());
    client->Close();
}
// --- 事件处理派发 ---
 // 处理监听事件
void WebServer::DealRead_(HttpConn *client) {
    assert(client);
    threadpool_->AddTask([this, client]() { OnRead_(client); });
}  // 处理读事件(丢进线程池)
void WebServer::DealWrite_(HttpConn *client) {
    assert(client);
    threadpool_->AddTask([this, client]() { OnWrite_(client); });
} // 处理写事件（丢进线程池）

// --- 线程池实际执行的函数 ---
void WebServer::OnRead_(HttpConn *client) {int readErrno = 0;
    int ret = client->Read(&readErrno);
    if(ret <= 0 && readErrno != EAGAIN) {
        CloseConn_(client); // 读错了，直接关掉
        return;
    }
    // 读取成功，开始状态机解析和处理业务
    OnProcess_(client);
}
void WebServer::OnWrite_(HttpConn *client) {
    int writeErrno = 0;
    int ret = client->Write(&writeErrno);
    if(client->ToWriteBytes() == 0) {
        // 全发完了！如果是长连接，继续监听读事件；否则关掉
        if(client->IsKeepAlive()) {
            OnProcess_(client);
            return;
        }
    }
    else if(ret < 0 && writeErrno == EAGAIN) {
        // 缓冲区满了发不进去了，继续监听可写事件
        epoller_->ModFd(client->GetFd(), connEvent_ | EPOLLOUT);
        return;
    }
    CloseConn_(client); // 其他情况或者短连接，直接关闭
}
void WebServer::OnProcess_(HttpConn *client) {
    if(client->Process()) {
        // 解析成功，有数据要发，监听可写事件
        epoller_->ModFd(client->GetFd(), connEvent_ | EPOLLOUT);
    } else {
        // 解析失败或者数据没收全，继续监听可读事件
        epoller_->ModFd(client->GetFd(), connEvent_ | EPOLLIN);
    }
}