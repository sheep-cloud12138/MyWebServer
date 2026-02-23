#include "webserver.h"
#include <iostream>
#include <signal.h>
#include <stdlib.h> // system

int main() {
    // 忽略 SIGPIPE 信号，防止客户端异常断开导致服务器崩溃 (面试防挂细节)
    signal(SIGPIPE, SIG_IGN); 
    
    // 创建网站的根目录和默认首页 (为了测试方便，在代码里动态生成)
    system("mkdir -p /tmp/www");
    system("echo '<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>MyWebServer</title></head><body><h1>Hello 985 Master!</h1><p>Welcome to your high-performance C++ Web Server.</p></body></html>' > /tmp/www/index.html");

    // 配置参数：
    int port = 8080;
    const char* srcDir = "/tmp/www"; // 刚生成的根目录
    
    // MySQL 连接参数 - 修改这里为你的实际配置
    const char* sqlUser = "wjh";      // 修改为你的 MySQL 用户名
    const char* sqlPwd = "031126";           // 修改为你的 MySQL 密码 (默认为空)
    const char* dbName = "test";    // 确保这个数据库存在
    
    int sqlPoolNum = 4;
    int threadNum = 8; // 八核动力

    std::cout << "Starting WebServer on port " << port << "..." << std::endl;
    std::cout << "Connecting to MySQL: user=" << sqlUser << ", db=" << dbName << std::endl;

    // 实例化服务器大管家并启动
    WebServer server(
        port, 
        srcDir, 
        sqlUser, sqlPwd, dbName, 
        sqlPoolNum, threadNum
    );

    server.Start(); // 死循环，永不退出

    return 0;
}