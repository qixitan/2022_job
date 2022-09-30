# 计算机网络

## OSI模型：

[参考资料](https://blog.csdn.net/w2009211777/article/details/124035909)

- **物理层【比特流】：**底层数据传输、如网卡；
- **数据链路层【帧】：**定义数据的基本格式，如何传输、如何标识；如网卡MAC地址； **地址解析协议 （[ARP](https://baike.baidu.com/item/ARP/609343?fromtitle=ARP%E5%8D%8F%E8%AE%AE&fromid=1742212&fr=aladdin)）:据IP地址获取物理地址的一个TCP/IP协议**、**反向地址转换协议（[RARP](https://baike.baidu.com/item/%E5%8F%8D%E5%90%91%E5%9C%B0%E5%9D%80%E8%BD%AC%E6%8D%A2%E5%8D%8F%E8%AE%AE?fromModule=lemma_search-box)）:发出要反向解析的物理地址并希望返回其对应的IP地址，应答包括由能够提供所需信息的RARP服务器发出的IP地址**
- **网络层【包】：**定义IP地址，定义路由功能；如不同设备数据的转发, ip协议
- **传输层【段】：**端到端数据传输的基本功能；如TCP、UDP、TLS。
- **会话层：**控制应用程序建的会话能力；如不同软件数据分发给不同软件
- **表示层：**数据格式标识，基本的压缩加密功能
- **应用层：**各种的应用软件；包括web应用， HTTP、 HTTPS

## TCP：

TCP（Transmission Control Protocol 传输控制协议）是一种面向连接的、可靠的、基于字节流的传输层
通信协议。

### 确保传输可靠性的方式TCP协议保证数据传输可靠性的方式主要有：

[详细参考](http://t.zoukankan.com/l199616j-p-11406914.html)

1. 校验和
2. 序列号
3. 确认应答
4. 超时重传
5. 连接管理
6. 流量控制
7. 拥塞控制

## [在浏览器中输入URL后都会发生什么](https://blog.csdn.net/Richardjgp/article/details/125159922)

## [交换机的原理](https://product.pconline.com.cn/itbk/software/dnyw/1707/9601528.html)

## [http协议](https://blog.csdn.net/u010710458/article/details/79636625)

## [http和https的区别](https://www.cnblogs.com/vickylinj/p/10925733.html)

## [android软件卡顿的原因](https://blog.csdn.net/weixin_52700281/article/details/122883733)

## [arp协议攻击](https://blog.csdn.net/qq_48708303/article/details/116446976)

## [arp广播](https://baike.baidu.com/item/ARP/609343?fr=aladdin)

## [ftp协议端口号](https://edu.iask.sina.com.cn/jy/a9m9tZtsyC2.html)

# 测试

## [自动化测试的优缺点](https://blog.csdn.net/qq_37964547/article/details/82384976)

## [局域网采用什么协议](https://baike.baidu.com/item/%E5%B1%80%E5%9F%9F%E7%BD%91%E5%8D%8F%E8%AE%AE/2298655)

## [测试方法有哪些](https://blog.csdn.net/qq_45042752/article/details/125893902)

## [如何对复制粘贴进行测试](https://blog.csdn.net/qq_60219215/article/details/118883402)
