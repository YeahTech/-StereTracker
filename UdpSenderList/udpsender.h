#ifndef UDPSENDER_H
#define UDPSENDER_H
/**
  没有定义广播的方法，需要进行广播，只需要将目标换成广播ip即可
  为了遵守单一职责原则，UdpSender只负责发送和广播，不负责接收。接收数据由Udpreceiver实现
  */


#include <QtNetwork>

typedef quint16 Port;   //重定义quint16为Port类型

class UdpSender:public QObject  //继承QObject是为了以后能用信号与槽
{
    Q_OBJECT
public:
    UdpSender();   //无参数构造

    UdpSender(QString destIp, Port destPort);  //通过目标ip和目标端口构造

    ~UdpSender();
private:

    QUdpSocket *sender;

    QHostAddress destIp;

    Port destPort;

    QString localIP;

public:
    void setDestIp(QString destIp);    //设置ip

    void setdestPort(Port destPort);   //设置端口

    void setDestPortAndIp(QString destIp, Port destPort);  //设置ip和端口

    qint64 sendString(QString sendBuffer);  //发送，需提前设置ip和端口

    qint64 sendString2dest(QString sendBuffer, QString destIp, Port destPort); //发送到目标ip和端口



    void getLocalIP();
    void writelocalip();
};

#endif // UDPSENDER_H
