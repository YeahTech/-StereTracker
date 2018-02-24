#ifndef UDPRECEIVER_H
#define UDPRECEIVER_H
/**
  Qt中的udp通信与mfc中不同的地方在于是以信号驱动的，所以没有所谓的阻塞情况，也不需要另外开线程
  来专门接收，只要创建了Udpreceiver对象，并绑定了端口，那么该对像便会监听该端口，一旦该对象接收
  到数据，那么会自动发送readyRead()信号，从而触发processPendingDatagram()函数接收数据，同时
  把发送方的ip记录下来,可通过getHostIp()方法来获取主机ip，通过sender发送。为了符合单一职能原则
  Udpreceiver类只负责接收数据和获取发送方ip，发送操作由UdpSender类实现。
  */

#include <QtNetwork>

typedef quint16 Port;
class Udpreceiver:public QObject
{
    Q_OBJECT
public:
    explicit Udpreceiver();
    ~Udpreceiver();

    void bindPort(Port port);  //绑定端口方法
    QHostAddress getHostIp();

public:
    QByteArray receiveBuffer;  //用于存储收到的数据
    QHostAddress hostIp;     //主机ip，即发送方ip

private:
    QUdpSocket *receiver;

signals:
    void dateReceived(QString receiveBuffer); //发送信号的同时将收到的数据传递给槽函数

public slots:
    void processPendingDatagram();  //用于接收数据的槽函数
};

#endif // UDPRECEIVER_H
