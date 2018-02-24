#include "udpreceiver.h"
#include <QDebug>
Udpreceiver::Udpreceiver()
{
    this->receiver = new QUdpSocket;
    this->receiveBuffer = NULL;
    this->hostIp = QHostAddress("0.0.0.0");
}

Udpreceiver::~Udpreceiver()
{
    delete this->receiver;
}

void Udpreceiver::bindPort(Port port)   //绑定端口
{
       bool ret = this->receiver->bind(port,QUdpSocket::ShareAddress);
       if(ret)
       {
           connect(this->receiver, SIGNAL(readyRead()), this, SLOT(processPendingDatagram()));//连接信号于槽函数
       }
       else
       {
//           QMessageBox::information(this,"warning","Bind port error");//提示绑定端口错误
           qDebug()<<"Bind port error";
       }
}

QHostAddress Udpreceiver::getHostIp()
{
    return hostIp;
}
void Udpreceiver::processPendingDatagram() //用于接收数据的槽函数
{
    while(this->receiver->hasPendingDatagrams())
    {
        QByteArray array;
        QHostAddress host;
        quint16 port;
        array.resize(this->receiver->pendingDatagramSize());
        this->receiver->readDatagram(array.data(), array.size(), &host, &port);

        this->hostIp = host;                      // 获得主机ip
        this->receiveBuffer = array;              //获得接收的数据

        qDebug()<<receiveBuffer.size();
        emit dateReceived((QString)this->receiveBuffer);
    }

}
