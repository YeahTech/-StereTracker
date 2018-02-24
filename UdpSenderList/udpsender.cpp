#include "udpsender.h"
#include <QDebug>
UdpSender::UdpSender()
{
    this->sender = new QUdpSocket;
    destIp = QHostAddress("0.0.0.0");
    destPort = 0;

}

UdpSender::UdpSender(QString destIp, Port destPort)
{
    this->setDestIp(destIp);
    this->setdestPort(destPort);
}

UdpSender::~UdpSender()
{
    delete this->sender;
}

void UdpSender::setDestIp(QString destIp)
{
    this->destIp=QHostAddress(destIp);
}


void UdpSender::setdestPort(Port destPort)
{
    this->destPort=destPort;
}

void UdpSender::setDestPortAndIp(QString destIp, Port destPort)
{
    this->setDestIp(destIp);
    this->setdestPort(destPort);
}

qint64 UdpSender::sendString(QString sendBuffer)
{
    if(destIp != QHostAddress ("0.0.0.0") && destPort != 0)
    {
         qint64 ret = this->sender->writeDatagram(sendBuffer.toUtf8(),this->destIp,this->destPort);
         return ret;
    }
    else
    {
        qDebug()<<"please set destIp and dest Port";
        return 0;
    }
}

qint64 UdpSender::sendString2dest(QString sendBuffer, QString destIp, Port destPort)
{
    this->setDestIp(destIp);
    this->setdestPort(destPort);
    qint64 ret = this->sendString(sendBuffer);
    return ret;
}

void UdpSender::getLocalIP()
{
    QList<QHostAddress> list = QNetworkInterface::allAddresses();
    foreach (QHostAddress address, list)
    {
       if(address.protocol() == QAbstractSocket::IPv4Protocol)
       {
           if(address.toString().startsWith("192.168.") && !address.toString().contains("192.168.14"))
           {
               this->localIP = address.toString();

           }
       }
    }
}

void UdpSender::writelocalip()
{
    QString dataToSend = QString("slaverIP#%1").arg(this->localIP);
    this->sendString2dest(dataToSend,"192.168.1.255",4444);
    qDebug()<<dataToSend;
}
