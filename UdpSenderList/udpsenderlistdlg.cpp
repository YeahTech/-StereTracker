#include "udpsenderlistdlg.h"
#include "ui_udpsenderlistdlg.h"
#include <QDebug>
#include <QMessageBox>

UdpSenderListDlg::UdpSenderListDlg(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::UdpSenderListDlg)
{
    ui->setupUi(this);
//    setWindowFlags(Qt::FramelessWindowHint);//ÎÞ±ß¿ò

	this->setWindowTitle("Pose Sender");

    rowCount = 1;
    RowHeight = 30;

    this->setFixedSize(230,120);

    ui->pushButton_add->setStyleSheet("border-image:url(:/senderListRs/add.png)");
    ui->pushButton_delete->setStyleSheet("border-image:url(:/senderListRs/delete.png)");
    ui->pushButton_flush->setStyleSheet("border-image:url(:/senderListRs/refresh.png)");

//    ui->pushButton_add->setStyleSheet( "QPushButton:pressed{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,stop: 0 #dadbde, stop: 1 #f6f7fa);}");

    ui->tableWidget->setColumnCount(2);
    QStringList header;
    header<<"IP"<<"Port";
    ui->tableWidget->setHorizontalHeaderLabels(header);

    ui->tableWidget->setRowCount(1);
    ui->tableWidget->setRowHeight(0,RowHeight);

    ui->tableWidget->setColumnWidth(0,140);
    ui->tableWidget->setColumnWidth(1,70);

    connect(ui->tableWidget,SIGNAL(itemChanged(QTableWidgetItem*)),this, SLOT(itemChanged_slot(QTableWidgetItem*)) );
	readIPconfigaFile();
}

UdpSenderListDlg::~UdpSenderListDlg()
{
    delete ui;
}

void UdpSenderListDlg::sendBroadcastList(QString info) 
{
    for(int i = 0; i < iPInfoList.size(); i++)
    {
        udpSender.sendString2dest(info,iPInfoList[i].ip,iPInfoList[i].port.toInt());
    }
}
void UdpSenderListDlg::addNewRow()
{

    ui->tableWidget->setRowCount(ui->tableWidget->rowCount()+1);
    ui->tableWidget->setRowHeight(ui->tableWidget->rowCount()-1,RowHeight);

    ui->tableWidget->resize(230,80+(ui->tableWidget->rowCount()-1)*RowHeight);
	

    this->setFixedSize(230,120+(ui->tableWidget->rowCount()-1)*RowHeight);
}
void UdpSenderListDlg::deleteRow()
{
    if(ui->tableWidget->rowCount()>=0)
    {
        ui->tableWidget->removeRow( ui->tableWidget->currentRow());
        ui->tableWidget->resize(230,80+(ui->tableWidget->rowCount()-1)*RowHeight);
        this->setFixedSize(230,120+(ui->tableWidget->rowCount()-1)*RowHeight);
    }
}

void UdpSenderListDlg::on_pushButton_add_clicked()
{
    addNewRow();
}

void UdpSenderListDlg::on_pushButton_delete_clicked()
{
    deleteRow();
}

void UdpSenderListDlg::on_pushButton_flush_clicked()
{
    iPInfoList.clear();
    for(int i= 0; i < ui->tableWidget->rowCount(); i++)
    {
       if(ui->tableWidget->item(i,0)==0)
           return;
       QString ip = ui->tableWidget->item(i,0)->text();
       QString port = ui->tableWidget->item(i,1)->text();

       if(!checkIplegal(ip))
       {
           QMessageBox::warning(this,  QString::fromLocal8Bit("¾¯¸æ"),  QString::fromLocal8Bit("IPµØÖ·´æÔÚ´íÎó"));
           return;
       }

       if(!checkPortlegal(port))
       {
           QMessageBox::warning(this,  QString::fromLocal8Bit("¾¯¸æ"),  QString::fromLocal8Bit("¶Ë¿ÚµØÖ·´æÔÚ´íÎó"));
           return;
       }

       iPInfoList.push_back(IpInfo(ip,port));
    }
}

void UdpSenderListDlg::itemChanged_slot(QTableWidgetItem *Item)
{
   if(Item->column()== 0) //ip
   {
       QString ip = Item->text();
       if(!checkIplegal(ip))
           QMessageBox::warning(this,  QString::fromLocal8Bit("¾¯¸æ"),  QString::fromLocal8Bit("IPµØÖ·´æÔÚ´íÎó"));
   }
   else if(Item->column()== 1) //port
   {
       QString port = Item->text();
       if(!checkPortlegal(port))
           QMessageBox::warning(this,  QString::fromLocal8Bit("¾¯¸æ"),  QString::fromLocal8Bit("¶Ë¿ÚµØÖ·´æÔÚ´íÎó"));
   }
}
bool UdpSenderListDlg::checkIplegal(QString ip)
{
    QRegExp rx2Ip("\\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\b");
    if( !rx2Ip.exactMatch(ip) )
        return false;
    else
        return true;
}

bool UdpSenderListDlg::checkPortlegal(QString port)
{
    QRegExp rx2port("^([0-9]|[1-9]\\d|[1-9]\\d{2}|[1-9]\\d{3}|[1-5]\\d{4}|6[0-4]\\d{3}|65[0-4]\\d{2}|655[0-2]\\d|6553[0-5])$");
    if( !rx2port.exactMatch(port))
        return false;
    else
        return true;
}


void UdpSenderListDlg::on_pushButton_add_released()
{
    ui->pushButton_add->resize(40,40);
}

void UdpSenderListDlg::on_pushButton_add_pressed()
{
    ui->pushButton_add->resize(45,45);
}

void UdpSenderListDlg::on_pushButton_delete_pressed()
{
    ui->pushButton_delete->resize(45,45);
}

void UdpSenderListDlg::on_pushButton_delete_released()
{
    ui->pushButton_delete->resize(40,40);
}

void UdpSenderListDlg::on_pushButton_flush_pressed()
{
     ui->pushButton_flush->resize(40,40);
}

void UdpSenderListDlg::on_pushButton_flush_released()
{
    ui->pushButton_flush->resize(35,35);
}

void UdpSenderListDlg::readIPconfigaFile()
{
	QSettings settings("trackerCofig.ini", QSettings::IniFormat);

	QString ip = settings.value("SenderTo/ip").toString();
	QString port = settings.value("SenderTo/Port").toString();

	iPInfoList.push_back(IpInfo(ip,port));
	addNewRow();
	
	ui->tableWidget->setItem(0,0,new QTableWidgetItem(ip));
	ui->tableWidget->setItem(0,1,new QTableWidgetItem(port));
}
