#ifndef UDPSENDERLISTDLG_H
#define UDPSENDERLISTDLG_H

#include <QDialog>
#include <QWidget>
#include <QTableWidgetItem>
#include"udpsender.h"

namespace Ui {
class UdpSenderListDlg;
}

class UdpSenderListDlg : public QDialog
{
    Q_OBJECT

    struct IpInfo
    {
        IpInfo()
        {

        }
        IpInfo(QString ip,QString port)
        {
            this->ip = ip;
            this->port = port;
        }

        QString ip;
        QString port;
    };

public:
    explicit UdpSenderListDlg(QWidget *parent = 0);
    ~UdpSenderListDlg();

    void sendBroadcastList(QString info);  //����Ϣ�㲥��listIp

private slots:
    void on_pushButton_add_clicked();

    void on_pushButton_delete_clicked();

    void on_pushButton_flush_clicked();

    void itemChanged_slot(QTableWidgetItem* Item);

    void on_pushButton_add_released();

    void on_pushButton_add_pressed();

    void on_pushButton_delete_pressed();

    void on_pushButton_delete_released();

    void on_pushButton_flush_pressed();

    void on_pushButton_flush_released();

	void readIPconfigaFile();

private:
    Ui::UdpSenderListDlg *ui;
    int rowCount ;
    int RowHeight;
    QVector<IpInfo> iPInfoList;

    bool checkIplegal(QString ip);
    bool checkPortlegal(QString port);
    void addNewRow();
    void deleteRow();

    UdpSender udpSender;
};

#endif // UDPSENDERLISTDLG_H
