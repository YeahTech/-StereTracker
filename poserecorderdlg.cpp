#include "poserecorderdlg.h"
#include "ui_poserecorderdlg.h"
#include <QFileDialog>
#include <QTextStream>
#include <QDebug>

Poserecorderdlg::Poserecorderdlg(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Poserecorderdlg)
{
    ui->setupUi(this);
	recordFlag = false;
}

Poserecorderdlg::~Poserecorderdlg()
{
    delete ui;
}

void Poserecorderdlg::on_pushButton_save_clicked()
{
    QString fileName = QFileDialog::getSaveFileName(this,
            tr("save MarkerInfo"),
            "",
            tr("MarkerInfo Files (*.txt)"));

        if (!fileName.isNull())
        {
            saveMarkerInfo(fileName);
        }
        else
        {
            
        }
}

void Poserecorderdlg::saveMarkerInfo(QString filename)
{
	
	QFile f(filename);
	if(!f.open(QIODevice::WriteOnly | QIODevice::Text))  
	{  
		cout << "Open failed." << endl;  
		return ;  
	}  

	QTextStream txtOutput(&f);

	for (int i =0; i< markerInfoList.size(); i++)
	{
		txtOutput<<markerInfoList[i];
		txtOutput<<"\n";
	}
	f.close();
}

void Poserecorderdlg::recordMarkerInfo(vector<DetectedMarker> Maker_Detected )
{
	if(!recordFlag)
		return;

	if(Maker_Detected.size() == 0)
		return;

	QString markersInfo;
	for(int i= 0; i <Maker_Detected.size(); i++ )
	{
		markersInfo.append(Maker_Detected[i].name);
		markersInfo.append("\n");
		for(int row = 0; row < 3; row++)
		{
			for (int col = 0; col < 3; col++)
			{
				markersInfo.append(QString::number(Maker_Detected[i].R(row,col),'f',3));
				markersInfo.append("\t");
			}
			markersInfo.append(QString::number(Maker_Detected[i].T(row),'f',3));
			markersInfo.append("\n");
		}
		markersInfo.append("0\t0\t0\t1\n");

		double Ylength = Maker_Detected[i].getYLength();

		//markersInfo.append(QString::number(Ylength,'f',3));
		//markersInfo.append("\n");
		

	}

	
	markerInfoList.append(markersInfo);
	ui->label_count->setText(QString("count:%1").arg(markerInfoList.size()));
	
}

void Poserecorderdlg::on_pushButton_record_clicked()
{
	if(ui->pushButton_record->text() == "Record")
	{
		ui->pushButton_record->setText("Stop");
		markerInfoList.clear();
		recordFlag = true;
	}
	else
	{
		ui->pushButton_record->setText("Record");
		recordFlag = false;
	}

}