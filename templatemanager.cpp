#include "templatemanager.h"
#include "ui_templatemanager.h"

#include <QDebug>
#include <fstream>
#include <QMessageBox>
#include <QKeyEvent>
#include <QFile>
#include "zoezentracker.h"


TemplateManager::TemplateManager(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::TemplateManager)
{

    ui->setupUi(this);
	setWindowFlags(windowFlags()&~Qt::WindowMaximizeButtonHint);
	setFixedSize(this->width(),this->height());

	markerPath = "markers";
}

TemplateManager::~TemplateManager()
{
    delete ui;
}



QFileInfoList TemplateManager::getFolderFileList(QString path ,QString fileSuffix)
{

    QFileInfoList fileInfo_list;

    //判断路径是否存在
    QDir dir(path);
    if(!dir.exists())
    {
        return fileInfo_list;
    }
    dir.setFilter(QDir::Files | QDir::NoSymLinks);
    QFileInfoList list = dir.entryInfoList();


    for(int i=0; i <list.count();i++)
    {
        QFileInfo fileInfo = list.at(i);
        QString suffix = fileInfo.suffix();
        if(QString::compare(suffix,fileSuffix, Qt::CaseInsensitive) == 0)
        {
            fileInfo_list.push_back(list[i]);
        }
    }

    return fileInfo_list;
}

bool TemplateManager::loadMarkers(QString path)
{
	markerPath = path;


    markerInput.clear();
	ui->listWidget->clear();

    QFileInfoList markerList = getFolderFileList(path,"txt");

	if(markerList.size() == 0)
		return false;

    for(int i = 0; i < markerList.size(); i++)
    {
        QString markpath =markerList[i].absoluteFilePath();

        InputMarker X;

        ifstream fin(markpath.toLatin1().data());
        if(!fin)
        {
            continue;
        }

		string name;
		fin>>name;	
		strcpy(X.name,name.data());
		//X.name = (char*)name.data();

        fin>>X.direction;
        
        fin>>X.num;

		for(int j = 0; j < X.num; j++)
		{
			cv::Point3f Y;
			char a, b;
			string point;
			fin>>point;
			const char* buffer;
			buffer = point.data();
			
			sscanf(buffer, "%f%c%f%c%f", &Y.x, &a, &Y.y, &b, &Y.z);
			X.coordinate.push_back(Y);

		}
        markerInput.push_back(X);


        ui->listWidget->addItem(QString(QLatin1String(X.name)));

        fin.close();
    }

	
	if(markerInput.size() == markerList.size())
		return true;
	else
		return false;
}

void TemplateManager::saveMarker(QString path,InputMarker marker)
{
	if ( !QDir(path).exists())
	{
		QDir dir;
		dir.mkdir(path);
	}


	QString filename = path+"\\"+QString(QLatin1String(marker.name))+".txt";


	ofstream fout(filename.toLatin1().data());
	if(fout)
	{
		qDebug()<<filename;
		fout<<marker.name<<endl<<marker.direction<<endl<<marker.num<<endl;

		for(int i = 0; i <marker.coordinate.size();i++)
			fout<<marker.coordinate[i].x<<","<<marker.coordinate[i].y<<","<<marker.coordinate[i].z<<endl;
	}

	fout.close();

}

vector<InputMarker> TemplateManager::getInputMarkers()
{
	return markerInput;
}

void TemplateManager::setSampleCount(int count)
{
	ui->label_count->setText(QString("count:%1").arg(count));
}

void TemplateManager::keyPressEvent( QKeyEvent *k )
{
    if(k->key() == Qt::Key_Delete)
    {
        deleteMarker();
    }
}


void TemplateManager::deleteMarker()
{

    QList<QListWidgetItem*> list = ui->listWidget->selectedItems();

    if(list.size() == 0)
        return;
    for(int i= 0; i <list.size(); i++ )
    {
        QListWidgetItem* sel = list[0];

        QString qustion = QString("Delete marker:   %1?").arg(sel->text());
        QMessageBox::StandardButton rb = QMessageBox::question(NULL, "question", qustion, QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
        if(rb == QMessageBox::Yes && sel )
        {
            // delete from listWidget
            int r = ui->listWidget->row(sel);
            ui->listWidget->takeItem(r);

            // delete file
            QString markerName = markerPath+QString("/")+sel->text()+QString(".txt");
            QFile::remove(markerName);

        }
    }

	loadMarkers(markerPath);

}

void TemplateManager::closeEvent(QCloseEvent *event)
{
	ZoezenTracker* pZoezenTracker = (ZoezenTracker*) QWidget::parentWidget();

	pZoezenTracker->setTrackerMode(TRACKING);
}

void TemplateManager::on_pushButton_save_clicked()
{
	
	QString filename = ui->lineEdit_name->text();
	if(filename.isEmpty())
	{
		QMessageBox::warning(this,"warning","marker name can't be empty!");
		return;
	}
		

	InputMarker marker;
	//marker.name = filename.toLatin1().data();
	strcpy(marker.name, filename.toLatin1().data());

	marker.direction = -1;
	marker.num = 3;
	marker.coordinate.push_back(cv::Point3f(0,0,0));
	marker.coordinate.push_back(cv::Point3f(samplePoints.getXAxisLength(),0,0));
	marker.coordinate.push_back(cv::Point3f(0,samplePoints.getYAxisLength(),0));


	this->saveMarker(markerPath,marker);
	setSampleCount(0);
	this->loadMarkers(markerPath);


}
void TemplateManager::on_pushButton_Samples_clicked()
{
	ZoezenTracker* pZoezenTracker = (ZoezenTracker*) QWidget::parentWidget();

	if(ui->pushButton_Samples->text() == QString("Samples"))
	{
		ui->pushButton_Samples->setText("Stop");

		pZoezenTracker->startGetSamples();
	}
	else if(ui->pushButton_Samples->text() == QString("Stop"))
	{
		ui->pushButton_Samples->setText("Samples");

		pZoezenTracker->stopGetSamples();

		samplePoints = pZoezenTracker->getSamples();
	}
}

void TemplateManager::on_pushButton_Apply_clicked()
{
	ZoezenTracker* pZoezenTracker = (ZoezenTracker*) QWidget::parentWidget();

	pZoezenTracker->setTrackerMode(TRACKING);
}
