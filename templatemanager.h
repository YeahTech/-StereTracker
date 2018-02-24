#ifndef TEMPLATEMANAGER_H
#define TEMPLATEMANAGER_H

#include <QDialog>
#include <iostream>
#include <QDir>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv.h>

using namespace cv;
using namespace std;
namespace Ui {
class TemplateManager;
}


class InputMarker
{
public:

    char name[100];
    int direction;
    int num;
    vector<cv::Point3f> coordinate;
};


class Samples
{
public:
	void clear()
	{
		oPos_Vec.clear();
		xPos_Vec.clear();
		yPos_Vec.clear();
	}

	void addSamplePoint(cv::Point3f oPos,cv::Point3f xPos,cv::Point3f yPos)
	{
		oPos_Vec.push_back(oPos);
		xPos_Vec.push_back(xPos);
		yPos_Vec.push_back(yPos);
	}

	double cal_distance(cv::Point3f a, cv::Point3f b)
	{
		return (sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z)));
	}

	double getXAxisLength()
	{
		double length = 0;

		if(oPos_Vec.size() == 0)
			return 0;

		for (int i = 0;i < oPos_Vec.size(); i++)
		{
			length += cal_distance(oPos_Vec[i],xPos_Vec[i]);	
		}

		return length/oPos_Vec.size();

	}

	double getYAxisLength()
	{
		double length = 0;

		if(oPos_Vec.size() == 0)
			return 0;

		for (int i = 0;i < oPos_Vec.size(); i++)
		{
			length += cal_distance(oPos_Vec[i],yPos_Vec[i]);	
		}

		return length/oPos_Vec.size();

	}

	int getCount()
	{
		return oPos_Vec.size();
	}

	vector<cv::Point3f> oPos_Vec;
	vector<cv::Point3f> xPos_Vec;
	vector<cv::Point3f> yPos_Vec;
};

class TemplateManager : public QDialog
{
    Q_OBJECT

public:
    explicit TemplateManager(QWidget *parent = 0);
    ~TemplateManager();

    bool loadMarkers(QString path);

    void saveMarker(QString path, InputMarker marker);

	vector<InputMarker> getInputMarkers();

	void setSampleCount(int count);


    void keyPressEvent(QKeyEvent *k);
    void deleteMarker();
	void closeEvent(QCloseEvent *event); 

	

private slots:
    void on_pushButton_save_clicked();

	void on_pushButton_Samples_clicked();
	void on_pushButton_Apply_clicked();
private:
    QFileInfoList getFolderFileList(QString path, QString fileSuffix);

    vector<InputMarker> markerInput;
	QString markerPath;

	Samples samplePoints;

	Ui::TemplateManager *ui;
};

#endif // TEMPLATEMANAGER_H
