#ifndef ZOEZENTRACKER_H
#define ZOEZENTRACKER_H

#include <QMainWindow>
#include "MarkerDetector.h"
#include "StereoTracker.h"
#include "CameraDS.h"
#include "QTimer"
#include "templatemanager.h"
#include "UdpSenderList/udpsenderlistdlg.h"
#include "poserecorderdlg.h"
#include "StereoCam.h"
#include "QTextEdit.h"




namespace Ui {
class ZoezenTracker;
}

enum TRACKER_MODE
{
	
	TRACKING,
	ADD_TEMPLATE,
	CALIBRATION,
	NODEAL,
	CORNERS,
	MATCH,

};

class ZoezenTracker : public QMainWindow
{
    Q_OBJECT

public:
    explicit ZoezenTracker(QWidget *parent = 0);
    ~ZoezenTracker();

    void paintEvent ( QPaintEvent * e);

	void closeDshowCamera();

	void setTrackerMode(TRACKER_MODE trackerMode);

	void startGetSamples();

	void stopGetSamples();

	Samples getSamples();


private:
    Ui::ZoezenTracker *ui;

private slots:
	void timerUpdateSlot();

	void actionNew_Template();

	void actionConnect_TrackerSlot();

	void actionPose_SenderSlot();

	void actionPose_RecorderSlot();

	void actionPose_SaveImgSlot();
	
	void actionPose_DetectCornersSlot(bool flag);

	void actionPose_TrackingSlot();

	void actionPose_NoDealSlot();

	void actionPose_MatchSlot();

	QImage cvMat2QImage(const cv::Mat& mat);

private:

    QPixmap pixmap;

	//CCameraDS dShowCam;

	StereoCam stereoCam;

	QTimer  *timerUpdate;

	StereoTracker stereoTracker;

	MarkerDetector detector;

	TRACKER_MODE mode;

	vector<cv::Point2f> cornersLeft,cornersRight;
	vector<cv::Point3f> Coordinate3D;
	
	vector<DetectedMarker> Maker_Detected;
	vector<DetectedMarker> Pattern;

	bool initCameraFlag;

	TemplateManager *templateManager;

	Poserecorderdlg *pPoseRecorder;

	UdpSenderListDlg* pListSender;

	Samples samplesPoints;
	bool samplesFlag;

	cv::Mat rightImage_Original,leftImage_Original;

	bool showNoteFlag;

	QTextEdit* pDebugOutPut;

	
};

#endif // ZOEZENTRACKER_H
