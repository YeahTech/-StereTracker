#include "zoezentracker.h"
#include "ui_zoezentracker.h"
#include "QPainter"
#include "QMessageBox"
#include "QDebug"

//#define USECUDA 

ZoezenTracker::ZoezenTracker(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::ZoezenTracker)
{
	
    ui->setupUi(this);
	setWindowFlags(windowFlags()&~Qt::WindowMaximizeButtonHint);
	setFixedSize(this->width(),this->height());
	ui->label_fps->setStyleSheet("background:transparent");
	

	templateManager = new TemplateManager(this);
	pListSender = new UdpSenderListDlg(this);
	pListSender = new UdpSenderListDlg(this);
	pPoseRecorder  = new Poserecorderdlg(this);


	connect(ui->actionConnect_Tracker, SIGNAL(triggered()), this, SLOT(actionConnect_TrackerSlot())); 
	connect(ui->actionNew_Template, SIGNAL(triggered()), this, SLOT(actionNew_Template())); 
	connect(ui->actionPose_Sender, SIGNAL(triggered()), this, SLOT(actionPose_SenderSlot())); 
	connect(ui->actionPose_Recorder, SIGNAL(triggered()), this, SLOT(actionPose_RecorderSlot())); 
	connect(ui->actionSave_Images, SIGNAL(triggered()), this, SLOT(actionPose_SaveImgSlot())); 
	connect(ui->actionTracking, SIGNAL(triggered()), this, SLOT(actionPose_TrackingSlot()));
	connect(ui->actionNoDeal, SIGNAL(triggered()), this, SLOT(actionPose_NoDealSlot()));
	connect(ui->actionMatch, SIGNAL(triggered()), this, SLOT(actionPose_MatchSlot()));
	connect(ui->actionDetect_Corners, SIGNAL(toggled(bool)), this, SLOT(actionPose_DetectCornersSlot(bool))); 
	ui->actionDetect_Corners->setCheckable(true);
	

	timerUpdate = new QTimer(this);
	connect(timerUpdate,SIGNAL(timeout()),this,SLOT(timerUpdateSlot()));//timeoutslot()为自定义


	initCameraFlag = false;
	samplesFlag = false;
	showNoteFlag = false;

	mode = NODEAL;
	//mode = ADD_TEMPLATE;
	//mode = NODEAL; 

	pDebugOutPut = new QTextEdit;

	pDebugOutPut->resize(200,600);

	//pDebugOutPut->show();
  
}

ZoezenTracker::~ZoezenTracker()
{

	delete templateManager;

	delete pListSender;

    delete ui;
	
}

void ZoezenTracker::paintEvent(QPaintEvent *e)
{
    QPainter painter;
    painter.begin(this);

    if(!pixmap.isNull())
        painter.drawPixmap(0,ui->mainToolBar->size().height()+ui->menuBar->size().height(), pixmap);//画打开的图片

    painter.end();
}

void ZoezenTracker::closeDshowCamera()
{
	 stereoCam.destoryStereoCam();
}


int calculateFPS()
{
	LARGE_INTEGER freq;
	LARGE_INTEGER stop_t;

	static LARGE_INTEGER start_t;

	double exe_time;

	QueryPerformanceFrequency(&freq);
	//fprintf(stdout, "The frequency of your pc is %d.\n", freq.QuadPart);
	QueryPerformanceCounter(&stop_t);

	exe_time = 1e3*(stop_t.QuadPart - start_t.QuadPart) / freq.QuadPart;
	//fprintf(stdout, "Your program executed time is %fms.\n", exe_time);
	start_t = stop_t;


	return int(1000/exe_time);
}

void cvtColorGpu(cv::Mat& src,cv::Mat& dst,int code)
{
	cv::gpu::GpuMat srcGpu(src);
	cv::gpu::GpuMat dstGpu;
	cv::gpu::cvtColor(srcGpu,dstGpu,code);
	
	dstGpu.download(dst);
}

void ZoezenTracker::timerUpdateSlot()
{
	
	stereoCam.getStereoImages(leftImage_Original,rightImage_Original);

	cv::Mat left = leftImage_Original.clone();
	cv::Mat right =rightImage_Original.clone();


	cv::cvtColor(left,left,CV_RGB2GRAY);
	cv::cvtColor(right,right,CV_RGB2GRAY);
	
	remap(left, left, stereoTracker.xMapL, stereoTracker.yMapL, INTER_LINEAR);
	remap(right, right,stereoTracker.xMapR, stereoTracker.yMapR, INTER_LINEAR);

	cv::Mat showImg;
	switch (mode)
	{
	case TRACKING:
		{
	/*		detector.RESPONSEDetector(left, cornersLeft, 3, 4, 350, 0);
			detector.RESPONSEDetector(right, cornersRight, 3, 4, 350, 0);*/
			//detector.FASTDetector(left, cornersLeft, 4);
			//detector.FASTDetector(right,cornersRight,4);
			detector.RESPONSEDetectorFaster(left, cornersLeft,3,3, 300);
			detector.RESPONSEDetectorFaster(right, cornersRight,3,3, 300);

			detector.MatchCorners(cornersLeft, cornersRight);

			detector.Calculate_3D(cornersLeft, cornersRight, Coordinate3D, StereoTracker::P1, StereoTracker::T);

			vector<InputMarker> inputMarkers = templateManager->getInputMarkers();

			detector.Marker_Recognize(cornersLeft, cornersRight, Coordinate3D,inputMarkers , Maker_Detected);

			detector.fiducialRegistration(inputMarkers, Maker_Detected);

			cvtColor(left,left,CV_GRAY2BGR);
			cvtColor(right,right,CV_GRAY2BGR);
		
			//send
			QString mInfo;
			for (int i = 0; i <Maker_Detected.size();i++ )
			{
				mInfo.append(Maker_Detected[i].toQString());
			}
			pListSender->sendBroadcastList(mInfo);

			detector.drawDetectedMarkers(left,Maker_Detected,'L',false);   

			pPoseRecorder->recordMarkerInfo(Maker_Detected);

			showImg = detector.Merge_Image(left(stereoTracker.leftValidArea),right(stereoTracker.rightValidArea));

			break;
		}

	case ADD_TEMPLATE:
		{
			detector.RESPONSEDetectorFaster(left, cornersLeft,3,3, 300);
			detector.RESPONSEDetectorFaster(right, cornersRight,3,3, 300);

			pDebugOutPut->insertPlainText(QString("left:%1Right:%2\n").arg(cornersLeft.size()).arg(cornersRight.size()));

			//detector.MatchCorners(cornersLeft, cornersRight);
			detector.MatchCornersFaster(cornersLeft, cornersRight);

			detector.Calculate_3D(cornersLeft, cornersRight, Coordinate3D, StereoTracker::P1, StereoTracker::T);

			detector.Marker_LikeDetect(cornersLeft,cornersRight,Coordinate3D,Maker_Detected);

			cvtColor(left,left,CV_GRAY2BGR);
			cvtColor(right,right,CV_GRAY2BGR);

			detector.drawDetectedMarkers(left,Maker_Detected,'L',false);

			showImg = detector.Merge_Image(left(stereoTracker.leftValidArea),right(stereoTracker.rightValidArea));

			//recored samples
			if(samplesFlag )
			{
				if (Maker_Detected.size() == 1)
				{
					samplesPoints.addSamplePoint(Maker_Detected[0].Coord3D[0],Maker_Detected[0].Coord3D[1],Maker_Detected[0].Coord3D[2]);
					templateManager->setSampleCount(samplesPoints.getCount());
				}
				else
				{
					ui->statusBar->showMessage(QString::fromLocal8Bit("too much markers!"));
				}
			}

			break;
		}

	case CORNERS:
		{
			
			detector.RESPONSEDetectorFaster(left, cornersLeft,3,3, 300);
			detector.RESPONSEDetectorFaster(right, cornersRight,3,3, 300);

			//detector.MatchCorners(cornersLeft, cornersRight);

			cvtColor(left,left,CV_GRAY2BGR);
			cvtColor(right,right,CV_GRAY2BGR);

			//detector.Calculate_3D(cornersLeft, cornersRight, Coordinate3D, StereoTracker::P1, StereoTracker::T);
			detector.drawVecPoints(left,cornersLeft,Coordinate3D,showNoteFlag);
			detector.drawVecPoints(right,cornersRight,Coordinate3D,showNoteFlag);
			
			showImg = detector.Merge_Image(left(stereoTracker.leftValidArea),right(stereoTracker.rightValidArea));

			break;
		}
		case MATCH:
		{
			detector.RESPONSEDetectorFaster(left, cornersLeft,3,3, 300);
			detector.RESPONSEDetectorFaster(right, cornersRight,3,3, 300);

			//detector.MatchCorners(cornersLeft, cornersRight);
			detector.MatchCornersFaster(cornersLeft, cornersRight);


			detector.Calculate_3D(cornersLeft, cornersRight, Coordinate3D, StereoTracker::P1, StereoTracker::T);

			cvtColor(left,left,CV_GRAY2BGR);
			cvtColor(right,right,CV_GRAY2BGR);

			showImg = detector.Merge_Image(left,right);

			detector.drawMatchCorners(showImg,cornersLeft,cornersRight);

			break;
		}
		default:
		{
			showImg = detector.Merge_Image(left,right);
		}

	}

	//showImg = detector.Merge_Image(left(stereoTracker.leftValidArea),right(stereoTracker.rightValidArea));

	cv::resize(showImg,showImg,cv::Size(showImg.cols*0.5,showImg.rows*0.5));

	ui->label_fps->setText(QString("%1FPS").arg(calculateFPS()));

	pixmap = QPixmap::fromImage(cvMat2QImage(showImg));
	this->update();

}

void ZoezenTracker::actionNew_Template()
{
	templateManager->show();
	this->setTrackerMode(ADD_TEMPLATE);
	
}

void ZoezenTracker::actionConnect_TrackerSlot()
{
	if(!timerUpdate->isActive())
	{
		if(!stereoCam.initStereoCam())
		{
			QMessageBox::warning(this,"warning","init camera failed,please check connect!");
		}
		else
		{
			ui->statusBar->showMessage(QString::fromLocal8Bit("camera connected!"));
			ui->actionConnect_Tracker->setIcon(QIcon(":Images/Standby-icon.png"));

			if(!stereoTracker.readCalibFile("calibfile/stereo_calibration_param_template.xml"))
			{
				QMessageBox::warning(this,"warning","read camera calib file failed,please check!");
				return;
			}

			if(!templateManager->loadMarkers("markers"))
			{
				QMessageBox::warning(this,"warning","read markers  failed,please check!");
			}

			detector.LOAD_SVM("data/fast_SVM.xml");
			this->setTrackerMode(NODEAL);
			timerUpdate->start(5);
		}
	}
	else
	{
		timerUpdate->stop();
		if(initCameraFlag)
			this->closeDshowCamera();
		ui->statusBar->showMessage(QString::fromLocal8Bit("camera closed!"));
		ui->actionConnect_Tracker->setIcon(QIcon(":Images/start-icon.png"));
		QPixmap nullPixmap;
		pixmap = nullPixmap;
		this->update();
	}	
}

void ZoezenTracker::actionPose_SenderSlot()
{
	pListSender->show();
}

void ZoezenTracker::actionPose_RecorderSlot()
{
	pPoseRecorder->show();
}
void ZoezenTracker::actionPose_SaveImgSlot()
{
	static int ImageNumb = 0;

	char leftfileseo[100]={0};
	sprintf(leftfileseo,"stereo_images/l%d.bmp",ImageNumb);
	char rightfileseo[100]={0};
	sprintf(rightfileseo,"stereo_images/r%d.bmp",ImageNumb);

	imwrite(leftfileseo,leftImage_Original);
	imwrite(rightfileseo,rightImage_Original);

	ImageNumb++;

}

void ZoezenTracker::actionPose_DetectCornersSlot(bool flag)
{
	setTrackerMode(CORNERS);
	if(flag)
		showNoteFlag = true;
	else
		showNoteFlag = false;
}

void ZoezenTracker::actionPose_TrackingSlot()
{
	setTrackerMode(TRACKING);
}
void ZoezenTracker::actionPose_NoDealSlot()
{
	setTrackerMode(NODEAL);
}
void ZoezenTracker::actionPose_MatchSlot()
{
	setTrackerMode(MATCH);
}

QImage ZoezenTracker::cvMat2QImage(const cv::Mat& mat)
{
	// 8-bits unsigned, NO. OF CHANNELS = 1
	if(mat.type() == CV_8UC1)
	{
		QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
		// Set the color table (used to translate colour indexes to qRgb values)
		image.setColorCount(256);
		for(int i = 0; i < 256; i++)
		{
			image.setColor(i, qRgb(i, i, i));
		}
		// Copy input Mat
		uchar* pSrc = mat.data;
		for(int row = 0; row < mat.rows; row++)
		{
			uchar* pDest = image.scanLine(row);
			memcpy(pDest, pSrc, mat.cols);
			pSrc += mat.step;
		}
		return image;
	}
	// 8-bits unsigned, NO. OF CHANNELS = 3
	else if(mat.type() == CV_8UC3)
	{
		// Copy input Mat
		const uchar* pSrc = (const uchar*)mat.data;
		// Create QImage with same dimensions as input Mat
		QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
		return image.rgbSwapped();
	}
	else if(mat.type() == CV_8UC4)
	{
		// Copy input Mat
		const uchar* pSrc = (const uchar*)mat.data;
		// Create QImage with same dimensions as input Mat
		QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
		return image.copy();
	}
	else
	{
		
		return QImage();
	}
}

void ZoezenTracker::setTrackerMode(TRACKER_MODE trackerMode)
{
	//label_showMode
	switch(trackerMode)
	{
	case TRACKING:
		{
			ui->label_showMode->setText("Mode:Tracking");
			break;
		}
	case ADD_TEMPLATE:
		{
			ui->label_showMode->setText("Mode:AddTemplate");
			break;
		}
	case CALIBRATION:
		{
			ui->label_showMode->setText("Mode:Calibration");
			break;
		}
	case NODEAL:
		{
			ui->label_showMode->setText("Mode:No Deal");
			break;
		}
	case CORNERS:
		{
			ui->label_showMode->setText("Mode:Corners");
			break;
		}
	case MATCH:
		{
			ui->label_showMode->setText("Mode:Match");
			break;
		}

	}
	this->mode = trackerMode;
}

void ZoezenTracker::startGetSamples()
{
	samplesPoints.clear();
	samplesFlag = true;
	this->setTrackerMode(ADD_TEMPLATE);
}

void ZoezenTracker::stopGetSamples()
{
	samplesFlag = false;
}

Samples ZoezenTracker::getSamples()
{
	return samplesPoints;
}


