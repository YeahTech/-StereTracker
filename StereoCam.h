#pragma once
#include "UsbCamera.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv.h>


class StereoCam
{
public:
	StereoCam(void);
	~StereoCam(void);

	bool initStereoCam();
	void getStereoImages(cv::Mat& leftImage,cv::Mat& rightImage);
	void destoryStereoCam();

private:
	UsbCamera leftCam,rightCam;
	UsbCamera* pCameraLeft;
	UsbCamera* pCameraRight;
	ImageBuffer bufferLeft, bufferRight;


};

