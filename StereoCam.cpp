#include "StdAfx.h"
#include "StereoCam.h"


StereoCam::StereoCam(void)
{
}


StereoCam::~StereoCam(void)
{
}

bool StereoCam::initStereoCam()
{
	pCameraLeft = leftCam.getCamera(1,30,1280,720,"MEDIASUBTYPE_RGB24");
	pCameraRight = rightCam.getCamera(0,30,1280,720,"MEDIASUBTYPE_RGB24");
	if(NULL == pCameraLeft || NULL == pCameraRight)
	{
		return false;
	}
	else
	{
		return true;
	}
}

void StereoCam::getStereoImages(cv::Mat& leftImage,cv::Mat& rightImage)
{
	bufferLeft = pCameraLeft->getImage();
	bufferRight = pCameraRight->getImage();

	leftImage = cv::Mat(bufferLeft.height,bufferLeft.width, CV_8UC3, bufferLeft.buffer);
	rightImage = cv::Mat(bufferLeft.height,bufferLeft.width, CV_8UC3, bufferRight.buffer);
}

void StereoCam::destoryStereoCam()
{
	pCameraLeft->destroyUsbCamera();
	pCameraRight->destroyUsbCamera();
}

