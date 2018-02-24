#ifndef IDSCAMERA_H
#define IDSCAMERA_H

#include "uEye.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv.h>

using namespace cv;




struct Camera
{
	HIDS		hCam;
	char*		pImageMem;
	int			nImageID;
	int			RenderMode;
	SENSORINFO	SensorInfo;
	CAMINFO		CamInfo;
	int         nBitsPerPixel;
	bool		bLive;
	bool        bOpened;
};
class IDSCamera
{
public:
    IDSCamera();

    ~IDSCamera();
    void GetImageFromCamera(Mat& leftImg,Mat& rightImg);
	bool initTracker();
    void closeTracker();
	bool isOpened();
	void saveImage();

private:
	int initCam(Camera &cam, const char *cfg);
	Camera cameraL,cameraR;
	int initFlag;
	Mat leftImage,rightImage;
};

#endif // IDSCAMERA_H
