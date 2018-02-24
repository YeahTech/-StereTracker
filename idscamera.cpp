#include "idscamera.h"

//Mat ConvertMat(const UINT16*, int, int);

IDSCamera::IDSCamera()
{

    cameraL.bOpened = false;
    cameraR.bOpened = false;
    cameraL.nBitsPerPixel = 8;
    cameraR.nBitsPerPixel = 8;
	int initFlag = 0;

    /*connect(ui.startCutImage,SIGNAL(triggered()),this,SLOT(startCutImage_clicked()));
    connect(ui.calibration, SIGNAL(triggered()), this, SLOT(stereo_calib()));
    connect(ui.correct, SIGNAL(triggered()), this, SLOT(creat_correct_map()));
    connect(yolo, SIGNAL(is_detected()), this, SLOT(paint_Picture()));*/

    //while (initFlag == 0)
        //initTracker();

    //timer = new QTimer(this);
    //connect(timer,SIGNAL(timeout()),this,SLOT(GetImageFromCamera()));

    //timer->start(30);
}

IDSCamera::~IDSCamera()
{

}

void IDSCamera::GetImageFromCamera(Mat& leftImg,Mat& rightImg)
{
    //camera_start.start();
    is_FreezeVideo(cameraL.hCam, IS_DONT_WAIT);
    is_FreezeVideo(cameraR.hCam, IS_DONT_WAIT);

    //cvtColor(leftImage, load_Network::SrcImage, CV_GRAY2BGR);//灰度转RGB
    //cvtColor(rightImage, load_Network::SrcImage2, CV_GRAY2BGR);
	leftImg = leftImage;
	rightImg = rightImage;
	//imshow("left",leftImage);
	//imshow("right",rightImage);
}

void IDSCamera::saveImage()
{

	static int ImageNumb = 1;

	char filename[100]={0};
	sprintf(filename,"stereo_images/l%d.bmp",ImageNumb);
	imwrite(filename, leftImage);
	sprintf(filename,"stereo_images/r%d.bmp",ImageNumb);  //此处用四个\是因为\r是换行符
	imwrite(filename, rightImage);

	ImageNumb++;

}
int IDSCamera::initCam(Camera & cam, const char * cfg)
{
    INT nRet = is_InitCamera(&cam.hCam, 0);

    if (nRet == IS_STARTER_FW_UPLOAD_NEEDED)
    {
        INT nUploadTime = 25000;
        cam.hCam = (HIDS)(((INT)cam.hCam) | IS_ALLOW_STARTER_FW_UPLOAD);
        nRet = is_InitCamera(&cam.hCam, 0);
    }
    if (nRet == IS_SUCCESS)
    {
        cam.bOpened = true;
        nRet = is_SetExternalTrigger(cam.hCam, IS_SET_TRIGGER_SOFTWARE);
        nRet = is_GetSensorInfo(cam.hCam, &cam.SensorInfo);
        nRet = is_GetCameraInfo(cam.hCam, &cam.CamInfo);
        int width = cam.SensorInfo.nMaxWidth;
        int height = cam.SensorInfo.nMaxHeight;
        nRet = is_SetColorMode(cam.hCam, IS_CM_MONO8);
        int pixelClock = 86;
        is_PixelClock(cam.hCam, IS_PIXELCLOCK_CMD_SET, (void*)&pixelClock, sizeof(pixelClock));
        nRet = is_AllocImageMem(cam.hCam, cam.SensorInfo.nMaxWidth, cam.SensorInfo.nMaxHeight,cam.nBitsPerPixel, &cam.pImageMem, &cam.nImageID);
        nRet = is_SetImageMem(cam.hCam, cam.pImageMem, cam.nImageID);
        if (string(cfg) == "left")
        {
            leftImage = Mat(height, width, CV_8UC1, cam.pImageMem);
        }
        else
        {
            rightImage = Mat(height, width, CV_8UC1, cam.pImageMem);
        }
        return 1;
    }
    return 0;
}
void IDSCamera::closeTracker()
{
    if (cameraL.bOpened)
    {
        int nRet = is_ExitCamera(cameraL.hCam);
        cameraL.bOpened = false;
    }
    if (cameraR.bOpened)
    {
        int nRet = is_ExitCamera(cameraR.hCam);
        cameraR.bOpened = false;
    }
}
bool IDSCamera::initTracker()
{
    int nCameraCount = 0;
    DWORD camDevIdL = -1;
    DWORD camDevIdR = -1;

    if (is_GetNumberOfCameras(&nCameraCount) == IS_SUCCESS)
    {
        PUEYE_CAMERA_LIST pucl = (PUEYE_CAMERA_LIST)new char[sizeof(DWORD) + nCameraCount * sizeof(UEYE_CAMERA_INFO)];
        pucl->dwCount = nCameraCount;

        if (is_GetCameraList(pucl) == IS_SUCCESS)
        {
            for (int i = 0; i < 2; i++)
            {
                    if(strcmp("4102648905", pucl->uci[i].SerNo))
                {
                    camDevIdL = pucl->uci[i].dwDeviceID;
                }
                else if (strcmp("4102648910", pucl->uci[i].SerNo))
                {
                    camDevIdR = pucl->uci[i].dwDeviceID;
                }
            }
        }
    }
    if (camDevIdL != -1 && camDevIdR != -1)
    {
        int nRet = 0;

        cameraL.hCam = ((HIDS)(camDevIdL | IS_USE_DEVICE_ID));
        cameraR.hCam = ((HIDS)(camDevIdR | IS_USE_DEVICE_ID));

        if (initCam(cameraL, "left") && initCam(cameraR, "right"))
        {
            initFlag = 1;
			return true;
        }
    }
	return false;
}

bool IDSCamera::isOpened()
{
	return initFlag;
}
//Mat ConvertMat(const UINT16* pBuffer, int nWidth, int nHeight)
//{
//    Mat img(nHeight, nWidth, CV_8UC1);
//    uchar* p_mat = img.data;//指向头指针
//
//    const UINT16* pBufferEnd = pBuffer + (nWidth * nHeight);//指向最后一个元素的指针
//    while (pBuffer < pBufferEnd)//16位最大值为65536
//    {
//        *p_mat = *pBuffer / 256.0;
//        p_mat++;
//        pBuffer++;
//    }
//    return img;
//}





