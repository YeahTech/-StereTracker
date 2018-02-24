#include "StereoTracker.h"
#include <fstream>
#include <ctime>
using namespace cv;
using namespace std;
Mat StereoTracker::Q;
Mat StereoTracker::P1;
Mat StereoTracker::T;
StereoTracker::StereoTracker(void)
{
}


StereoTracker::~StereoTracker(void)
{

}

bool StereoTracker::readCalibFile(const char * file)
{
    FileStorage fs;
	if(fs.open(file,FileStorage::READ))
	{

		//之前读取文件太慢
		//fs["mapLx"] >> xMapL;
		//fs["mapLy"] >> yMapL;
		//fs["P1"] >> P1;
		//fs["mapRx"] >> xMapR;
		//fs["mapRy"] >> yMapR;
		//fs["P2"] >> P2;
		//fs["Q"] >> Q;
		////calibFlag = 1;
		//fs["T"] >> T;
		//fs["R"] >> R;
		//fs["cameraMatrix_left"] >> cam_matrix_left;
		//fs["cameraMatrix_right"] >> cam_matrix_right;
		//fs["distCoeffs_left"] >> dist_coef_left;
		//fs["distCoeffs_right"] >> dist_coef_right;

		//fs["leftValidArea"] >> leftValidArea;
		//fs["rightValidArea"] >> rightValidArea;

		//new calibfile
		cv::Size ImageSize;
		Mat Rmatlab;


		fs["ImageSize"] >> ImageSize;

		fs["cameraMatrix_left"] >> cam_matrix_left;
		fs["cameraMatrix_right"] >> cam_matrix_right;
		fs["distCoeffs_left"] >> dist_coef_left;
		fs["distCoeffs_right"] >> dist_coef_right;

		fs["R"] >> Rmatlab;
		cv::Rodrigues(Rmatlab,R);
		fs["T"] >> T;



		Mat R1, R2;
		/*  R1– 输出第一个相机的3x3矫正变幻 (旋转矩阵) .
		    R2– 输出第二个相机的3x3矫正变幻 (旋转矩阵) .
		    P1–在第一台相机的新的坐标系统(矫正过的)输出 3x4 的投影矩阵
		    P2–在第二台相机的新的坐标系统(矫正过的)输出 3x4 的投影矩阵
			Q–输出深度视差映射矩阵*/
	double alpha = -1.0;          // 双目校正效果的缩放系数，取值 0~1 或 -1
	std::cout<<"calculate..."<<std::endl;
	stereoRectify(
		cam_matrix_left, 
		dist_coef_left, 
		cam_matrix_right, 
		dist_coef_right, 
		ImageSize, 
		R, T, R1, R2, P1, P2, Q, 
		cv::CALIB_ZERO_DISPARITY,
		alpha, 
		ImageSize,
		&leftValidArea, &rightValidArea
		);

	initUndistortRectifyMap(cam_matrix_left, dist_coef_left, R1, P1,ImageSize, CV_32FC1, xMapL, yMapL);
	initUndistortRectifyMap(cam_matrix_right, dist_coef_right, R2, P2, ImageSize, CV_32FC1, xMapR, yMapR);

	return true;

	}
	else
		return false;
}

