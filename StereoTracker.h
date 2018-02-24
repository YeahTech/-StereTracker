#ifndef WJCSTEREOTRACKER_H
#define WJCSTEREOTRACKER_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "opencv2/gpu/gpu.hpp"

using namespace cv;
using namespace std;

class StereoTracker
{
	
public:
    
	StereoTracker(void);
	virtual ~StereoTracker(void);
   
	bool readCalibFile(const char *);

public:
	
	cv::Mat xMapL;
	cv::Mat yMapL;
	cv::Mat xMapR;
	cv::Mat yMapR;

	cv::gpu::GpuMat xMapLGpu;
	cv::gpu::GpuMat yMapLGpu;
	cv::gpu::GpuMat xMapRGpu;
	cv::gpu::GpuMat yMapRGpu;

	 

    Mat cam_matrix_left;
    Mat cam_matrix_right;
    Mat dist_coef_left;
    Mat dist_coef_right;

	cv::Rect leftValidArea, rightValidArea; //左右视图的有效区域
	
	static Mat T;
    Mat R;           //rotation between cameras;

	static Mat P1;
	cv::Mat P2;
	static Mat Q;
	
	int calibFlag;
	
};


#endif
