#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include<sstream>
#include <Eigen/Eigen>

#include<opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "templatemanager.h"

using namespace std;
using namespace Eigen;


#define RESPONSE_ONLY 0
#define RESPONSE_SVM 1


class DetectedMarker
{
public:
	DetectedMarker()
	{
		name = "NULL";
		R.Zero();
		T.Zero();
	}

	double getYLength()
	{
		Point3f a = Coord3D[0];
		Point3f b = Coord3D[2];

		return (sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z)));
	}
	QString toQString()
	{
		QString info;
		info = QString(name);
		info.append("\n");

		info.append(QString("%1,%2,%3,%4\n").arg(R(0,0)).arg(R(0,1)).arg(R(0,2)).arg(T(0)) );

		info.append(QString("%1,%2,%3,%4\n").arg(R(1,0)).arg(R(1,1)).arg(R(1,2)).arg(T(1)) );

		info.append(QString("%1,%2,%3,%4\n").arg(R(2,0)).arg(R(2,1)).arg(R(2,2)).arg(T(2)) );

		//info.append(QString("%1,%2,%3,%4\n").arg(R(2,0)).arg(R(2,1)).arg(R(2,2)).arg(T(2)) );

		return info;
	}

	char *name;
    vector<cv::Point2f> Coord2L;
    vector<cv::Point2f> Coord2R;
    vector<cv::Point3f> Coord3D;

	Eigen::Matrix3d R;
	Eigen::Vector3d T;

};


class MarkerDetector
{
public:
	MarkerDetector();
	~MarkerDetector();

	CvSVM SVM;
	int if_SVM_moduleLoaded ;
	int SubPixel ;
	float fastthreshold ;
	int chess_threshold ;
    cv::Size zeroZone ;
    cv::TermCriteria criteria ;  //epsilon=0.001
    cv::Mat Merge_Image(cv::Mat&, cv::Mat&);//将两张图片拼接
	void LOAD_SVM(const char *);
    static double cal_distance(cv::Point3f,cv::Point3f);
    void paintResult(cv::Mat &leftImg, cv::Mat &left_Img, cv::Mat &rightImg,cv::Mat &right_Img,int ifmarker,vector<cv::Point2f> &corners2l, vector<cv::Point2f> &corners2r,vector<cv::Point3f> & Coordinate3D,vector<DetectedMarker> & Mymarker,vector<cv::Point3f> & probpoint, vector<DetectedMarker> & MyPattern,cv::Mat & Q,cv::Mat &T);
    bool distance_cal(cv::Point3f, cv::Point3f, float);
    void FASTDetector(cv::Mat &InputImage, vector<cv::Point2f> &output_vector,int SUBcorners_Windowsize, float threshold,int clustering);
    void FASTDetector(cv::Mat &InputImage, vector<cv::Point2f> &output_vector,int clustering);
    void RESPONSEDetector(cv::Mat &InputImage, vector<cv::Point2f> &output_vector, int SUBcorners_Windowsize, int clusterthreshold,float threshold,int ifSVM);
	void RESPONSEDetectorBitCal(cv::Mat &InputImage, vector<cv::Point2f> &output_vector, int SUBcorners_Windowsize, int clusterthreshold,float threshold,int ifSVM);
    void RP_SVM_Classify(cv::Mat &InputImage, cv::Mat rpImage,vector<cv::Point2f> &output_vector, int SUBcorners_Windowsize, int clusterthreshold, float threshold, int ifSVM);
    void ClassifySamples(vector<cv::Point2f> &input_vector, int imagewidth, vector<cv::Point2f> &output_vector, vector<cv::Point2f> &output_vector2);
    void MatchCorners(vector<cv::Point2f> &cornersLeft, vector<cv::Point2f> &cornersRight);
   	void MatchCornersFaster(vector<cv::Point2f>& cornersLeft, vector<cv::Point2f>& cornersRight);
	void Calculate_3D_coordinates(vector<cv::Point2f> &cornersLeft, vector<cv::Point2f> &cornersRight, vector<cv::Point3f>&Coordinate3D,cv::Mat Q);
    void Calculate_3D(vector<cv::Point2f> &cornersLeft, vector<cv::Point2f> &cornersRight, vector<cv::Point3f>&Coordinate3D, cv::Mat P1,cv::Mat T);
	//输入的marker识别
    void Marker_Recognize(vector<cv::Point2f>& corners2l, vector<cv::Point2f>& corners2r, vector<cv::Point3f> &corners3d, vector<InputMarker>&,vector<DetectedMarker>&);
	//输入的标定板识别
	bool readMarker(const char *path, const char *prefix, int num, vector<InputMarker> &markerInput);
    void Pattern_Recognize(vector<cv::Point2f>& corners2l, vector<cv::Point2f>& corners2r, vector<cv::Point3f> &corners3d, vector<DetectedMarker>& MyPattern);
    void probLocation(vector<DetectedMarker> &MyMarker,cv::Point3f prob,vector<cv::Point3f> &prob_point);
    void probCalibration(vector<DetectedMarker> &MyMarker, vector<cv::Point3f> &prob_point);

	void RESPONSEDetectorFaster(cv::Mat& InputImage,
		vector<cv::Point2f>& output_vector,
		int SUBcorners_Windowsize,
		int candidateWindowsize,
		float responseThreshold);


	void fiducialRegistration(std::vector<InputMarker> & markerInput, std::vector<DetectedMarker> & detectedMarker);

	void drawDetectedMarkers(cv::Mat& inputImg,vector<DetectedMarker> Maker_Detected,char leftORright,bool show_T);

	void drawVecPoints(cv::Mat & inputImg,vector<cv::Point2f> corners,vector<cv::Point3f> Coordinate3D, bool noteFlag);

	void drawMatchCorners(cv::Mat & inputImg,vector<cv::Point2f> cornersLeft,vector<cv::Point2f> cornersRight);
	
	int Marker_LikeDetect(vector<cv::Point2f>& corners2l, vector<cv::Point2f>& corners2r, vector<cv::Point3f>& corners3d, vector<DetectedMarker>& DetectedLikeMarkers);
};
