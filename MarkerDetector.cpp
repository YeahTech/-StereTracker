#include "MarkerDetector.h"
#include <cstdint>
#include<omp.h>

int clustering;
cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create("SURF");

bool _EqPredicate(const cv::KeyPoint& a, const cv::KeyPoint& b)
{
    return ((b.pt.y - a.pt.y) * (b.pt.y - a.pt.y) + (b.pt.x - a.pt.x) * (b.pt.x - a.pt.x) < clustering * clustering);
}

bool _isNeibour(const cv::Point2i& a, const cv::Point2i& b)
{
	if(abs((a-b).x) < 7 && abs((a-b).y) <7)
		return true;
	else
		return false;
}

MarkerDetector::MarkerDetector()
{
    cv::initModule_nonfree();
    if_SVM_moduleLoaded = 0;
    SubPixel = 4;
    fastthreshold = 30;
    chess_threshold = 150;
    zeroZone = cv::Size(-1, -1);
    criteria = cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER,
                                                 40,     // maxCount=10
                                                 0.001); // epsilon=0.001
}


MarkerDetector::~MarkerDetector() {}

void MarkerDetector::FASTDetector(cv::Mat& InputImage, vector<cv::Point2f>& output_vector, int subpixel, float _threshold, int clusterthreshold)
{
	cv::Mat imgRGB = InputImage.clone();
	//cvtColor(InputImage,InputImage,CV_BGR2GRAY);

    output_vector.clear();
    clustering = clusterthreshold;
    SubPixel = subpixel;
    if(if_SVM_moduleLoaded == 0)
        return;
    vector<cv::KeyPoint> Original;
    vector<cv::KeyPoint> CandidatekeyPoints;
    cv::FastFeatureDetector fast(_threshold); // 检测的阈值
    fast.detect(InputImage, Original);
    cv::Mat result;
    vector<int> labels;
    int labelnum = 0;
    int sampleCount = Original.size();
    float filter[50000][3] = { { 0 } };
    if(sampleCount > 0)
    {
        long count = partition(Original, labels, _EqPredicate); //聚类
        vector<cv::KeyPoint>::iterator it = Original.begin();
        for(int i = 0; i < sampleCount; i++)
        {
            int Idx = labels.at(i);
            if(filter[Idx][0] == 0)
                labelnum++;                     //统计类别数
            filter[Idx][0]++;                   //点的个数
            filter[Idx][1] += Original[i].pt.x; //把同一类中的点坐标相加
            filter[Idx][2] += Original[i].pt.y;
            //++it;
            // it = CandidatekeyPoints.erase(it);//删除同一类的点0
        }
        for(int i = 0; i < labelnum; i++)
        {
            Original[i].pt.x = filter[i][1] / filter[i][0];
            Original[i].pt.y = filter[i][2] / filter[i][0];
            CandidatekeyPoints.push_back(Original[i]);
        }
        vector<cv::Point2f> subCorners;
        for(int i = 0; i < CandidatekeyPoints.size(); i++)
        {
            cv::Point2f X;
            X.x = CandidatekeyPoints[i].pt.x;
            X.y = CandidatekeyPoints[i].pt.y;
            subCorners.push_back(X);
        }
        cv::Size winSize = cv::Size(SubPixel, SubPixel);

        cornerSubPix(InputImage, subCorners, winSize, zeroZone, criteria); //提取亚像素角点
        for(int i = 0; i < CandidatekeyPoints.size(); i++)
        {
            CandidatekeyPoints[i].pt.x = subCorners[i].x;
            CandidatekeyPoints[i].pt.y = subCorners[i].y;
        }
        extractor->compute(InputImage, CandidatekeyPoints, result); //计算surf特征
        cv::Mat predict_result;
        SVM.predict(result, predict_result); //返回类标
        vector<int> clusteringlabel;
        int filter[50000] = { 0 };
        long count2 = partition(CandidatekeyPoints, clusteringlabel, _EqPredicate); //再聚类
        for(int i = 0; i < CandidatekeyPoints.size(); i++)
        {
            if(predict_result.at<float>(i, 0) == 1.0)
            {
                if(filter[clusteringlabel[i]] == 0)
                {
                    cv::Point2f X;
                    X.x = CandidatekeyPoints[i].pt.x;
                    X.y = CandidatekeyPoints[i].pt.y;
                    output_vector.push_back(X);

					cv::circle(imgRGB,cv::Point(X.x,X.y),2,cv::Scalar(0,0,255),2);
					cv::imshow("cor",imgRGB);


                    filter[clusteringlabel[i]]++; //这句话注释掉之后就会有很多重合的点
                }
            }
        }
    }
}


void MarkerDetector::FASTDetector(cv::Mat& InputImage, vector<cv::Point2f>& output_vector, int clusterthreshold)
{
	//cv::Mat imgRGB = InputImage.clone();
	//cvtColor(InputImage,InputImage,CV_BGR2GRAY);

    output_vector.clear();
    clustering = clusterthreshold;
    if(if_SVM_moduleLoaded == 0)
        return;
    vector<cv::KeyPoint> Original;
    vector<cv::KeyPoint> CandidatekeyPoints;
    cv::FastFeatureDetector fast(fastthreshold); // 检测的阈值
    fast.detect(InputImage, Original);
    cv::Mat result;
    vector<int> labels;
    int labelnum = 0;
    int sampleCount = Original.size();
    float filter[50000][3] = { { 0 } };
    if(sampleCount > 0)
    {
        long count = partition(Original, labels, _EqPredicate); //聚类
        vector<cv::KeyPoint>::iterator it = Original.begin();

        for(int i = 0; i < sampleCount; i++)
        {
            int Idx = labels.at(i);
            if(filter[Idx][0] == 0)
                labelnum++;                     //统计类别数
            filter[Idx][0]++;                   //点的个数
            filter[Idx][1] += Original[i].pt.x; //把同一类中的点坐标相加
            filter[Idx][2] += Original[i].pt.y;
            //++it;
            // it = CandidatekeyPoints.erase(it);//删除同一类的点0
        }
        for(int i = 0; i < labelnum; i++)
        {
            Original[i].pt.x = filter[i][1] / filter[i][0];
            Original[i].pt.y = filter[i][2] / filter[i][0];
            CandidatekeyPoints.push_back(Original[i]);
        }
        vector<cv::Point2f> subCorners;
        for(int i = 0; i < CandidatekeyPoints.size(); i++)
        {
            cv::Point2f X;
            X.x = CandidatekeyPoints[i].pt.x;
            X.y = CandidatekeyPoints[i].pt.y;
            subCorners.push_back(X);
        }
        cv::Size winSize = cv::Size(SubPixel, SubPixel);

        cornerSubPix(InputImage, subCorners, winSize, zeroZone, criteria); //提取亚像素角点
        for(int i = 0; i < CandidatekeyPoints.size(); i++)
        {
            CandidatekeyPoints[i].pt.x = subCorners[i].x;
            CandidatekeyPoints[i].pt.y = subCorners[i].y;
        }
        extractor->compute(InputImage, CandidatekeyPoints, result); //计算surf特征
        cv::Mat predict_result;
        SVM.predict(result, predict_result); //返回类标
        vector<int> clusteringlabel;
        int filter[50000] = { 0 };
        long count2 = partition(CandidatekeyPoints, clusteringlabel, _EqPredicate); //再聚类
        for(int i = 0; i < CandidatekeyPoints.size(); i++)
        {
            if(predict_result.at<float>(i, 0) == 1.0)
            {
                if(filter[clusteringlabel[i]] == 0)
                {
                    cv::Point2f X;
                    X.x = CandidatekeyPoints[i].pt.x;
                    X.y = CandidatekeyPoints[i].pt.y;
                    output_vector.push_back(X);

					/*cv::circle(imgRGB,cv::Point(X.x,X.y),2,cv::Scalar(0,0,255),2);
					cv::imshow("cor",imgRGB);*/

                    filter[clusteringlabel[i]]++; //这句话注释掉之后就会有很多重合的点
                }
            }
        }
    }
}


void MarkerDetector::RESPONSEDetector(cv::Mat& InputImage,
                                      vector<cv::Point2f>& output_vector,
                                      int SUBcorners_Windowsize,
                                      int clusterthreshold,
                                      float threshold,
                                      int ifSVM)
{
	//cvtColor(InputImage,InputImage,CV_BGR2GRAY);


    output_vector.clear();
    clustering = clusterthreshold;
    SubPixel = SUBcorners_Windowsize;
    chess_threshold = threshold;
    vector<cv::KeyPoint> Original;
    int w = InputImage.cols;
    int h = InputImage.rows;
    int16_t* response = new int16_t[w * h];
    uint8_t* image = InputImage.data;
    int x, y;

	omp_set_nested(1);//设置支持嵌套并行
	#pragma omp parallel for

    for(y = 7; y < h - 7; y++)
	{
		#pragma omp parallel for

        for(x = 7; x < w - 7; x++)
        {
            unsigned int offset = x + y * w;

            uint8_t circular_sample[16];

            circular_sample[2] = image[offset - 2 - 5 * w];
            circular_sample[1] = image[offset - 5 * w];
            circular_sample[0] = image[offset + 2 - 5 * w];
            circular_sample[8] = image[offset - 2 + 5 * w];
            circular_sample[9] = image[offset + 5 * w];
            circular_sample[10] = image[offset + 2 + 5 * w];
            circular_sample[3] = image[offset - 4 - 4 * w];
            circular_sample[15] = image[offset + 4 - 4 * w];
            circular_sample[7] = image[offset - 4 + 4 * w];
            circular_sample[11] = image[offset + 4 + 4 * w];
            circular_sample[4] = image[offset - 5 - 2 * w];
            circular_sample[14] = image[offset + 5 - 2 * w];
            circular_sample[6] = image[offset - 5 + 2 * w];
            circular_sample[12] = image[offset + 5 + 2 * w];
            circular_sample[5] = image[offset - 5];
            circular_sample[13] = image[offset + 5];

            uint16_t local_mean = (image[offset - 1] + image[offset] + image[offset + 1] +
                                   image[offset + w] + image[offset - w]) *
                                  16 / 5;
            uint16_t sum_response = 0;
            uint16_t diff_response = 0;
            uint16_t mean = 0;

            int sub_idx;
            for(sub_idx = 0; sub_idx < 4; ++sub_idx)
            {
                uint8_t a = circular_sample[sub_idx];
                uint8_t b = circular_sample[sub_idx + 4];
                uint8_t c = circular_sample[sub_idx + 8];
                uint8_t d = circular_sample[sub_idx + 12];

                sum_response += abs(a - b + c - d);
                diff_response += abs(a - c) + abs(b - d);
                mean += a + b + c + d;
            }

            response[offset] = sum_response - diff_response - abs(mean - local_mean);
        }
	}
    // non-maximum suppression

	#pragma omp parallel for

    for(y = 7; y < h - 7; y++)
	{
		#pragma omp parallel for

        for(x = 7; x < w - 7; x++)
        {
            unsigned int offset = x + y * w;
            if(response[offset] <= 0)
            {
                response[offset] = 0;
            }
            else
            {
                for(int j = -5; j <= 5; ++j)
                    for(int i = -5; i <= 5; ++i)
                    {
                        if(response[offset] < response[offset + i + j * w])
                        {
                            response[offset] = 0;
                            break;
                        }
                    }
            }
        }
	}

	//#pragma omp parallel for

    for(int i = 0; i < w * h; i++)
    {
        if(response[i] > chess_threshold)
        {
            cv::Point2f X;
            X.x = i % w;
            X.y = i / w;
            output_vector.push_back(X);

            cv::KeyPoint Y;
            Y.pt.x = i % w;
            Y.pt.y = i / w;

            Y.size = 7.0;
            Y.angle = -1;
            Y.octave = 0;
            Y.class_id = -1;
            Y.response = response[i];
            Original.push_back(Y);
        }
    }
    cv::Size winSize = cv::Size(SubPixel, SubPixel);
    vector<int> newlabel;
	

    //if(ifSVM == RESPONSE_SVM)
    //{
    //    output_vector.clear();
    //    cv::Mat result;
    //    if(Original.size() > 0)
    //    {
    //        extractor->compute(InputImage, Original, result); //计算surf特征
    //        cv::Mat predict_result;
    //        SVM.predict(result, predict_result); //返回类标
    //        vector<int> clusteringlabel;
    //        long count2 = partition(Original, clusteringlabel, _EqPredicate); //再聚类
    //        for(int i = 0; i < Original.size(); i++)
    //        {
    //            if(predict_result.at<float>(i, 0) >= 0)
    //            {
    //                cv::Point2f X;
    //                X.x = Original[i].pt.x;
    //                X.y = Original[i].pt.y;
    //                output_vector.push_back(X);
    //                newlabel.push_back(clusteringlabel[i]);
    //            }
    //        }
    //    }
    //}
    if(output_vector.size() > 0)
        cornerSubPix(InputImage, output_vector, winSize, zeroZone,criteria); //提取亚像素角点
    // delete image;
    //int filter[50000] = { 0 };  by yaoxinghua
    //if(ifSVM == RESPONSE_SVM)
    //{
    //    vector<cv::Point2f>::iterator it = output_vector.begin();
    //    vector<int>::iterator itn = newlabel.begin();
    //    int k = output_vector.size();
    //    for(int i = 0, j = 0; i < k; i++)
    //    {
    //        if(filter[newlabel[j]] > 0)
    //        {
    //            it = output_vector.erase(it); //删除同一类的点0
    //            itn = newlabel.erase(itn);    //删除同一类的点0
    //        }
    //        else
    //        {
    //            it++;
    //            itn++;
    //            filter[newlabel[j]]++;
    //            j++;
    //        }
    //    }
    //}

    delete response;
}
void MarkerDetector::RESPONSEDetectorBitCal(cv::Mat& InputImage,
	vector<cv::Point2f>& output_vector,
	int SUBcorners_Windowsize,
	int clusterthreshold,
	float threshold,
	int ifSVM)
{
	//cvtColor(InputImage,InputImage,CV_BGR2GRAY);


	output_vector.clear();
	clustering = clusterthreshold;
	SubPixel = SUBcorners_Windowsize;
	chess_threshold = threshold;
	//vector<cv::KeyPoint> Original;
	int w = InputImage.cols;
	int h = InputImage.rows;
	int16_t* response = new int16_t[w * h];
	uint8_t* image = InputImage.data;
	int x, y;

	int w5 = (w<<2)-w;
	int w4 = (w<<2);
	int w2 = (w<<1);

	omp_set_nested(1);//设置支持嵌套并行
#pragma omp parallel for

	for(y = 7; y < h - 7; y++)
	{

		for(x = 7; x < w - 7; x++)
		{
			unsigned int offset = x + y * w;

			uint8_t circular_sample[16];

			circular_sample[2] = image[offset - 2 - w5];
			circular_sample[1] = image[offset - w5];
			circular_sample[0] = image[offset + 2 - w5];
			circular_sample[8] = image[offset - 2 + w5];
			circular_sample[9] = image[offset + w5];
			circular_sample[10] = image[offset + 2 + w5];
			circular_sample[3] = image[offset - 4 - w4];
			circular_sample[15] = image[offset + 4 - w4];
			circular_sample[7] = image[offset - 4 + w4];
			circular_sample[11] = image[offset + 4 + w4];
			circular_sample[4] = image[offset - 5 - w2];
			circular_sample[14] = image[offset + 5 - w2];
			circular_sample[6] = image[offset - 5 + w2];
			circular_sample[12] = image[offset + 5 + w2];
			circular_sample[5] = image[offset - 5];
			circular_sample[13] = image[offset + 5];

			uint16_t local_mean = ((image[offset - 1] + image[offset] + image[offset + 1] +
				image[offset + w] + (image[offset - w])) <<4)/5;
			uint16_t sum_response = 0;
			uint16_t diff_response = 0;
			uint16_t mean = 0;

			int sub_idx;
			for(sub_idx = 0; sub_idx < 4; ++sub_idx)
			{
				uint8_t a = circular_sample[sub_idx];
				uint8_t b = circular_sample[sub_idx + 4];
				uint8_t c = circular_sample[sub_idx + 8];
				uint8_t d = circular_sample[sub_idx + 12];

				sum_response += abs(a - b + c - d);
				diff_response += abs(a - c) + abs(b - d);
				mean += a + b + c + d;
			}

			response[offset] = sum_response - diff_response - abs(mean - local_mean);
		}
	}
	// non-maximum suppression

#pragma omp parallel for

	for(y = 7; y < h - 7; y++)
	{
#pragma omp parallel for

		for(x = 7; x < w - 7; x++)
		{
			unsigned int offset = x + y * w;
			if(response[offset] <= 0)
			{
				response[offset] = 0;
			}
			else
			{
				for(int j = -5; j <= 5; ++j)
					for(int i = -5; i <= 5; ++i)
					{
						if(response[offset] < response[offset + i + j * w])
						{
							response[offset] = 0;
							break;
						}
					}
			}
		}
	}

	//#pragma omp parallel for

	for(int i = 0,iend = w * h ; i <iend; i++)
	{
		if(response[i] > chess_threshold)
		{
			cv::Point2f X;
			X.x = i % w;
			X.y = i / w;
			output_vector.push_back(X);

			/*cv::KeyPoint Y;
			Y.pt.x = i % w;
			Y.pt.y = i / w;

			Y.size = 7.0;
			Y.angle = -1;
			Y.octave = 0;
			Y.class_id = -1;
			Y.response = response[i];
			Original.push_back(Y);*/
		}
	}
	cv::Size winSize = cv::Size(SubPixel, SubPixel);
	vector<int> newlabel;


	//if(ifSVM == RESPONSE_SVM)
	//{
	//    output_vector.clear();
	//    cv::Mat result;
	//    if(Original.size() > 0)
	//    {
	//        extractor->compute(InputImage, Original, result); //计算surf特征
	//        cv::Mat predict_result;
	//        SVM.predict(result, predict_result); //返回类标
	//        vector<int> clusteringlabel;
	//        long count2 = partition(Original, clusteringlabel, _EqPredicate); //再聚类
	//        for(int i = 0; i < Original.size(); i++)
	//        {
	//            if(predict_result.at<float>(i, 0) >= 0)
	//            {
	//                cv::Point2f X;
	//                X.x = Original[i].pt.x;
	//                X.y = Original[i].pt.y;
	//                output_vector.push_back(X);
	//                newlabel.push_back(clusteringlabel[i]);
	//            }
	//        }
	//    }
	//}
	if(output_vector.size() > 0)
		cornerSubPix(InputImage, output_vector, winSize, zeroZone,criteria); //提取亚像素角点
	
	// delete image;
	//int filter[50000] = { 0 };  by yaoxinghua
	//if(ifSVM == RESPONSE_SVM)
	//{
	//    vector<cv::Point2f>::iterator it = output_vector.begin();
	//    vector<int>::iterator itn = newlabel.begin();
	//    int k = output_vector.size();
	//    for(int i = 0, j = 0; i < k; i++)
	//    {
	//        if(filter[newlabel[j]] > 0)
	//        {
	//            it = output_vector.erase(it); //删除同一类的点0
	//            itn = newlabel.erase(itn);    //删除同一类的点0
	//        }
	//        else
	//        {
	//            it++;
	//            itn++;
	//            filter[newlabel[j]]++;
	//            j++;
	//        }
	//    }
	//}

	delete response;
}

void MarkerDetector::RESPONSEDetectorFaster(cv::Mat& InputImage,
	vector<cv::Point2f>& output_vector,
	int SUBcorners_Windowsize,
	int candidateWindowsize,
	float responseThreshold)
{
	//step1:Extraction Fast Point as candidate
	vector<cv::KeyPoint> Original;
	cv::FastFeatureDetector fast(30); 
	fast.detect(InputImage, Original);

	output_vector.clear();

	ushort w = InputImage.cols;
	ushort h = InputImage.rows;

	uint8_t* image = InputImage.data;
	int x, y;

	short w5 = (w<<2)+w;
	short w4 = (w<<2);
	short w2 = (w<<1);

	//针对每个候选点，计算其候选窗口内的最大response点和响应值，记录在maxResponseVec和maxResponsePointsVec
	vector<cv::Point2i>  maxResponsePointsVec;
	vector<int> sameVec;
	vector<int>  maxResponseVec;

	for (int i = 0,iend = Original.size(); i < iend; i++)
	{
		cv::Point2i maxResponsePoint;
		int maxResponse = 0;

		for(y = Original[i].pt.y-candidateWindowsize; y < Original[i].pt.y+candidateWindowsize; y++)
		{
			if (y<5 || y>h-5)
				continue;

			for(x = Original[i].pt.x-candidateWindowsize; x < Original[i].pt.x+candidateWindowsize; x++)
			{
				if( x < 5 || x>w-5)
					continue;

				unsigned int offset = x + y * w;

				uint8_t circular_sample[16];

				circular_sample[2] = image[offset - 2 - w5];
				circular_sample[1] = image[offset - w5];
				circular_sample[0] = image[offset + 2 - w5];
				circular_sample[8] = image[offset - 2 + w5];
				circular_sample[9] = image[offset + w5];
				circular_sample[10] = image[offset + 2 + w5];
				circular_sample[3] = image[offset - 4 - w4];
				circular_sample[15] = image[offset + 4 - w4];
				circular_sample[7] = image[offset - 4 + w4];
				circular_sample[11] = image[offset + 4 + w4];
				circular_sample[4] = image[offset - 5 - w2];
				circular_sample[14] = image[offset + 5 - w2];
				circular_sample[6] = image[offset - 5 + w2];
				circular_sample[12] = image[offset + 5 + w2];
				circular_sample[5] = image[offset - 5];
				circular_sample[13] = image[offset + 5];

				uint16_t local_mean = ((image[offset - 1] + image[offset] + image[offset + 1] +
					image[offset + w] + (image[offset - w])) <<4 )/5;
				uint16_t sum_response = 0;
				uint16_t diff_response = 0;
				uint16_t mean = 0;

				//cal SR DR MR
				for(short sub_idx = 0; sub_idx < 4; ++sub_idx)
				{
					uint8_t a = circular_sample[sub_idx];
					uint8_t b = circular_sample[sub_idx + 4];
					uint8_t c = circular_sample[sub_idx + 8];
					uint8_t d = circular_sample[sub_idx + 12];

					sum_response += abs(a - b + c - d);
					diff_response += abs(a - c) + abs(b - d);
					mean += a + b + c + d;
				}

				//response[offset] = sum_response - diff_response - abs(mean - local_mean);
				int R = sum_response - diff_response - abs(mean - local_mean);
				if(R > maxResponse)
				{
					maxResponse = R;
					maxResponsePoint = cv::Point2f(x,y);

				}

			}
		}

		//对最大响应点阈值判断
		if (maxResponse > responseThreshold)
		{
			int nRet = std::count(maxResponsePointsVec.begin(),maxResponsePointsVec.end(),maxResponsePoint);
			if (nRet==0)
			{
				maxResponsePointsVec.push_back(maxResponsePoint);
				maxResponseVec.push_back(maxResponse);
			}
		}

	}

	//去除临近点
	if(maxResponsePointsVec.size()> 0)
	{
		vector<int> labels;
		int labelCount = partition(maxResponsePointsVec, labels, _isNeibour); //聚类
		if (labelCount !=maxResponsePointsVec.size() )
		{
			for (int currlabel = 0;currlabel < labelCount;currlabel++ )
			{
				cv::Point2f pointTemp(0,0);
				int numbTemp = 0;
				for (int i = 0; i <labels.size();i++)
				{
					if(labels[i] == currlabel)
					{
						pointTemp += cv::Point2f(maxResponsePointsVec[i]);
						numbTemp++;
					}
				}
				output_vector.push_back(cv::Point2f(pointTemp.x/numbTemp,pointTemp.y/numbTemp));
			}
		}
	}

	//subpix
	if(output_vector.size() > 0)
		cornerSubPix(InputImage, output_vector, cv::Size(SUBcorners_Windowsize, SUBcorners_Windowsize), zeroZone,criteria); //提取亚像素角点
}

void MarkerDetector::RP_SVM_Classify(cv::Mat& InputImage,
                                     cv::Mat rpImage,
                                     vector<cv::Point2f>& output_vector,
                                     int SUBcorners_Windowsize,
                                     int clusterthreshold,
                                     float threshold,
                                     int ifSVM)
{
    vector<cv::KeyPoint> Original;
    chess_threshold = threshold;
    for(int i = 0; i < rpImage.rows; i++)
    {
        float* data = rpImage.ptr<float>(i);
        for(int j = 0; j < rpImage.cols; j++)
        {
            if(data[j] > chess_threshold)
            {
                cv::Point2f X;
                X.x = j;
                X.y = i;
                output_vector.push_back(X);

                cv::KeyPoint Y;
                Y.pt.x = j;
                Y.pt.y = i;

                Y.size = 7.0;
                Y.angle = -1;
                Y.octave = 0;
                Y.class_id = -1;
                Y.response = data[j];
                Original.push_back(Y);
            }
        }
    }
    cv::Size winSize = cv::Size(SubPixel, SubPixel);
    vector<int> newlabel;
    if(ifSVM == RESPONSE_SVM)
    {
        output_vector.clear();
        cv::Mat result;
        if(Original.size() > 0)
        {
            extractor->compute(InputImage, Original, result); //计算surf特征
            cv::Mat predict_result;
            SVM.predict(result, predict_result); //返回类标
            vector<int> clusteringlabel;
            long count2 = partition(Original, clusteringlabel, _EqPredicate); //再聚类
            for(int i = 0; i < Original.size(); i++)
            {
                if(predict_result.at<float>(i, 0) >= 0)
                {
                    // if (filter[clusteringlabel[i]] == 0)
                    //{
                    cv::Point2f X;
                    X.x = Original[i].pt.x;
                    X.y = Original[i].pt.y;
                    output_vector.push_back(X);
                    newlabel.push_back(clusteringlabel[i]);
                    // filter[clusteringlabel[i]]++;//这句话注释掉之后就会有很多重合的点
                    //}
                }
            }
        }
        //}
    }
    if(output_vector.size() > 0)
        cornerSubPix(InputImage, output_vector, winSize, zeroZone, criteria); //提取亚像素角点
                                                                              // delete image;
    int filter[50000] = { 0 };
    if(ifSVM == RESPONSE_SVM)
    {
        vector<cv::Point2f>::iterator it = output_vector.begin();
        vector<int>::iterator itn = newlabel.begin();
        int k = output_vector.size();
        for(int i = 0, j = 0; i < k; i++)
        {
            if(filter[newlabel[j]] > 0)
            {
                it = output_vector.erase(it); //删除同一类的点0
                itn = newlabel.erase(itn);    //删除同一类的点0
            }
            else
            {
                it++;
                itn++;
                filter[newlabel[j]]++;
                j++;
            }
        }
    }
}

//将两张图合并为一张图进行检测
cv::Mat MarkerDetector::Merge_Image(cv::Mat& left, cv::Mat& right)
{
    cv::Size size(left.cols + right.cols, MAX(left.rows, right.rows));
    cv::Mat img_merge;
    cv::Mat outImg_left, outImg_right;

	img_merge.create(size, left.type());


    img_merge = cv::Scalar::all(0);
    outImg_left = img_merge(cv::Rect(0, 0, left.cols, left.rows));
    outImg_right = img_merge(cv::Rect(left.cols, 0, right.cols, right.rows));
    left.copyTo(outImg_left);
    right.copyTo(outImg_right);
    return img_merge;
}


void MarkerDetector::ClassifySamples(vector<cv::Point2f>& input_vector,
                                     int imagewidth,
                                     vector<cv::Point2f>& output_vector,
                                     vector<cv::Point2f>& output_vector2)
{
    for(int i = 0; i < input_vector.size(); i++) //将原来合在一起的点分开
    {
        if(input_vector[i].x >= imagewidth)
        {
            input_vector[i].x = input_vector[i].x - imagewidth;
            output_vector2.push_back(input_vector[i]);
        }
        else
        {
            output_vector.push_back(input_vector[i]);
        }
    }
}


void MarkerDetector::MatchCorners(vector<cv::Point2f>& cornersLeft, vector<cv::Point2f>& cornersRight)
{
    if(cornersLeft.empty() || cornersRight.empty())
    {
        cornersLeft.clear();
        cornersRight.clear();
    }
    else
    {
        std::vector<cv::Point2f> v1(cornersLeft.size());
        v1.swap(cornersLeft);
        std::vector<cv::Point2f> v2(cornersRight.size());
        v2.swap(cornersRight);
        cornersLeft.clear();
        cornersRight.clear();

        for(int i = 0; i < v1.size(); i++)
        {
            int _threshold = 3;
            float min_h = 15;

            for(int k = 0; k < v1.size(); k++) //确定阈值
            {
                if(k == i)
                    continue;
                fabs(v1[i].y - v1[k].y) < min_h ? (min_h = fabs(v1[i].y - v1[k].y)) : (min_h = min_h);
            }
            if(min_h < 6)
                _threshold = 20;
            int sameLevel = 0;
            int whichSameLevel[1000] = { 0 };
            for(int k = 0; k < v1.size(); k++) //寻找左图中是否有处于同一水平线上的点
            {
                if(k == i)
                    continue;
                if(v1[i].y - v1[k].y > _threshold || v1[i].y - v1[k].y < 0 - _threshold)
                    continue;
                whichSameLevel[sameLevel++] = k;
            }
            int isMatched = 0;
            int whichMatched[1000] = { 0 };
            for(int j = 0; j < v2.size(); j++) //寻找右图中是否有处于同一水平线上的点
            {
                if(v1[i].y - v2[j].y > _threshold || v1[i].y - v2[j].y < 0 - _threshold)
                    continue;
                if(v1[i].x < v2[j].x)
                    continue;
                whichMatched[isMatched++] = j;
            }
            if(isMatched == 0) //没有匹配点
                continue;
            if(sameLevel >= 0 && isMatched >= 1)
            {
                int rankingX = 0;                  //左图中点横坐标排序
                for(int m = 0; m < sameLevel; m++) //获取当前点在左图中同一水平线上的点的排序
                {
                    if(v1[whichSameLevel[m]].x < v1[i].x)
                        rankingX++;
                }
                for(int m = 0; m < isMatched - 1; m++) //将右图中的点排序
                {
                    for(int n = m + 1; n < isMatched; n++)
                    {
                        if(v2[whichMatched[m]].x > v2[whichMatched[n]].x)
                        {
                            int k = whichMatched[m];
                            whichMatched[m] = whichMatched[n];
                            whichMatched[n] = k;
                        }
                    }
                }
                if(sameLevel >= isMatched)
                {
                    if(sameLevel - isMatched >= rankingX)
                        continue;
                    cornersLeft.push_back(v1[i]);
                    cornersRight.push_back(v2[whichMatched[rankingX + isMatched - 1 - sameLevel]]);
                }
                else
                {
                    cornersLeft.push_back(v1[i]);
                    cornersRight.push_back(v2[whichMatched[rankingX]]);
                }
            }
        }
    }
}

void MarkerDetector::MatchCornersFaster(vector<cv::Point2f>& cornersLeft, vector<cv::Point2f>& cornersRight)
{
	if(cornersLeft.empty() || cornersRight.empty())
	{
		cornersLeft.clear();
		cornersRight.clear();
	}
	else
	{
		std::vector<cv::Point2f> v1(cornersLeft.size());
		v1.swap(cornersLeft);
		std::vector<cv::Point2f> v2(cornersRight.size());
		v2.swap(cornersRight);
		cornersLeft.clear();
		cornersRight.clear();

		for(int i = 0; i < v1.size(); i++)
		{
			int _threshold = 3;
			float min_h = 15;

			for(int k = 0; k < v1.size(); k++) //确定阈值
			{
				if(k == i)
					continue;
				fabs(v1[i].y - v1[k].y) < min_h ? (min_h = fabs(v1[i].y - v1[k].y)) : (min_h = min_h);
			}

			/*if(min_h < 6)
				_threshold = 20;*/
			int sameLevel = 0;
			int whichSameLevel[1000] = { 0 };
			for(int k = 0; k < v1.size(); k++) //寻找左图中是否有处于同一水平线上的点
			{
				if(k == i)
					continue;
				if(fabs(v1[i].y - v1[k].y) > _threshold)
					continue;
				whichSameLevel[sameLevel++] = k;
			}
			int isMatched = 0;
			int whichMatched[1000] = { 0 };
			for(int j = 0; j < v2.size(); j++) //寻找右图中是否有处于同一水平线上的点
			{
				if(fabs(v1[i].y - v2[j].y) > _threshold)
					continue;
				if(v1[i].x < v2[j].x)
					continue;
				whichMatched[isMatched++] = j;
			}
			if(isMatched == 0) //没有匹配点
				continue;
			if(sameLevel >= 0 && isMatched >= 1)
			{
				int rankingX = 0;                  //左图中点横坐标排序
				for(int m = 0; m < sameLevel; m++) //获取当前点在左图中同一水平线上的点的排序
				{
					if(v1[whichSameLevel[m]].x < v1[i].x)
						rankingX++;
				}
				for(int m = 0; m < isMatched - 1; m++) //将右图中的点排序
				{
					for(int n = m + 1; n < isMatched; n++)
					{
						if(v2[whichMatched[m]].x > v2[whichMatched[n]].x)
						{
							int k = whichMatched[m];
							whichMatched[m] = whichMatched[n];
							whichMatched[n] = k;
						}
					}
				}
				if(sameLevel >= isMatched)
				{
					if(sameLevel - isMatched >= rankingX)
						continue;
					cornersLeft.push_back(v1[i]);
					cornersRight.push_back(v2[whichMatched[rankingX + isMatched - 1 - sameLevel]]);
				}
				else
				{
					cornersLeft.push_back(v1[i]);
					cornersRight.push_back(v2[whichMatched[rankingX]]);
				}
			}
		}
	}
}

//计算三维坐标
void MarkerDetector::Calculate_3D_coordinates(vector<cv::Point2f>& cornersLeft,
                                              vector<cv::Point2f>& cornersRight,
                                              vector<cv::Point3f>& Coordinate3D,
                                              cv::Mat Q)
{
    for(int i = 0; i < cornersLeft.size(); i++)
    {
        double a[4];
        a[0] = cornersLeft[i].x;
        a[1] = cornersLeft[i].y;
        a[2] = cornersLeft[i].x - cornersRight[i].x;
        a[3] = 1;
        double b[4];
        for(int j = 0; j < 4; j++)
        {
            b[j] = 0;
            for(int k = 0; k < 4; k++)
            {
                b[j] = b[j] + Q.at<double>(j, k) * a[k];
            }
        }
        cv::Point3f X;
        X.x = b[0] / b[3];
        X.y = b[1] / b[3];
        X.z = b[2] / b[3];
        Coordinate3D.push_back(X);
    }
}
void MarkerDetector::Calculate_3D(vector<cv::Point2f>& cornersLeft,
                                  vector<cv::Point2f>& cornersRight,
                                  vector<cv::Point3f>& Coordinate3D,
                                  cv::Mat P1,
                                  cv::Mat T)
{
	Coordinate3D.clear();

    double cx = P1.at<double>(0, 2);
    double cy = P1.at<double>(1, 2);
    double f = P1.at<double>(0, 0);
    double b = -T.at<double>(0, 0);
    for(int i = 0; i < cornersLeft.size(); i++)
    {
        cv::Point3f X;
        X.z = f * b / (cornersLeft[i].x - cornersRight[i].x);
        X.x = X.z * (cornersLeft[i].x + cornersRight[i].x - 2 * cx) / (2 * f);
        X.y = X.z * (cornersLeft[i].y + cornersRight[i].y - 2 * cy) / (2 * f);

        Coordinate3D.push_back(X);
    }
}


// Marker检测
void MarkerDetector::Marker_Recognize(vector<cv::Point2f>& corners2l,
                                      vector<cv::Point2f>& corners2r,
                                      vector<cv::Point3f>& corners3d,
                                      vector<InputMarker>& markerInput,
                                      vector<DetectedMarker>& MyMarker)
{

	MyMarker.clear();

    if(corners3d.size() <= 2)
        return;

    for(int c0 = 0; c0 < corners3d.size(); c0++)
    {
        for(int c1 = 0; c1 < corners3d.size(); c1++)
        {
            if(c0 == c1)
                continue;
            for(int m = 0; m < markerInput.size(); m++)
            {
                if(distance_cal(corners3d[c0], corners3d[c1],
                                cal_distance(markerInput[m].coordinate[0], markerInput[m].coordinate[1])))
                {
                    for(int c2 = 0; c2 < corners3d.size(); c2++)
                    {
                        if(c2 == c0 || c2 == c1)
                            continue;
                        if(distance_cal(corners3d[c0], corners3d[c2],
                                        cal_distance(markerInput[m].coordinate[0], markerInput[m].coordinate[2])) &&
                           distance_cal(corners3d[c1], corners3d[c2],
                                        cal_distance(markerInput[m].coordinate[2], markerInput[m].coordinate[1])))
                        {
                            if(((corners3d[c1].x - corners3d[c0].x) *
                                (corners3d[c2].y - corners3d[c0].y) -
                                (corners3d[c1].y - corners3d[c0].y) *
                                (corners3d[c2].x - corners3d[c0].x)) *
                               markerInput[m].direction >
                               0)
                            { //识别Marker

                                DetectedMarker marker;
                                marker.name = markerInput[m].name;
                                marker.Coord2L.push_back(corners2l[c0]);
                                marker.Coord2L.push_back(corners2l[c1]);
                                marker.Coord2L.push_back(corners2l[c2]);
                                marker.Coord2R.push_back(corners2r[c0]);
                                marker.Coord2R.push_back(corners2r[c1]);
                                marker.Coord2R.push_back(corners2r[c2]);
                                marker.Coord3D.push_back(corners3d[c0]);
                                marker.Coord3D.push_back(corners3d[c1]);
                                marker.Coord3D.push_back(corners3d[c2]);
                                if(markerInput[m].coordinate.size() == 3)
                                    MyMarker.push_back(marker);
                                if(markerInput[m].coordinate.size() > 3)
                                {
                                    float X1[3];
                                    X1[0] = corners3d[c1].x - corners3d[c0].x;
                                    X1[1] = corners3d[c1].y - corners3d[c0].y;
                                    X1[2] = corners3d[c1].z - corners3d[c0].z;

                                    float Y1[3];
                                    Y1[0] = corners3d[c2].x - corners3d[c0].x;
                                    Y1[1] = corners3d[c2].y - corners3d[c0].y;
                                    Y1[2] = corners3d[c2].z - corners3d[c0].z;

                                    float Z1[3];
                                    Z1[0] = X1[1] * Y1[2] - X1[2] * Y1[1];
                                    Z1[1] = X1[2] * Y1[0] - X1[0] * Y1[2];
                                    Z1[2] = X1[0] * Y1[1] - X1[1] * Y1[0];

                                    float T[4][4];
                                    T[0][0] = X1[0] / sqrtf(X1[0] * X1[0] + X1[1] * X1[1] + X1[2] * X1[2]);
                                    T[1][0] = X1[1] / sqrtf(X1[0] * X1[0] + X1[1] * X1[1] + X1[2] * X1[2]);
                                    T[2][0] = X1[2] / sqrtf(X1[0] * X1[0] + X1[1] * X1[1] + X1[2] * X1[2]);

                                    T[0][1] = Y1[0] / sqrtf(Y1[0] * Y1[0] + Y1[1] * Y1[1] + Y1[2] * Y1[2]);
                                    T[1][1] = Y1[1] / sqrtf(Y1[0] * Y1[0] + Y1[1] * Y1[1] + Y1[2] * Y1[2]);
                                    T[2][1] = Y1[2] / sqrtf(Y1[0] * Y1[0] + Y1[1] * Y1[1] + Y1[2] * Y1[2]);

                                    T[0][2] = Z1[0] / sqrtf(Z1[0] * Z1[0] + Z1[1] * Z1[1] + Z1[2] * Z1[2]);
                                    T[1][2] = Z1[1] / sqrtf(Z1[0] * Z1[0] + Z1[1] * Z1[1] + Z1[2] * Z1[2]);
                                    T[2][2] = Z1[2] / sqrtf(Z1[0] * Z1[0] + Z1[1] * Z1[1] + Z1[2] * Z1[2]);

                                    T[0][3] = corners3d[c0].x;
                                    T[1][3] = corners3d[c0].y;
                                    T[2][3] = corners3d[c0].z;

                                    T[3][0] = 0;
                                    T[3][1] = 0;
                                    T[3][2] = 0;
                                    T[3][3] = 1;
                                    if(markerInput[m].coordinate.size() >= 4)
                                    {
                                        for(int c3 = 0; c3 < corners3d.size(); c3++)
                                        {
                                            if(c3 == c0 || c3 == c1 || c3 == c2)
                                                continue;
                                            cv::Point3f X3;
                                            X3.x = T[0][0] * markerInput[m].coordinate[3].x +
                                                   T[0][1] * markerInput[m].coordinate[3].y +
                                                   T[0][2] * markerInput[m].coordinate[3].z + T[0][3];
                                            X3.y = T[1][0] * markerInput[m].coordinate[3].x +
                                                   T[1][1] * markerInput[m].coordinate[3].y +
                                                   T[1][2] * markerInput[m].coordinate[3].z + T[1][3];
                                            X3.z = T[2][0] * markerInput[m].coordinate[3].x +
                                                   T[2][1] * markerInput[m].coordinate[3].y +
                                                   T[2][2] * markerInput[m].coordinate[3].z + T[2][3];
                                            if(cal_distance(X3, corners3d[c3]) <= 3)
                                            {
                                                marker.Coord2L.push_back(corners2l[c3]);
                                                marker.Coord2R.push_back(corners2r[c3]);
                                                marker.Coord3D.push_back(corners3d[c3]);
                                                if(markerInput[m].coordinate.size() == 4)
                                                    MyMarker.push_back(marker);
                                                if(markerInput[m].coordinate.size() >= 5)
                                                {
                                                    for(int c4 = 0; c4 < corners3d.size(); c4++)
                                                    {
                                                        if(c4 == c0 || c4 == c1 || c4 == c2 || c4 == c3)
                                                            continue;
                                                        cv::Point3f X4;
                                                        X4.x =
                                                        T[0][0] * markerInput[m].coordinate[4].x +
                                                        T[0][1] * markerInput[m].coordinate[4].y +
                                                        T[0][2] * markerInput[m].coordinate[4].z + T[0][3];
                                                        X4.y =
                                                        T[1][0] * markerInput[m].coordinate[4].x +
                                                        T[1][1] * markerInput[m].coordinate[4].y +
                                                        T[1][2] * markerInput[m].coordinate[4].z + T[1][3];
                                                        X4.z =
                                                        T[2][0] * markerInput[m].coordinate[4].x +
                                                        T[2][1] * markerInput[m].coordinate[4].y +
                                                        T[2][2] * markerInput[m].coordinate[4].z + T[2][3];
                                                        if(cal_distance(X4, corners3d[c4]) <= 3)
                                                        {
                                                            marker.Coord2L.push_back(corners2l[c4]);
                                                            marker.Coord2R.push_back(corners2r[c4]);
                                                            marker.Coord3D.push_back(corners3d[c4]);
                                                            if(markerInput[m].coordinate.size() == 5)
                                                                MyMarker.push_back(marker);
                                                            if(markerInput[m].coordinate.size() >= 6)
                                                            {
                                                                for(int c5 = 0; c5 < corners3d.size(); c5++)
                                                                {
                                                                    if(c5 == c0 || c5 == c1 ||
                                                                       c5 == c2 || c5 == c3 || c5 == c4)
                                                                        continue;
                                                                    cv::Point3f X5;
                                                                    X5.x =
                                                                    T[0][0] *
                                                                    markerInput[m].coordinate[5].x +
                                                                    T[0][1] *
                                                                    markerInput[m].coordinate[5].y +
                                                                    T[0][2] *
                                                                    markerInput[m].coordinate[5].z +
                                                                    T[0][3];
                                                                    X5.y =
                                                                    T[1][0] *
                                                                    markerInput[m].coordinate[5].x +
                                                                    T[1][1] *
                                                                    markerInput[m].coordinate[5].y +
                                                                    T[1][2] *
                                                                    markerInput[m].coordinate[5].z +
                                                                    T[1][3];
                                                                    X5.z =
                                                                    T[2][0] *
                                                                    markerInput[m].coordinate[5].x +
                                                                    T[2][1] *
                                                                    markerInput[m].coordinate[5].y +
                                                                    T[2][2] *
                                                                    markerInput[m].coordinate[5].z +
                                                                    T[2][3];
                                                                    if(cal_distance(X5, corners3d[c5]) <= 3)
                                                                    {
                                                                        marker.Coord2L.push_back(corners2l[c5]);
                                                                        marker.Coord2R.push_back(corners2r[c5]);
                                                                        marker.Coord3D.push_back(corners3d[c5]);
                                                                        if(markerInput[m].coordinate.size() == 6)
                                                                            MyMarker.push_back(marker);
                                                                        if(markerInput[m].coordinate.size() >= 7)
                                                                        {
                                                                            for(int c6 = 0;
                                                                                c6 < corners3d.size(); c6++)
                                                                            {
                                                                                if(c6 == c0 || c6 == c1 ||
                                                                                   c6 == c2 || c6 == c3 ||
                                                                                   c6 == c4 || c6 == c5)
                                                                                    continue;
                                                                                cv::Point3f X6;
                                                                                X6.x = T[0][0] *
                                                                                       markerInput[m]
                                                                                       .coordinate[6]
                                                                                       .x +
                                                                                       T[0][1] *
                                                                                       markerInput[m]
                                                                                       .coordinate[6]
                                                                                       .y +
                                                                                       T[0][2] *
                                                                                       markerInput[m]
                                                                                       .coordinate[6]
                                                                                       .z +
                                                                                       T[0][3];
                                                                                X6.y = T[1][0] *
                                                                                       markerInput[m]
                                                                                       .coordinate[6]
                                                                                       .x +
                                                                                       T[1][1] *
                                                                                       markerInput[m]
                                                                                       .coordinate[6]
                                                                                       .y +
                                                                                       T[1][2] *
                                                                                       markerInput[m]
                                                                                       .coordinate[6]
                                                                                       .z +
                                                                                       T[1][3];
                                                                                X6.z = T[2][0] *
                                                                                       markerInput[m]
                                                                                       .coordinate[6]
                                                                                       .x +
                                                                                       T[2][1] *
                                                                                       markerInput[m]
                                                                                       .coordinate[6]
                                                                                       .y +
                                                                                       T[2][2] *
                                                                                       markerInput[m]
                                                                                       .coordinate[6]
                                                                                       .z +
                                                                                       T[2][3];
                                                                                if(cal_distance(X6, corners3d[c6]) <= 3)
                                                                                {
                                                                                    marker.Coord2L.push_back(
                                                                                    corners2l[c6]);
                                                                                    marker.Coord2R.push_back(
                                                                                    corners2r[c6]);
                                                                                    marker.Coord3D.push_back(
                                                                                    corners3d[c6]);
                                                                                    if(markerInput[m]
                                                                                       .coordinate.size() == 7)
                                                                                        MyMarker.push_back(marker);
                                                                                    if(markerInput[m]
                                                                                       .coordinate.size() >= 8)
                                                                                    {
                                                                                        for(int c7 = 0;
                                                                                            c7 < corners3d
                                                                                                 .size();
                                                                                            c7++)
                                                                                        {
                                                                                            if(c7 == c0 ||
                                                                                               c7 == c1 ||
                                                                                               c7 == c2 ||
                                                                                               c7 == c3 ||
                                                                                               c7 == c4 ||
                                                                                               c7 == c5 ||
                                                                                               c7 == c6)
                                                                                                continue;
                                                                                            cv::Point3f X7;
                                                                                            X7.x =
                                                                                            T[0][0] *
                                                                                            markerInput[m]
                                                                                            .coordinate[7]
                                                                                            .x +
                                                                                            T[0][1] *
                                                                                            markerInput[m]
                                                                                            .coordinate[7]
                                                                                            .y +
                                                                                            T[0][2] *
                                                                                            markerInput[m]
                                                                                            .coordinate[7]
                                                                                            .z +
                                                                                            T[0][3];
                                                                                            X7.y =
                                                                                            T[1][0] *
                                                                                            markerInput[m]
                                                                                            .coordinate[7]
                                                                                            .x +
                                                                                            T[1][1] *
                                                                                            markerInput[m]
                                                                                            .coordinate[7]
                                                                                            .y +
                                                                                            T[1][2] *
                                                                                            markerInput[m]
                                                                                            .coordinate[7]
                                                                                            .z +
                                                                                            T[1][3];
                                                                                            X7.z =
                                                                                            T[2][0] *
                                                                                            markerInput[m]
                                                                                            .coordinate[7]
                                                                                            .x +
                                                                                            T[2][1] *
                                                                                            markerInput[m]
                                                                                            .coordinate[7]
                                                                                            .y +
                                                                                            T[2][2] *
                                                                                            markerInput[m]
                                                                                            .coordinate[7]
                                                                                            .z +
                                                                                            T[2][3];
                                                                                            if(cal_distance(X7, corners3d[c7]) <= 3)
                                                                                            {
                                                                                                marker
                                                                                                .Coord2L
                                                                                                .push_back(corners2l[c7]);
                                                                                                marker
                                                                                                .Coord2R
                                                                                                .push_back(corners2r[c7]);
                                                                                                marker
                                                                                                .Coord3D
                                                                                                .push_back(corners3d[c7]);
                                                                                                if(markerInput[m]
                                                                                                   .coordinate
                                                                                                   .size() == 8)
                                                                                                    MyMarker
                                                                                                    .push_back(marker);
                                                                                                if(markerInput[m]
                                                                                                   .coordinate
                                                                                                   .size() >= 9)
                                                                                                {
                                                                                                    for(int c8 = 0;
                                                                                                        c8 < corners3d
                                                                                                             .size();
                                                                                                        c8++)
                                                                                                    {
                                                                                                        if(c8 == c0 ||
                                                                                                           c8 == c1 ||
                                                                                                           c8 == c2 ||
                                                                                                           c8 == c3 ||
                                                                                                           c8 == c4 ||
                                                                                                           c8 == c5 ||
                                                                                                           c8 == c6 ||
                                                                                                           c8 == c7)
                                                                                                            continue;
                                                                                                        cv::Point3f X8;
                                                                                                        X8
                                                                                                        .x =
                                                                                                        T[0][0] *
                                                                                                        markerInput[m]
                                                                                                        .coordinate[8]
                                                                                                        .x +
                                                                                                        T[0][1] *
                                                                                                        markerInput[m]
                                                                                                        .coordinate[8]
                                                                                                        .y +
                                                                                                        T[0][2] *
                                                                                                        markerInput[m]
                                                                                                        .coordinate[8]
                                                                                                        .z +
                                                                                                        T[0][3];
                                                                                                        X8
                                                                                                        .y =
                                                                                                        T[1][0] *
                                                                                                        markerInput[m]
                                                                                                        .coordinate[8]
                                                                                                        .x +
                                                                                                        T[1][1] *
                                                                                                        markerInput[m]
                                                                                                        .coordinate[8]
                                                                                                        .y +
                                                                                                        T[1][2] *
                                                                                                        markerInput[m]
                                                                                                        .coordinate[8]
                                                                                                        .z +
                                                                                                        T[1][3];
                                                                                                        X8
                                                                                                        .z =
                                                                                                        T[2][0] *
                                                                                                        markerInput[m]
                                                                                                        .coordinate[8]
                                                                                                        .x +
                                                                                                        T[2][1] *
                                                                                                        markerInput[m]
                                                                                                        .coordinate[8]
                                                                                                        .y +
                                                                                                        T[2][2] *
                                                                                                        markerInput[m]
                                                                                                        .coordinate[8]
                                                                                                        .z +
                                                                                                        T[2][3];
                                                                                                        if(cal_distance(X8, corners3d[c8]) <=
                                                                                                           3)
                                                                                                        {
                                                                                                            marker
                                                                                                            .Coord2L
                                                                                                            .push_back(corners2l[c8]);
                                                                                                            marker
                                                                                                            .Coord2R
                                                                                                            .push_back(corners2r[c8]);
                                                                                                            marker
                                                                                                            .Coord3D
                                                                                                            .push_back(corners3d[c8]);
                                                                                                            if(markerInput[m]
                                                                                                               .coordinate
                                                                                                               .size() == 9)
                                                                                                                MyMarker
                                                                                                                .push_back(marker);
                                                                                                            if(markerInput[m]
                                                                                                               .coordinate
                                                                                                               .size() >= 10)
                                                                                                            {
                                                                                                                for(int c9 = 0;
                                                                                                                    c9 < corners3d
                                                                                                                         .size();
                                                                                                                    c9++)
                                                                                                                {
                                                                                                                    if(c9 == c0 ||
                                                                                                                       c9 == c1 ||
                                                                                                                       c9 == c2 ||
                                                                                                                       c9 == c3 ||
                                                                                                                       c9 == c4 ||
                                                                                                                       c9 == c5 ||
                                                                                                                       c9 == c6 ||
                                                                                                                       c9 == c7 ||
                                                                                                                       c9 == c8)
                                                                                                                        continue;
                                                                                                                    cv::Point3f X9;
                                                                                                                    X9
                                                                                                                    .x =
                                                                                                                    T[0][0] *
                                                                                                                    markerInput[m]
                                                                                                                    .coordinate[9]
                                                                                                                    .x +
                                                                                                                    T[0][1] *
                                                                                                                    markerInput[m]
                                                                                                                    .coordinate[9]
                                                                                                                    .y +
                                                                                                                    T[0][2] *
                                                                                                                    markerInput[m]
                                                                                                                    .coordinate[9]
                                                                                                                    .z +
                                                                                                                    T[0][3];
                                                                                                                    X9
                                                                                                                    .y =
                                                                                                                    T[1][0] *
                                                                                                                    markerInput[m]
                                                                                                                    .coordinate[9]
                                                                                                                    .x +
                                                                                                                    T[1][1] *
                                                                                                                    markerInput[m]
                                                                                                                    .coordinate[9]
                                                                                                                    .y +
                                                                                                                    T[1][2] *
                                                                                                                    markerInput[m]
                                                                                                                    .coordinate[9]
                                                                                                                    .z +
                                                                                                                    T[1][3];
                                                                                                                    X9
                                                                                                                    .z =
                                                                                                                    T[2][0] *
                                                                                                                    markerInput[m]
                                                                                                                    .coordinate[9]
                                                                                                                    .x +
                                                                                                                    T[2][1] *
                                                                                                                    markerInput[m]
                                                                                                                    .coordinate[9]
                                                                                                                    .y +
                                                                                                                    T[2][2] *
                                                                                                                    markerInput[m]
                                                                                                                    .coordinate[9]
                                                                                                                    .z +
                                                                                                                    T[2][3];
                                                                                                                    if(cal_distance(X9, corners3d[c9]) <=
                                                                                                                       3)
                                                                                                                    {
                                                                                                                        marker
                                                                                                                        .Coord2L
                                                                                                                        .push_back(corners2l[c9]);
                                                                                                                        marker
                                                                                                                        .Coord2R
                                                                                                                        .push_back(corners2r[c9]);
                                                                                                                        marker
                                                                                                                        .Coord3D
                                                                                                                        .push_back(corners3d[c9]);
                                                                                                                        if(markerInput[m]
                                                                                                                           .coordinate
                                                                                                                           .size() == 10)
                                                                                                                            MyMarker
                                                                                                                            .push_back(marker);
                                                                                                                    }
                                                                                                                }
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

double MarkerDetector::cal_distance(cv::Point3f a, cv::Point3f b)
{
    return (sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z)));
}

// Pattern板检测
void MarkerDetector::Pattern_Recognize(vector<cv::Point2f>& corners2l,
                                       vector<cv::Point2f>& corners2r,
                                       vector<cv::Point3f>& corners3d,
                                       vector<DetectedMarker>& MyPattern)
{
    if(corners3d.size() <= 2)
        return;
    for(int i = 0; i < corners3d.size(); i++)
    {
        for(int j = 0; j < corners3d.size(); j++)
        {
            if(i == j)
                continue;
            if(distance_cal(corners3d[i], corners3d[j], 62.2))
                for(int k = 0; k < corners3d.size(); k++)
                {
                    if(k == i || k == j)
                        continue;
                    if(distance_cal(corners3d[i], corners3d[k], 62.1) &&
                       distance_cal(corners3d[j], corners3d[k], 86.2))
                        for(int n = 0; n < corners3d.size(); n++)
                        {
                            if(n == i || n == j || n == k)
                                continue;
                            if(distance_cal(corners3d[i], corners3d[n], 90.0) &&
                               distance_cal(corners3d[j], corners3d[n], 62.2) &&
                               distance_cal(corners3d[k], corners3d[n], 62.2))
                            {
                                DetectedMarker marker;
                                marker.name = "Pattern";
                                marker.Coord2L.push_back(corners2l[i]);
                                marker.Coord2L.push_back(corners2l[j]);
                                marker.Coord2L.push_back(corners2l[k]);
                                marker.Coord2L.push_back(corners2l[n]);

                                marker.Coord2R.push_back(corners2r[i]);
                                marker.Coord2R.push_back(corners2r[j]);
                                marker.Coord2R.push_back(corners2r[k]);
                                marker.Coord2R.push_back(corners2r[n]);

                                marker.Coord3D.push_back(corners3d[i]);
                                marker.Coord3D.push_back(corners3d[j]);
                                marker.Coord3D.push_back(corners3d[k]);
                                marker.Coord3D.push_back(corners3d[n]);
                                MyPattern.push_back(marker);
                            }
                        }
                }
        }
    }
}

//判断两点之间距离和输入Marker边长的吻合程度，小于阈值时认为吻合
bool MarkerDetector::distance_cal(cv::Point3f a, cv::Point3f b, float d)
{
	double threshold = 2.5;
    float distance = cal_distance(a, b);
    if(d - distance < threshold && d - distance > -threshold)
        return true;
    else
        return false;
}

bool MarkerDetector::readMarker(const char* path, const char* prefix, int n, vector<InputMarker>& markerInput)
{
    /*Marker以 Marker1.txt Marker2.txt...等的形式命名txt文件中有名字、方向和三个数字，
    分别是Marker的短直角边、长直角边和斜边长度，单位毫米，以回车分隔。如：
    tracker
    -1
    3
    0,0,0
    66.3
    81.4
    从Marker1开始读取。*/
    markerInput.clear();
    char markpath[100];
    for(int i = 0; i < n; i++)
    {
        InputMarker X;
        sprintf(markpath, "%s\\%s%d.txt", path, prefix, i + 1);
        ifstream fin(markpath);
		if(!fin)
		{
			continue;
		}

		string name;
		fin>>name;	
		strcpy(X.name,name.data());
		//X.name = (char*)name.data();

		fin>>X.direction;

		fin>>X.num;

		for(int j = 0; j < X.num; j++)
		{
			cv::Point3f Y;
			char a, b;
			string point;
			fin>>point;
			const char* buffer;
			buffer = point.data();

			sscanf(buffer, "%f%c%f%c%f", &Y.x, &a, &Y.y, &b, &Y.z);
			X.coordinate.push_back(Y);
		}
		
        markerInput.push_back(X);
        fin.close();
    }
    return true;
}

void MarkerDetector::probCalibration(vector<DetectedMarker>& MyMarker, vector<cv::Point3f>& prob_point)
{
    int k = 0;
    int n = 0;
    for(int i = 0; i < MyMarker.size(); i++)
    {
        if(strcmp(MyMarker[i].name, "prob") == 0)
        {
            k = i;
        }
        if(strcmp(MyMarker[i].name, "Pattern") == 0)
        {
            n = i;
        }
    }
    if(k == 0 || n == 0)
        return;
    cv::Point3f X;
    X.x = MyMarker[n].Coord3D[0].x + MyMarker[n].Coord3D[1].x + MyMarker[n].Coord3D[2].x;
    X.y = MyMarker[n].Coord3D[0].y + MyMarker[n].Coord3D[1].y + MyMarker[n].Coord3D[2].y;
    X.z = MyMarker[n].Coord3D[0].z + MyMarker[n].Coord3D[1].z + MyMarker[n].Coord3D[2].z;

    float X1[3];
    X1[0] = MyMarker[k].Coord3D[1].x - MyMarker[k].Coord3D[0].x;
    X1[1] = MyMarker[k].Coord3D[1].y - MyMarker[k].Coord3D[0].y;
    X1[2] = MyMarker[k].Coord3D[1].z - MyMarker[k].Coord3D[0].z;

    float Y1[3];
    Y1[0] = MyMarker[k].Coord3D[2].x - MyMarker[k].Coord3D[0].x;
    Y1[1] = MyMarker[k].Coord3D[2].y - MyMarker[k].Coord3D[0].y;
    Y1[2] = MyMarker[k].Coord3D[2].z - MyMarker[k].Coord3D[0].z;

    float Z1[3];
    Z1[0] = X1[1] * Y1[2] - X1[2] * Y1[1];
    Z1[1] = X1[2] * Y1[0] - X1[0] * Y1[2];
    Z1[2] = X1[0] * Y1[1] - X1[1] * Y1[0];

    double T_[4][4];
    T_[0][0] = X1[0] / sqrtf(X1[0] * X1[0] + X1[1] * X1[1] + X1[2] * X1[2]);
    T_[0][1] = X1[1] / sqrtf(X1[0] * X1[0] + X1[1] * X1[1] + X1[2] * X1[2]);
    T_[0][2] = X1[2] / sqrtf(X1[0] * X1[0] + X1[1] * X1[1] + X1[2] * X1[2]);

    T_[1][0] = Y1[0] / sqrtf(Y1[0] * Y1[0] + Y1[1] * Y1[1] + Y1[2] * Y1[2]);
    T_[1][1] = Y1[1] / sqrtf(Y1[0] * Y1[0] + Y1[1] * Y1[1] + Y1[2] * Y1[2]);
    T_[1][2] = Y1[2] / sqrtf(Y1[0] * Y1[0] + Y1[1] * Y1[1] + Y1[2] * Y1[2]);

    T_[2][0] = Z1[0] / sqrtf(Z1[0] * Z1[0] + Z1[1] * Z1[1] + Z1[2] * Z1[2]);
    T_[2][1] = Z1[1] / sqrtf(Z1[0] * Z1[0] + Z1[1] * Z1[1] + Z1[2] * Z1[2]);
    T_[2][2] = Z1[2] / sqrtf(Z1[0] * Z1[0] + Z1[1] * Z1[1] + Z1[2] * Z1[2]);

    T_[0][3] = -(MyMarker[k].Coord3D[0].x * T_[0][0] + MyMarker[k].Coord3D[0].y * T_[0][1] +
                 MyMarker[k].Coord3D[0].z * T_[0][2]);
    T_[1][3] = -(MyMarker[k].Coord3D[0].x * T_[1][0] + MyMarker[k].Coord3D[0].y * T_[1][1] +
                 MyMarker[k].Coord3D[0].z * T_[1][2]);
    T_[2][3] = -(MyMarker[k].Coord3D[0].x * T_[2][0] + MyMarker[k].Coord3D[0].y * T_[2][1] +
                 MyMarker[k].Coord3D[0].z * T_[2][2]);

    T_[3][0] = 0;
    T_[3][1] = 0;
    T_[3][2] = 0;
    T_[3][3] = 1;

    cv::Point3f Y;
    Y.x = T_[0][0] * X.x + T_[0][1] * X.y + T_[0][2] * X.z + T_[0][3];
    Y.y = T_[1][0] * X.x + T_[1][1] * X.y + T_[1][2] * X.z + T_[1][3];
    Y.z = T_[2][0] * X.x + T_[2][1] * X.y + T_[2][2] * X.z + T_[2][3];

    prob_point.push_back(Y);
}

void MarkerDetector::LOAD_SVM(const char* path)
{
    SVM.load(path);
    if_SVM_moduleLoaded = 1;
}

void MarkerDetector::paintResult(cv::Mat& leftImg,
                                 cv::Mat& left_image,
                                 cv::Mat& rightImg,
                                 cv::Mat& right_image,
                                 int ifmarker,
                                 vector<cv::Point2f>& corners2l,
                                 vector<cv::Point2f>& corners2r,
                                 vector<cv::Point3f>& Coordinate3D,
                                 vector<DetectedMarker>& MyMarker,
                                 vector<cv::Point3f>& probpoint,
                                 vector<DetectedMarker>& MyPattern,
                                 cv::Mat& cameraMatrix,
                                 cv::Mat& T)
{
    cvtColor(leftImg, left_image, CV_GRAY2RGB); //灰度转RGB
    cvtColor(rightImg, right_image, CV_GRAY2RGB);
    if(!ifmarker) //不识别marker,仅绘制匹配的角点
    {
        char text[100] = { NULL };
        for(int i = 0; i < corners2l.size(); i++)
            cv::circle(left_image, corners2l[i], 5, cv::Scalar(255, 0, 0), -1);
        for(int i = 0; i < corners2r.size(); i++)
            cv::circle(right_image, corners2r[i], 5, cv::Scalar(255, 0, 0), -1);

        for(int i = 0; i < Coordinate3D.size(); i++)
        {
            sprintf(text, "(%d,%d,%d)", (int)Coordinate3D[i].x, (int)Coordinate3D[i].y,
                    (int)Coordinate3D[i].z);
            cv::putText(left_image, text, cv::Point((int)corners2l[i].x, (int)corners2l[i].y), 0,
                        0.5, CV_RGB(0, 255, 255));
        }
    }
    else
    {
        if(MyPattern.size() > 0)
        {
            for(int i = 0; i < MyPattern.size(); i++)
                for(int j = 0; j < 4; j++)
                    cv::circle(left_image, MyPattern[i].Coord2L[j], 5, cv::Scalar(0, 255, 255), -1);
            for(int i = 0; i < MyPattern.size(); i++)
            {
                cv::Point2f X;
                X.x = cameraMatrix.at<double>(0, 0) *
                      (MyPattern[i].Coord3D[0].x + MyPattern[i].Coord3D[1].x +
                       MyPattern[i].Coord3D[2].x + MyPattern[i].Coord3D[3].x - 2 * T.at<double>(0, 0)) /
                      (MyPattern[i].Coord3D[0].z + MyPattern[i].Coord3D[1].z +
                       MyPattern[i].Coord3D[2].z + MyPattern[i].Coord3D[3].z) +
                      cameraMatrix.at<double>(0, 2);
                X.y = cameraMatrix.at<double>(1, 1) *
                      (MyPattern[i].Coord3D[0].y + MyPattern[i].Coord3D[1].y +
                       MyPattern[i].Coord3D[2].y + MyPattern[i].Coord3D[3].y) /
                      (MyPattern[i].Coord3D[0].z + MyPattern[i].Coord3D[1].z +
                       MyPattern[i].Coord3D[2].z + MyPattern[i].Coord3D[3].z) +
                      cameraMatrix.at<double>(1, 2);
                cv::circle(left_image, X, 5, cv::Scalar(0, 255, 255), -1);
            }
        }
        if(1)
            for(int i = 0; i < probpoint.size(); i++)
            {
                cv::Point2f X;
                X.x = cameraMatrix.at<double>(0, 0) * (probpoint[i].x - 0.5 * T.at<double>(0, 0)) /
                      probpoint[i].z +
                      cameraMatrix.at<double>(0, 2);
                X.y = cameraMatrix.at<double>(1, 1) * probpoint[i].y / probpoint[i].z +
                      cameraMatrix.at<double>(1, 2);
                cv::circle(left_image, X, 5, cv::Scalar(0, 255, 255), -1);
            }
        for(int i = 0; i < MyMarker.size(); i++)
        {
            cv::line(left_image, MyMarker[i].Coord2L[0], MyMarker[i].Coord2L[1], cv::Scalar(0, 255, 255), 2);
            cv::line(left_image, MyMarker[i].Coord2L[0], MyMarker[i].Coord2L[2], cv::Scalar(0, 255, 255), 2);
            char textclass[100];
            sprintf(textclass, "%s", MyMarker[i].name);
            cv::putText(left_image, textclass, MyMarker[i].Coord2L[0], 0, 0.5, CV_RGB(0, 255, 255));
            cv::putText(left_image, "O", cv::Point(MyMarker[i].Coord2L[0].x, MyMarker[i].Coord2L[0].y + 20),
                        2, 0.5, CV_RGB(0, 255, 255));
            cv::putText(left_image, "X", cv::Point(MyMarker[i].Coord2L[1].x, MyMarker[i].Coord2L[1].y + 20),
                        2, 0.5, CV_RGB(0, 255, 255));
            cv::putText(left_image, "Y", cv::Point(MyMarker[i].Coord2L[2].x, MyMarker[i].Coord2L[2].y + 20),
                        2, 0.5, CV_RGB(0, 255, 255));

            for(int j = 0; j < MyMarker[i].Coord2L.size(); j++)
            {
                cv::circle(left_image, MyMarker[i].Coord2L[j], 5, cv::Scalar(0, 0, 255), -1);
            }
        }

        for(int i = 0; i < MyMarker.size(); i++)
        {
            cv::circle(right_image, MyMarker[i].Coord2R[0], 5, cv::Scalar(255, 0, 255), -1);
            cv::circle(right_image, MyMarker[i].Coord2R[1], 5, cv::Scalar(255, 0, 255), -1);
            cv::circle(right_image, MyMarker[i].Coord2R[2], 5, cv::Scalar(255, 0, 255), -1);
        }
    }
}

//已知探针尖的位置求探针在画面中的位置
void MarkerDetector::probLocation(vector<DetectedMarker>& MyMarker, cv::Point3f prob, vector<cv::Point3f>& probpoint)
{
    probpoint.clear();
    for(int k = 0; k < MyMarker.size(); k++)
    {
        if(!strcmp(MyMarker[k].name, "prob"))
        {
            float X1[3];
            X1[0] = MyMarker[k].Coord3D[1].x - MyMarker[k].Coord3D[0].x;
            X1[1] = MyMarker[k].Coord3D[1].y - MyMarker[k].Coord3D[0].y;
            X1[2] = MyMarker[k].Coord3D[1].z - MyMarker[k].Coord3D[0].z;

            float Y1[3];
            Y1[0] = MyMarker[k].Coord3D[2].x - MyMarker[k].Coord3D[0].x;
            Y1[1] = MyMarker[k].Coord3D[2].y - MyMarker[k].Coord3D[0].y;
            Y1[2] = MyMarker[k].Coord3D[2].z - MyMarker[k].Coord3D[0].z;

            float Z1[3];
            Z1[0] = X1[1] * Y1[2] - X1[2] * Y1[1];
            Z1[1] = X1[2] * Y1[0] - X1[0] * Y1[2];
            Z1[2] = X1[0] * Y1[1] - X1[1] * Y1[0];

            float T[4][4];
            T[0][0] = X1[0] / sqrtf(X1[0] * X1[0] + X1[1] * X1[1] + X1[2] * X1[2]);
            T[1][0] = X1[1] / sqrtf(X1[0] * X1[0] + X1[1] * X1[1] + X1[2] * X1[2]);
            T[2][0] = X1[2] / sqrtf(X1[0] * X1[0] + X1[1] * X1[1] + X1[2] * X1[2]);

            T[0][1] = Y1[0] / sqrtf(Y1[0] * Y1[0] + Y1[1] * Y1[1] + Y1[2] * Y1[2]);
            T[1][1] = Y1[1] / sqrtf(Y1[0] * Y1[0] + Y1[1] * Y1[1] + Y1[2] * Y1[2]);
            T[2][1] = Y1[2] / sqrtf(Y1[0] * Y1[0] + Y1[1] * Y1[1] + Y1[2] * Y1[2]);

            T[0][2] = Z1[0] / sqrtf(Z1[0] * Z1[0] + Z1[1] * Z1[1] + Z1[2] * Z1[2]);
            T[1][2] = Z1[1] / sqrtf(Z1[0] * Z1[0] + Z1[1] * Z1[1] + Z1[2] * Z1[2]);
            T[2][2] = Z1[2] / sqrtf(Z1[0] * Z1[0] + Z1[1] * Z1[1] + Z1[2] * Z1[2]);

            T[0][3] = MyMarker[k].Coord3D[0].x;
            T[1][3] = MyMarker[k].Coord3D[0].y;
            T[2][3] = MyMarker[k].Coord3D[0].z;

            T[3][0] = 0;
            T[3][1] = 0;
            T[3][2] = 0;
            T[3][3] = 1;

            cv::Point3f X;
            X.x = T[0][0] * prob.x + T[0][1] * prob.y + T[0][2] * prob.z + T[0][3];
            X.y = T[1][0] * prob.x + T[1][1] * prob.y + T[1][2] * prob.z + T[1][3];
            X.z = T[2][0] * prob.x + T[2][1] * prob.y + T[2][2] * prob.z + T[2][3];
            probpoint.push_back(X);
        }
    }
}

void MarkerDetector::fiducialRegistration(std::vector<InputMarker> & markerInput, std::vector<DetectedMarker> & detectedMarker)
{
	
	Eigen::Matrix3d R;
	Eigen::Vector3d t;
	int m = 0;
	for (int i = 0; i < detectedMarker.size(); i++)
	{
		vector<double>  x;
		vector<double>  y;

		for (int j = 0; j < markerInput.size(); j++)
		{
			if (strcmp(detectedMarker[i].name, markerInput[j].name) == 0)
			{
				m = j;
				break;
			}
		}
		for (int k = 0; k < detectedMarker[i].Coord3D.size(); k++)
		{
			x.push_back(markerInput[m].coordinate[k].x);
			x.push_back(markerInput[m].coordinate[k].y);
			x.push_back(markerInput[m].coordinate[k].z);

			y.push_back(detectedMarker[i].Coord3D[k].x);
			y.push_back(detectedMarker[i].Coord3D[k].y);
			y.push_back(detectedMarker[i].Coord3D[k].z);
		}
		if (x.size() != y.size())
		{
			cerr << "size must be the same!\n";
			return;
		}
		unsigned int n = (unsigned int)x.size() / 3;

		Map<const Matrix3Xd> X(&x[0], 3, n);
		Map<const Matrix3Xd> Y(&y[0], 3, n);
		Vector3d x0 = X.rowwise().mean();
		Vector3d y0 = Y.rowwise().mean();
		Matrix3d M = (Y.colwise() - y0) * (X.colwise() - x0).transpose();
		JacobiSVD<Matrix3d> svd(M, ComputeFullU | ComputeFullV);
		R = svd.matrixU() * svd.matrixV();
		if (R.determinant() < 0)
		{
			R(0, 2) = -R(0, 2);
			R(1, 2) = -R(1, 2);
			R(2, 2) = -R(2, 2);
		}
		t = y0 - R * x0;

		detectedMarker[i].R = R;
		detectedMarker[i].T = t;
		
	}
}


void MarkerDetector::drawDetectedMarkers(cv::Mat& inputImg,vector<DetectedMarker> Maker_Detected,char leftORright,bool show_T)
{

	for (int i = 0; i < Maker_Detected.size(); i++)
	{
		char showbuffer[50];
		if(show_T)
			sprintf(showbuffer,"%s:(%.2f ,%.2f ,%.2f)",Maker_Detected[i].name,Maker_Detected[i].T[0],Maker_Detected[i].T[1],Maker_Detected[i].T[2]);
		else
			sprintf(showbuffer,"%s",Maker_Detected[i].name);

		if(leftORright == 'L')
		{
			cv::line(inputImg,Maker_Detected[i].Coord2L[0],Maker_Detected[i].Coord2L[1],cv::Scalar(0,255,0),2);
			cv::line(inputImg,Maker_Detected[i].Coord2L[0],Maker_Detected[i].Coord2L[2],cv::Scalar(0,0,255),2);

			cv::putText(inputImg, showbuffer,cv::Point(Maker_Detected[i].Coord2L[0].x,Maker_Detected[i].Coord2L[0].y),CV_FONT_HERSHEY_COMPLEX,0.8, CV_RGB(255,0,0),1,8,false);

		}	
		else if(leftORright == 'R')
		{
			cv::line(inputImg,Maker_Detected[i].Coord2R[0],Maker_Detected[i].Coord2R[1],cv::Scalar(0,255,0),2);
			cv::line(inputImg,Maker_Detected[i].Coord2R[0],Maker_Detected[i].Coord2R[2],cv::Scalar(0,0,255),2);

			cv::putText(inputImg, showbuffer,cv::Point(Maker_Detected[i].Coord2R[0].x,Maker_Detected[i].Coord2R[0].y),CV_FONT_HERSHEY_COMPLEX,0.8, CV_RGB(255,0,0),1,8,false);
		}		

	}
}

void MarkerDetector::drawVecPoints(cv::Mat & inputImg,vector<cv::Point2f> corners,vector<cv::Point3f> Coordinate3D, bool noteFlag)
{
	cout<<"corners:"<<corners.size()<<"  Coordinate3D"<<Coordinate3D.size()<<endl;

	if(Coordinate3D.size() != corners.size())
	{
		int size = corners.size();
		for (int i = 0; i <size; i++)
		{ 
			cv::circle(inputImg,cv::Point(corners[i].x,corners[i].y),2,cv::Scalar(0,255,0),2);
		}
	}
	else
	{
		int size = corners.size();
		for (int i = 0; i <size; i++)
		{ 
			cv::circle(inputImg,cv::Point(corners[i].x,corners[i].y),2,cv::Scalar(0,255,0),2);

			char showbuffer[50];
			sprintf(showbuffer,"  (%.2f ,%.2f ,%.2f)",Coordinate3D[i].x,Coordinate3D[i].y,Coordinate3D[i].z);

			/*if(noteFlag)
				cv::putText(inputImg, showbuffer,cv::Point(corners[i].x,corners[i].y),CV_FONT_HERSHEY_COMPLEX,0.5, CV_RGB(255,0,0),1,8,false);*/
		}

		if (Coordinate3D.size() == 2)
		{
			;
			char showbuffer[50];
			sprintf(showbuffer,"Distance:(%.2f)",MarkerDetector::cal_distance(Coordinate3D[0],Coordinate3D[1]));

			//cv::putText(inputImg, showbuffer,cv::Point(20,20),CV_FONT_HERSHEY_COMPLEX,1, CV_RGB(255,0,0),1,8,false);
		}

	}
}

void MarkerDetector::drawMatchCorners(cv::Mat & inputImg,vector<cv::Point2f> cornersLeft,vector<cv::Point2f> cornersRight)
{
	if(cornersLeft.size()!=cornersRight.size() )
		return;

	for(int i = 0,iend =cornersLeft.size();i< iend; i++ )
	{
		cv::line(inputImg,cornersLeft[i],cornersRight[i]+cv::Point2f(1280,0),CV_RGB(0,255,0),1);
		char showbuffer[10];
		sprintf(showbuffer,"%d",i);
		cv::putText(inputImg, showbuffer,cornersLeft[i],CV_FONT_HERSHEY_COMPLEX,1, CV_RGB(255,0,0),1,8,false);
		cv::putText(inputImg, showbuffer,cornersRight[i]+cv::Point2f(1280,0),CV_FONT_HERSHEY_COMPLEX,1, CV_RGB(255,0,0),1,8,false);
	}
}

bool UDgreater ( std::pair <double,int> elem1, std::pair <double,int> elem2 )  
{  
	return elem1.first < elem2.first;  
}  

int MarkerDetector::Marker_LikeDetect(vector<cv::Point2f>& corners2l,
	vector<cv::Point2f>& corners2r,
	vector<cv::Point3f>& corners3d,
	vector<DetectedMarker>& DetectedLikeMarkers)
{

	DetectedLikeMarkers.clear();

	int pointNum = corners3d.size();
	if( pointNum<=2 )
		return 0;


	//thresh
	double threshLength = 50;

	for (int i = 0; i <pointNum;i++ )
		for (int j = i+1; j< pointNum;j++)
			for (int k = j+1 ; k < pointNum; k++)
			{

				std::vector<std::pair<double,int> > length;

				length.push_back(std::pair <double,int>(cal_distance(corners3d[i],corners3d[j]),k));
				length.push_back(std::pair <double,int>(cal_distance(corners3d[j],corners3d[k]),i));
				length.push_back(std::pair <double,int>(cal_distance(corners3d[k],corners3d[i]),j));


				sort(length.begin(),length.end(),UDgreater);

				double error = cv::pow( length[2].first,2) - cv::pow( length[1].first,2) - cv::pow( length[0].first,2) ;
				cout<<"error"<<error<<endl;

				if( fabs(error)<threshLength)
				{
					DetectedMarker LikeMarker;

					int orgPosIdx = length[2].second;
					int xPosIdx = length[1].second;
					int yPosIdx = length[0].second;

					LikeMarker.Coord3D.push_back(corners3d[orgPosIdx]);
					LikeMarker.Coord3D.push_back(corners3d[xPosIdx]);
					LikeMarker.Coord3D.push_back(corners3d[yPosIdx]);

					LikeMarker.Coord2L.push_back(corners2l[orgPosIdx]);
					LikeMarker.Coord2L.push_back(corners2l[xPosIdx]);
					LikeMarker.Coord2L.push_back(corners2l[yPosIdx]);


					LikeMarker.Coord2R.push_back(corners2r[orgPosIdx]);
					LikeMarker.Coord2R.push_back(corners2r[xPosIdx]);
					LikeMarker.Coord2R.push_back(corners2r[yPosIdx]);

					DetectedLikeMarkers.push_back(LikeMarker);
				}
			}

			return DetectedLikeMarkers.size();
}