#include <iostream>
#include <ctype.h>
#include <algorithm> 
#include <iterator> 
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/opencv.hpp"

#ifdef WITH_CUDA
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

using namespace cv;
using namespace std;

static void download(const cv::cuda::GpuMat& d_mat, vector<Point2f>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
	d_mat.download(mat);
}

static void download(const cv::cuda::GpuMat& d_mat, vector<uchar>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
	d_mat.download(mat);
}

enum Processor {
	CPU, CUDA
};

void featureTracking(const Mat& img_1, const Mat& img_2,
	vector<Point2f>& points1, vector<Point2f>& points2,
	vector<uchar>& status,
	Processor p = CPU) {
	//this function automatically gets rid of points for which tracking fails
	vector<float> err;
	Size winSize = Size(21, 21);
	int iters = 30;
	int maxLevel = 3;
	TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);

	switch (p) {
	case CPU:
		calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
#ifdef WITH_CUDA
	case CUDA: {
		cv::cuda::GpuMat gpu_img1(img_1);
		cv::cuda::GpuMat gpu_img2(img_2);
		cv::cuda::GpuMat gpu_points1(points1);
		cv::cuda::GpuMat gpu_points2;
		cv::cuda::GpuMat d_status;

		cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(winSize, maxLevel, iters);
		d_pyrLK_sparse->calc(gpu_img1, gpu_img2, gpu_points1, gpu_points2, d_status);

		points2.clear();
		points2.resize(gpu_points2.cols); download(gpu_points2, points2);
		status.clear();
		status.resize(d_status.cols); download(d_status, status);
	}
#endif
	}
	//getting rid of points for which the KLT tracking failed or those who have gone outside the frame
	int indexCorrection = 0;
	for (int i = 0; i < status.size(); i++) {
		Point2d pt = points2.at(i - indexCorrection);
		if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)) {
			if ((pt.x < 0) || (pt.y < 0)) {
				status.at(i) = 0;
			}
			points1.erase(points1.begin() + (i - indexCorrection));
			points2.erase(points2.begin() + (i - indexCorrection));
			indexCorrection++;
		}
	}
}

static bool createDetectorDescriptorMatcher(const string& detector_type,
	cv::Ptr<FeatureDetector>& detector) {
	cout << "<Creating feature detector" << endl;
	if (detector_type == "SIFT") {
		detector = cv::xfeatures2d::SIFT::create();
	}
	else if (detector_type == "SURF") {
		detector = cv::xfeatures2d::SURF::create();
	}
	else if (detector_type == "ORB") {
		detector = cv::ORB::create();
	}
	bool isCreated = !detector.empty();
	if (!isCreated)
		cout << "Can not create feature detector of given types." << endl << ">" << endl;
	return isCreated;
}

void featureDetection(const Ptr<FeatureDetector>& detector,
	const Mat& img,
	vector<Point2f>& points,
	std::string mode,
	Processor p = CPU) {
	int num_points = 4000;
	double minDist = 0;
	points.clear();
	switch (p) {
	case CPU: {
		vector<KeyPoint> keypoints;
		if (mode == "FAST") {
			//uses FAST as of now, modify parameters as necessary
			int fast_threshold = 20;
			bool nonmaxSuppression = true;
			FAST(img, keypoints, fast_threshold, nonmaxSuppression);
		}
		else if (mode == "NeFAST") {
			detector->detect(img, keypoints);
		}
		else {
			cv::Mat mask;
			int maxCorners = 300;
			double qualityLevel = 0.01;
			double minDistance = 20.;
			int blockSize = 3;
			bool useHarrisDetector = false;
			double k = 0.04;
			cv::goodFeaturesToTrack(img, points, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);
			return;
		}
		KeyPoint::convert(keypoints, points, vector<int>());
	}
#ifdef WITH_CUDA
	case CUDA: {
		cv::cuda::GpuMat gpu_points;
		cv::cuda::GpuMat gpu_img(img);
		cv::Ptr<cv::cuda::CornersDetector> detector = cv::cuda::createGoodFeaturesToTrackDetector(gpu_img.type(), num_points, 0.01, minDist);
		detector->detect(gpu_img, gpu_points);
		points.resize(gpu_points.cols);
		download(gpu_points, points);
	}
#endif
	}
}