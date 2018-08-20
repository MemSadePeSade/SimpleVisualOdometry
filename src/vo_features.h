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

enum class FeatureType {
	BRISK, ORB, MSER, FAST,
	AGAST, GFTT, KAZE, SURF,
	SIFT
};

enum Processor {
	CPU, CUDA
};

static void download(const cv::cuda::GpuMat& d_mat, std::vector<cv::Point2f>& vec)
{
	vec.resize(d_mat.cols);
	cv::Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
	d_mat.download(mat);
}

static void download(const cv::cuda::GpuMat& d_mat, std::vector<uchar>& vec)
{
	vec.resize(d_mat.cols);
	cv::Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
	d_mat.download(mat);
}

void featureTracking(const cv::Mat& img_1, const cv::Mat& img_2,
	std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2,
	std::vector<uchar>& status,
	Processor p = CUDA) {
	//this function automatically gets rid of points for which tracking fails
	std::vector<float> err;
	cv::Size winSize = cv::Size(21, 21);
	int iters = 30;
	int maxLevel = 3;
	cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

	switch (p) {
	default:
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
		cv::Point2d pt = points2.at(i - indexCorrection);
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

static bool createDetectorDescriptorMatcher(FeatureType feature_type,
	cv::Ptr<cv::FeatureDetector>& detector) {
	std::cout << "<Creating feature detector" << std::endl;
	if (feature_type == FeatureType::SIFT) {
		detector = cv::xfeatures2d::SIFT::create();
	}
	else if (feature_type == FeatureType::SURF) {
		detector = cv::xfeatures2d::SURF::create();
	}
	else if (feature_type == FeatureType::ORB) {
		detector = cv::ORB::create();
	}
	else
		return true;
	bool isCreated = !detector.empty();
	if (!isCreated)
		std::cout << "Can not create feature detector of given types." << std::endl << ">" << std::endl;
	return isCreated;
}

void featureDetection(const cv::Ptr<cv::FeatureDetector>& detector,
	const cv::Mat& img,
	std::vector<cv::Point2f>& points,
	FeatureType feature_type,
	Processor p = CUDA) {
	points.clear();
	std::vector<cv::KeyPoint> keypoints;
	if (feature_type == FeatureType::FAST) {
		//uses FAST as of now, modify parameters as necessary
		int  fast_threshold = 20;
		bool nonmaxSuppression = true;
		int  type = 2;

		switch (p) {
#ifdef WITH_CUDA
		case CUDA:
			cv::Ptr<cv::cuda::FastFeatureDetector> gpuFastDetector = cv::cuda::FastFeatureDetector::create(fast_threshold,
				nonmaxSuppression, type);
			cv::cuda::GpuMat gpu_img;
			gpu_img.upload(img);
			gpuFastDetector->detect(gpu_img, keypoints);
#endif
		default:
			FAST(img, keypoints, fast_threshold, nonmaxSuppression, type);
		}
	}
	else if (feature_type == FeatureType::GFTT) {
		cv::Mat mask;
		int maxCorners = 300;
		double qualityLevel = 0.01;
		double minDistance = 20.;
		int blockSize = 3;
		bool useHarrisDetector = false;
		double k = 0.04;

		switch (p) {
#ifdef WITH_CUDA
		case CUDA:
			cv::cuda::GpuMat gpu_points;
			cv::cuda::GpuMat gpu_img(img);
			cv::Ptr<cv::cuda::CornersDetector> detector = cv::cuda::createGoodFeaturesToTrackDetector(gpu_img.type(),
				maxCorners, qualityLevel, minDistance);
			detector->detect(gpu_img, gpu_points);
			points.resize(gpu_points.cols);
			download(gpu_points, points);
			return
#endif
		default: {
				cv::goodFeaturesToTrack(img, points, maxCorners, qualityLevel, minDistance,
					mask, blockSize, useHarrisDetector, k);
				return;
			}
		}
	}
	else
		detector->detect(img, keypoints);
	cv::KeyPoint::convert(keypoints, points, std::vector<int>());
}