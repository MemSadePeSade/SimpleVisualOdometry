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

namespace kFAST {
	const int  fast_threshold = 20;
	const bool nonmaxSuppression = true;
	const int  type = 2;
}
namespace kGFTT {
	const int maxCorners = 300;
	const double qualityLevel = 0.01;
	const double minDistance = 20.;
	const int blockSize = 3;
	const bool useHarrisDetector = false;
	const double k = 0.04;
}
namespace kOptFlow {
	cv::Size winSize = cv::Size(21, 21);
	int iters = 30;
	int maxLevel = 3;
	cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
}

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

#ifdef WITH_CUDA
inline void featureTracking(const cv::cuda::GpuMat& img_1, const cv::cuda::GpuMat& img_2,
	const cv::cuda::GpuMat& points1, cv::cuda::GpuMat& points2,
	cv::cuda::GpuMat d_status) {
	using namespace kOptFlow;
	cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse =
		cv::cuda::SparsePyrLKOpticalFlow::create(winSize, maxLevel, iters);
	d_pyrLK_sparse->calc(gpu_img1, gpu_img2, gpu_points1, gpu_points2, d_status);
}
#endif

inline void featureTracking(const cv::Mat& img_1, const cv::Mat& img_2,
	std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2,
	std::vector<uchar>& status) {
	//this function automatically gets rid of points for which tracking fails
	std::vector<float> err;
	using namespace kOptFlow;
	calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
}

inline void CorrectIndex(std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2,
	std::vector<uchar>& status) {
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

#ifdef WITH_CUDA
void featureDetection(const cv::gpu::Mat& img, cv::cuda::GpuMat& gpu_points,
	FeatureType feature_type) {
	if (feature_type == FeatureType::FAST) {
		std::vector<cv::KeyPoint> keypoints;
		std::vector<cv::Point2f>  points;
		using namespace kFAST;
		cv::Ptr<cv::cuda::FastFeatureDetector> gpuFastDetector = cv::cuda::FastFeatureDetector::create(fast_threshold,
			nonmaxSuppression, type);
		cv::cuda::GpuMat gpu_img;
		gpu_img.upload(img);
		gpuFastDetector->detect(gpu_img, keypoints);
		cv::KeyPoint::convert(keypoints, points, std::vector<int>());//  fix
		cv::cuda::GpuMat gpu_points_tmp(points);
		gpu_points = gpu_points_tmp;
	}
	else if (feature_type == FeatureType::GFTT) {
		cv::cuda::GpuMat gpu_points;
		cv::cuda::GpuMat gpu_img(img);
		using namespace kGFTT;
		cv::Ptr<cv::cuda::CornersDetector> detector = cv::cuda::createGoodFeaturesToTrackDetector(gpu_img.type(),
			maxCorners, qualityLevel, minDistance);
		detector->detect(gpu_img, gpu_points);
	}
	else
		std::cout << "This detector do not support" << std::endl;
}
#endif
void featureDetection(const cv::Mat& img, std::vector<cv::Point2f>& points,
	FeatureType feature_type) {
	std::vector<cv::KeyPoint> keypoints;
	if (feature_type == FeatureType::FAST) {
		using namespace kFAST;
		FAST(img, keypoints, fast_threshold, nonmaxSuppression, type);
		cv::KeyPoint::convert(keypoints, points, std::vector<int>());
	}
	else if (feature_type == FeatureType::GFTT) {
		cv::Mat mask;
		using namespace kGFTT;
		cv::goodFeaturesToTrack(img, points, maxCorners, qualityLevel, minDistance,
			mask, blockSize, useHarrisDetector, k);
		return;
	}
	else
		std::cout << "This detector do not support" << std::endl;
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

/*
class Tracker {
	public:
		Tracker(cv::Mat img);
		~Tracker();
		std::vector<cv::Point2f> ToTrack(cv::Mat& img);
		void UpdateStateTrackerImage(cv::Mat& img);
		void UpdateStateTrackerPoints(std::vector<cv::Point2f> points);
};*/
namespace Tracker {
	class CPUTracker {
	public:
		CPUTracker(cv::Mat img);
		~CPUTracker();
		std::vector<cv::Point2f> ToTrack(cv::Mat& img);
		void UpdateStateTrackerImage(cv::Mat& img);
		void UpdateStateTrackerPoints(std::vector<cv::Point2f> points);
	private:
		struct StateCPUTracker {
			cv::Mat img;
			std::vector<cv::Point2f> points;
		};
		StateCPUTracker state;
	};




	class GPUTracker {
	};
}