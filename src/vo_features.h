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

using namespace cv;
using namespace std;

void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status) {
	//this function automatically gets rid of points for which tracking fails
	vector<float> err;
	Size winSize = Size(21, 21);
	TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);

	calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
	//getting rid of points for which the KLT tracking failed or those who have gone outside the frame
	int indexCorrection = 0;
	for (int i = 0; i < status.size(); i++) {
		Point2f pt = points2.at(i - indexCorrection);
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
	Ptr<FeatureDetector>& detector) {
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
	std::string mode) {
	vector<KeyPoint> keypoints;
	if (mode == "FAST") {
		//uses FAST as of now, modify parameters as necessary
		int fast_threshold = 20;
		bool nonmaxSuppression = true;
		FAST(img, keypoints, fast_threshold, nonmaxSuppression);
	}
	else {
		detector->detect(img, keypoints);
	}
	KeyPoint::convert(keypoints, points, vector<int>());
}