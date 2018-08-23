#include<cmath>
#include<string>
#include<vector>
#include<algorithm>
#include<functional>
#include<numeric>    

#include <boost/filesystem/operations.hpp>
#include "opencv2/core/utility.hpp"

#include "tracker.h"
#include "draw.h"

namespace {
#define MAX_FRAME 384
#define MIN_NUM_FEAT 2000
const detector::FeatureType kFeatureType = detector::FeatureType::GFTT;
const Processor kProcessor = CPU;
//const char *img_path = "C:\\Users\\vponomarev\\Desktop\\01\\img_datasets\\e1i90v1a30_undistorted\\frame%06d.jpg";
const char *img_path = "C:\\Users\\vponomarev\\Desktop\\01\\img_datasets\\e1i90v1a30\\frame%06d.jpg";
const char* keys =
		"{help h usage ?  |    | print help message }"
		"{@calib   |        | specify calib file }";
} //unnamed namespace

int main(int argc, const char* argv[]) {
	cv::CommandLineParser cmd(argc, argv, keys);
	if (cmd.has("help") || !cmd.check()){
		cmd.printMessage();
		cmd.printErrors();
		return 0;
	}

	std::string calib_filename = cmd.get<std::string>("@calib"); calib_filename = "z.yaml";
	CameraParam camera_param;
	if (!boost::filesystem::exists(calib_filename))
		std::cout << "No calibrated file" << std::endl;
	else
		LoadCameraParam(calib_filename, camera_param);

	char filename[200];
	int  counter = 0;
	sprintf(filename, img_path, counter);

	cv::Mat img = cv::imread(filename);
	if( img.empty())
		return -1;
	detector::FeatureDetector* detector = detector::InitFeatureDetector(kProcessor,kFeatureType);
	if (detector == nullptr)
		return -1;
	preprocess::PreProcess* preprocess = preprocess::InitPreprocessor(kProcessor, camera_param);
	if (preprocess == nullptr)
		return -1;
	tracker::Tracker* tracker = tracker::InitTracker(kProcessor);
	if (tracker == nullptr)
		return -1;
	
	auto preprocess_img = preprocess->operator()(img);
	auto points = detector->operator()(preprocess_img);
	tracker->UpdateStateTrackerImage(preprocess_img);
	tracker->UpdateStateTrackerPoints(points);
	
	cv::Mat traj = cv::Mat::zeros(600, 600, CV_8UC3);
	cv::Matx31d t_f(0, 0, 0);
	cv::Matx33d R_f(1, 0, 0, 0, 1, 0, 0, 0, 1);

	std::clock_t start;
	double duration;
	start = std::clock();
	// cycle through the directory
	std::vector<cv::Point2f> points_prev_conv;
	std::vector<cv::Point2f> points_curr_conv;
	for (counter = 1; counter < 384; ++counter) {
		sprintf(filename, img_path, counter);
		cv::Mat img_curr = cv::imread(filename);
		
		auto&& preprocess_img_curr = preprocess->operator()(img_curr);
		auto&& points_curr = tracker->ToTrackandCorrectIndex(preprocess_img_curr);
		auto&& points_prev = tracker->GetPoints();
			
		Converter(points_prev, points_prev_conv);
		Converter(points_curr, points_curr_conv);
		
		cv::Matx31d t;
		cv::Matx33d R;
		cv::Mat E, mask;

		E = cv::findEssentialMat(points_curr_conv, points_prev_conv, camera_param.focal_length,
			camera_param.pp, cv::RANSAC, 0.999, 1.0, mask);

		// init R,T
		if (counter == 1) {
			cv::recoverPose(E, points_curr_conv, points_prev_conv, R, t, camera_param.focal_length,
				camera_param.pp, mask);
			R_f = R;
			t_f = t;
			continue;
		}// estimate R,T
		else {
			cv::Matx33d R1, R2;
			cv::Matx31d T;
			cv::decomposeEssentialMat(E, R1, R2, T);
			auto euler_angles1 = rotationMatrixToEulerAngles(R1);
			auto euler_angles2 = rotationMatrixToEulerAngles(R2);
			auto dist1 = abs(euler_angles1(0)) + abs(euler_angles1(1)) + abs(euler_angles1(2));
			auto dist2 = abs(euler_angles2(0)) + abs(euler_angles2(1)) + abs(euler_angles2(2));
			if (dist1 < dist2)
				cv::recoverPose(E, points_curr_conv, points_prev_conv, R, t, camera_param.focal_length,
					camera_param.pp, mask);
			else {R = R2; t = T;}
		}
		
		tracker->UpdateStateTrackerImage(preprocess_img_curr);
		if (tracker->NumPoints() < MIN_NUM_FEAT) 
			points_curr = detector->operator()(preprocess_img_curr);
		tracker->UpdateStateTrackerPoints(points_curr);
		
		double scale = 4.0;
		if ((scale > 0.1) && (t(2) > t(0)) && (t(2) > t(1))) {
			t_f = t_f + scale * (R_f*t);
			R_f = R * R_f;
		}

		DrawTrajectory(t_f, traj);
		if (cv::waitKey(1) == 27)
			break;
	}
	cv::destroyAllWindows();
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "printf: " << duration << '\n';
}
