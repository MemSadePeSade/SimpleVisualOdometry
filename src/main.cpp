#include<cmath>
#include<string>
#include<vector>
#include<algorithm>

#include <boost/filesystem/operations.hpp>
#include "vo_features.h"

struct CameraParam {
	CameraParam() : pp(607.1928, 185.2157), intrisic_mat(1, 0, 0, 0, 1, 0, 0, 0, 1),
		dist_coeff(1, 5, CV_32F, cv::Scalar(0))
	{
		pp = cv::Point2f(0, 0);
	}
	cv::Mat     dist_coeff;
	cv::Matx33f intrisic_mat;
	float focal_length = 718.8560;
	cv::Point2f pp;
};

void MakeIntrisicMatFromVector(CameraParam& camera_param,
	const std::vector<float>& parametrs) {
	camera_param.intrisic_mat(0, 0) = parametrs[0]; //fx
	camera_param.intrisic_mat(0, 2) = parametrs[2]; //cx
	camera_param.intrisic_mat(1, 1) = parametrs[1]; //fy
	camera_param.intrisic_mat(1, 2) = parametrs[3]; //cy
	camera_param.focal_length = std::sqrt(std::pow(2, parametrs[0]) + std::pow(2, parametrs[1]));
	camera_param.pp.x = parametrs[2];
	camera_param.pp.y = parametrs[3];
}

int LoadCameraParam(const std::string& filename, CameraParam& camera_param) {
	FileStorage fs;
	fs.open(filename, FileStorage::READ);
	if (!fs.isOpened())
	{
		cerr << "Failed to open " << filename << endl;
		return 1;
	}

	FileNode n = fs["cam0"];
	FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
	for (; it != it_end; ++it)
	{
		cout << (*it).name() << endl;
		if ((*it).name() == "distortion_coeffs") {
			std::vector<float> data;
			(*it) >> data;
			camera_param.dist_coeff = cv::Mat(1, 4, CV_64F, data.data());
			for (auto& elem : data)
				std::cout << elem << std::endl;
		}
		if ((*it).name() == "intrinsics") {
			std::vector<float> data;
			(*it) >> data;
			MakeIntrisicMatFromVector(camera_param, data);
			for (auto& elem : data)
				std::cout << elem << std::endl;
		}
		if ((*it).name() == "resolution") {
			std::vector<int> data;
			(*it) >> data;
			for (auto& elem : data)
				std::cout << elem << std::endl;
		}
	}
	fs.release();
	return 0;
}

using namespace boost::filesystem;
int main(int argc, char** argv) {
	std::string mode = "Track";

	char text[100];
	int fontFace = FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int thickness = 1;
	cv::Point textOrg(10, 50);

	std::string filename = "C://data//z.yaml";
	CameraParam camera_param;

	LoadCameraParam(filename, camera_param);

	path p("C:\\data\\img1");
	//path p("C:\\data\\kitti\\kitti\\data");
	directory_iterator end_itr;
	directory_iterator curr_itr(p);

	// cycle through the directory
	Ptr<FeatureDetector> featureDetector;
	if (!createDetectorDescriptorMatcher("ORB", featureDetector))
		return -1;

	cv::Mat img_prev = imread(curr_itr->path().string());
	cv::Mat track_draw = img_prev;
	cvtColor(img_prev, img_prev, COLOR_BGR2GRAY);
	cv::Mat  img_prev_dst;
	cv::undistort(img_prev, img_prev_dst, camera_param.intrisic_mat, camera_param.dist_coeff);
	vector<Point2f> points_prev;
	featureDetection(featureDetector, img_prev_dst, points_prev, "OOO");//detect features in img1

	cv::Mat traj = Mat::zeros(600, 600, CV_8UC3);
	cv::Matx31d t_f(0, 0, 0);
	cv::Matx33d R_f(1, 0, 0, 0, 1, 0, 0, 0, 1);
	cv::Mat E_prev;

	++curr_itr;
	for (curr_itr; curr_itr != end_itr; ++curr_itr) {
		cv::Mat  img_curr_dst;
		Mat img_curr = imread(curr_itr->path().string());
		cvtColor(img_curr, img_curr, COLOR_BGR2GRAY);
		cv::undistort(img_curr, img_curr_dst, camera_param.intrisic_mat, camera_param.dist_coeff);

		// feature detection, tracking
		vector<Point2f> points_curr;//vectors to store the coordinates of the feature points
		vector<uchar> status;
		featureTracking(img_prev_dst, img_curr_dst, points_prev, points_curr, status);
		cv::Mat img_curr_keypoints;
		std::vector<cv::KeyPoint> draw_points_curr;
		draw_points_curr.resize(points_curr.size());
		std::for_each(draw_points_curr.begin(), draw_points_curr.end(),
			[&points_curr, counter = 0](auto& it)  mutable
		{
			it.pt = points_curr[counter];
			counter++;
		});
		drawKeypoints(img_curr_dst, draw_points_curr, img_curr_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		cv::imshow("Img_curr", img_curr_keypoints);
		if (points_curr.empty()) {
			std::cout << "no point tracking" << std::endl;
			continue;
		}
		/// DrawOpticalFlow
		int counter = 0;
		for (const auto& pt2 : points_prev) {
			auto pt1 = points_curr[counter];
			cv::line(track_draw, pt1, pt2, Scalar(0, 125, 0));
			cv::circle(track_draw, pt1, 2, Scalar(0, 0, 0), -1);
			cv::imshow("TrackDraw", track_draw);
			counter++;
		}

		cv::Matx31d t;
		cv::Matx33d R;
		Mat E, mask;
		if (points_curr.size() < 25) {
			featureDetection(featureDetector, img_curr_dst, points_curr, "OOO");//detect features in img1
			points_prev = points_curr;
			continue;
		}
		if (points_curr.size() < 25) {
			featureDetection(featureDetector, img_curr_dst, points_curr, "OOO");//detect features in img1
			points_prev = points_curr;
			continue;
		}

		E = cv::findEssentialMat(points_prev, points_curr, camera_param.focal_length,
			camera_param.pp, RANSAC, 0.9, 1.0, mask);

		// estimate R,T
		cv::recoverPose(E, points_prev, points_curr, R, t, camera_param.focal_length,
			camera_param.pp, mask);

		if (mode != "Not_Track")
			points_prev = points_curr;
		else
			featureDetection(featureDetector, img_curr_dst, points_prev, "OOO");


		double scale = 1.0;
		t_f = scale * t + scale * (R * t_f);

		int x = int(t_f(0)) + 300;
		int y = int(t_f(2)) + 100;
		circle(traj, Point(x, y), 1, CV_RGB(255, 0, 0), 2);

		rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);
		sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f(0), t_f(1), t_f(2));
		putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
		cv::imshow("Trajectory", traj);

		// assign current file name to current_file and echo it out to the console.
		string current_file = curr_itr->path().string();
		cout << current_file << endl;

		if (cv::waitKey(0) == 27)
			break;
	}
	cv::destroyAllWindows();
}