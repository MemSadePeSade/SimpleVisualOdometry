#include<cmath>
#include<string>
#include<vector>
#include<algorithm>

#include <boost/filesystem/operations.hpp>
#include "vo_features.h"

#define MAX_FRAME 384
#define MIN_NUM_FEAT 2000

// Checks if a matrix is a valid rotation matrix.
bool isRotationMatrix(cv::Matx33d& R){
	cv::Matx33d Rt;
	transpose(R, Rt);
	cv::Matx33d shouldBeIdentity = Rt * R;
	Mat I = Mat::eye(3, 3, CV_64F);
	return  norm(I, shouldBeIdentity) < 1e-6;
}

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
Vec3f rotationMatrixToEulerAngles(cv::Matx33d &R){
	assert(isRotationMatrix(R));

	float sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));

	bool singular = sy < 1e-6; // If

	float x, y, z;
	if (!singular)
	{
		x = atan2(R(2, 1), R(2, 2));
		y = atan2(-R(2, 0), sy);
		z = atan2(R(1, 0), R(0, 0));
	}
	else
	{
		x = atan2(-R(1, 2), R(1, 1));
		y = atan2(-R(2, 0), sy);
		z = 0;
	}
	return Vec3f(x, y, z);
}

struct CameraParam {
	CameraParam() : pp(358.9874749825216, 201.7120939366421), intrisic_mat(1, 0, 0, 0, 1, 0, 0, 0, 1),
		dist_coeff(0,0,0,0){}
	cv::Matx<double,1,4> dist_coeff;
	cv::Matx33d intrisic_mat;
    float focal_length = 681.609;
	cv::Point2d pp;
};

void MakeIntrisicMatFromVector(CameraParam& camera_param,
	const std::vector<double>& parametrs) {
	camera_param.intrisic_mat(0, 0) = parametrs[0]; //fx
	camera_param.intrisic_mat(0, 2) = parametrs[2]; //cx
	camera_param.intrisic_mat(1, 1) = parametrs[1]; //fy
	camera_param.intrisic_mat(1, 2) = parametrs[3]; //cy
	camera_param.focal_length = 681.609;//cv::norm(cv::Point2d(parametrs[0], parametrs[1]));
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
			std::vector<double> data;
			(*it) >> data;
			camera_param.dist_coeff = cv::Mat(1, 4, CV_64F, data.data());
			for (auto& elem : data)
				std::cout << elem << std::endl;
		}
		if ((*it).name() == "intrinsics") {
			std::vector<double> data;
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

	cv::Mat traj = Mat::zeros(1200, 1200, CV_8UC3);
	cv::Matx31d t_f(0, 0, 0);
	cv::Matx33d R_f(1, 0, 0, 0, 1, 0, 0, 0, 1);

	++curr_itr;
	int counter = 0;
	for (counter,curr_itr; curr_itr != end_itr; ++curr_itr, ++counter) {
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
		auto draw_img = img_curr_dst.clone();
		int counter = 0;
		for (const auto& pt2 : points_prev) {
			auto pt1 = points_curr[counter];
			cv::line(draw_img, pt1, pt2, Scalar(0, 125, 0));
			cv::circle(draw_img, pt2, 2, Scalar(255, 0, 0), -1);
			cv::circle(draw_img, pt1, 2, Scalar(0, 0, 255), -1);
			cv::imshow("TrackDraw", draw_img);
			counter++;
		}
		
		cv::Matx31d t;
		cv::Matx33d R;
		Mat E, mask;
		
		if (points_curr.size() < 25) {
			featureDetection(featureDetector, img_curr_dst, points_curr, "OOO");//detect features in img1
			points_prev = points_curr;
			img_prev_dst = img_curr_dst.clone();
			continue;
		}
		if (points_curr.size() < 25) {
			featureDetection(featureDetector, img_curr_dst, points_curr, "OOO");//detect features in img1
			points_prev = points_curr;
			img_prev_dst = img_curr_dst.clone();
			continue;
		}
		
		E = cv::findEssentialMat(points_curr, points_prev, camera_param.focal_length,
			camera_param.pp, RANSAC, 0.9, 1.0, mask);

		// estimate R,T
		if (counter == 0)
			cv::recoverPose(E, points_curr, points_prev, R, t, camera_param.focal_length,
				camera_param.pp, mask);
		else {
			cv::Matx33d R1, R2;
			cv::decomposeEssentialMat(E, R1, R2, t);
			auto euler_angles1 = rotationMatrixToEulerAngles(R1);
			//std::cout << euler_angles1 << std::endl;
			auto euler_angles2 = rotationMatrixToEulerAngles(R2);
			//std::cout << euler_angles2 << std::endl;

			auto dist1 = abs(euler_angles1(0))+ abs(euler_angles1(1))+ abs(euler_angles1(2));
			auto dist2 = abs(euler_angles2(0)) + abs(euler_angles2(1)) + abs(euler_angles2(2));
			if (dist1 < dist2) 
				cv::recoverPose(E, points_curr, points_prev, R, t, camera_param.focal_length,
					camera_param.pp, mask); //R = R1;
			else
				R = R2;
		}
		
		// a redetection is triggered in case the number of feautres being trakced go below a particular threshold
		if (points_prev.size() < MIN_NUM_FEAT) {
			//cout << "Number of tracked features reduced to " << prevFeatures.size() << endl;
			//cout << "trigerring redection" << endl;
			featureDetection(featureDetector, img_prev_dst, points_prev, "OOO");
			featureTracking(img_prev_dst, img_curr_dst, points_prev, points_curr, status);
		}
		
		double scale = 4.0;
		
		if ((scale>0.1) && (t(2) > t(0)) && (t(2) > t(1))) {
			t_f = t_f + scale * (R_f*t);
			R_f = R * R_f;
		}
		else {}
		
		img_prev_dst = img_curr_dst.clone();
		points_prev = points_curr;
		
		int x = int(t_f(0)) + 400;
		int y = int(t_f(2)) + 400;
		circle(traj, Point(800 - (x / 1 + 200), y / 1 + 300), 1, CV_RGB(255, 0, 0), 1);

		rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);
		sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f(0), t_f(1), t_f(2));
		putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
		cv::imshow("Trajectory", traj);

		// assign current file name to current_file and echo it out to the console.
		string current_file = curr_itr->path().string();
		cout << current_file << endl;

		if (cv::waitKey(1) == 27)
			break;
	}
	cv::destroyAllWindows();
}
