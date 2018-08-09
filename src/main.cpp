#include<cmath>
#include<string>
#include <vector>
#include <algorithm>

#include <boost/filesystem/operations.hpp>
#include "vo_features.h"

struct CameraParam {
	CameraParam() : pp(607.1928, 185.2157),intrisic_mat(1, 0, 0, 0, 1, 0, 0, 0, 1) {}
	cv::Mat     dist_coeff ;
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

int LoadCameraParam(const std::string& filename, CameraParam& camera_param){
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
			camera_param.dist_coeff = cv::Mat(1, 5, CV_32F, data.data());
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
	char text[100];
	int fontFace = FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int thickness = 1;
	cv::Point textOrg(10, 50);
	
	std::string filename = "C://data//z.yaml";
	CameraParam camera_param;
	
	//LoadCameraParam(filename,camera_param);
	
	//cv::namedWindow("Road facing camera1", WINDOW_AUTOSIZE);// Create a window for display.
	cv::namedWindow("Road facing camera2", WINDOW_AUTOSIZE);// Create a window for display.
	cv::namedWindow("Trajectory", WINDOW_AUTOSIZE);// Create a window for display.

	//path p("C:\\data\\img");  
	path p("C:\\data\\kitti\\kitti\\data");
	directory_iterator end_itr;
	directory_iterator prev_itr(p);
	directory_iterator curr_itr(p); ++curr_itr;

	// cycle through the directory
	Ptr<FeatureDetector> featureDetector;
	if (!createDetectorDescriptorMatcher("ORB", featureDetector))
	   return -1;

	cv::Mat traj = Mat::zeros(600, 600, CV_8UC3);
	cv::Matx31d t_f(0,0,0);
	cv::Matx33d R_f(1, 0, 0, 0, 1, 0, 0, 0, 1);
	cv::Mat E_prev;
	for (prev_itr,curr_itr; curr_itr != end_itr; ++curr_itr,++prev_itr){
		Mat img_prev = imread(prev_itr->path().string());
		Mat img_curr = imread(curr_itr->path().string());
		cvtColor(img_prev, img_prev, COLOR_BGR2GRAY);
		cvtColor(img_curr, img_curr, COLOR_BGR2GRAY);
		
		// feature detection, tracking
		vector<Point2f> points1, points2;//vectors to store the coordinates of the feature points
		featureDetection(featureDetector, img_prev, points1, "NeFast");//detect features in img1
		vector<uchar> status;
		featureTracking(img_prev, img_curr, points1, points2, status);
		
		cv::Mat img2_keypoints;
		std::vector<cv::KeyPoint> draw_points2;
		draw_points2.resize(points2.size());
		std::for_each(draw_points2.begin(), draw_points2.end(), 
			[&points2, counter = 0](auto& it)  mutable
		    {
			     it.pt = points2[counter];
				 counter++;
		    });
		drawKeypoints(img_curr, draw_points2, img2_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		/*
		cv::Mat img1_keypoints;
		std::vector<cv::KeyPoint> draw_points1;
		draw_points1.resize(points1.size());
		std::for_each(draw_points1.begin(), draw_points1.end(),
			[&points1, counter = 0](auto& it)  mutable
		{
			it.pt = points1[counter];
			counter++;
		});
		drawKeypoints(img1, draw_points1, img1_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		*/
		
		cv::Matx31d t;
		cv::Matx33d R;
		Mat E, mask;
		E = cv::findEssentialMat(points1, points2, camera_param.focal_length, 
			                     camera_param.pp, RANSAC, 0.999, 1.0, mask);
		// undistortion
		if(!E_prev.empty())
		  cv::undistortPoints(points1, points1, E_prev, camera_param.dist_coeff);
		else
			E_prev = cv::Mat::eye(3, 3, CV_64F);
		cv::undistortPoints(points2, points2, E, camera_param.dist_coeff);
		E_prev *= E;
		// estimate R,T
		cv::recoverPose(E, points1, points2, R, t, camera_param.focal_length,
			            camera_param.pp, mask);
		
		double scale = 1.0;
		t_f = scale*t + scale * (R * t_f);

		int x = int(t_f(0)) + 300;
		int y = int(t_f(2)) + 100;
		circle(traj, Point(x, y), 1, CV_RGB(255, 0, 0), 2);

		rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);
		sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f(0), t_f(1), t_f(2));
		putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

		// assign current file name to current_file and echo it out to the console.
		string current_file = curr_itr->path().string();
		cout << current_file << endl;
		//cv::imshow("Road facing camera1", img1_keypoints);
		cv::imshow("Road facing camera2", img2_keypoints);
		cv::imshow("Trajectory", traj);
	
		if (cv::waitKey(0)  == 27)
		   break;
	}

	cv::destroyAllWindows();
}