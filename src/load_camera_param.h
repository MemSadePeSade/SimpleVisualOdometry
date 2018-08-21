#pragma once
#include <vector>
#include <string>
#include <iostream>

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

struct CameraParam {
	CameraParam() : pp(358.9874749825216, 201.7120939366421), intrisic_mat(1, 0, 0, 0, 1, 0, 0, 0, 1),
		dist_coeff(0, 0, 0, 0) {}
	cv::Matx<double, 1, 4> dist_coeff;
	cv::Matx33d intrisic_mat;
	double focal_length = 681.609;
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
	cv::FileStorage fs;
	fs.open(filename, cv::FileStorage::READ);
	if (!fs.isOpened())
	{
		std::cerr << "Failed to open " << filename << std::endl;
		return 1;
	}

	cv::FileNode n = fs["cam0"];
	cv::FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
	for (; it != it_end; ++it)
	{
		std::cout << (*it).name() << std::endl;
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