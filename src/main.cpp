#include<cmath>
#include<string>
#include<vector>
#include<algorithm>
#include<functional>
#include<numeric>    

#include <boost/filesystem/operations.hpp>
#include "opencv2/core/utility.hpp"

#include "load_camera_param.h"
#include "vo_features.h"
#include "draw.h"

#define MAX_FRAME 384
#define MIN_NUM_FEAT 2000

// Checks if a matrix is a valid rotation matrix.
bool isRotationMatrix(const cv::Matx33d& R) {
	cv::Matx33d Rt;
	cv::transpose(R, Rt);
	cv::Matx33d shouldBeIdentity = Rt * R;
	cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
	return  cv::norm(I, shouldBeIdentity) < 1e-6;
}

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
cv::Vec3d rotationMatrixToEulerAngles(const cv::Matx33d &R) {
	assert(isRotationMatrix(R));
	double sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));
    bool singular = sy < 1e-6; // If
	double x, y, z;
	if (!singular){
		x = atan2(R(2, 1), R(2, 2));
		y = atan2(-R(2, 0), sy);
		z = atan2(R(1, 0), R(0, 0));
	}
	else{
		x = atan2(-R(1, 2), R(1, 1));
		y = atan2(-R(2, 0), sy);
		z = 0;
	}
	return cv::Vec3d(x, y, z);
}

int main(int argc, const char* argv[]) {
	const FeatureType kFeatureType = FeatureType::GFTT;
	const char* keys =
		"{help h usage ?  |    | print help message }"
		"{@calib   |        | specify calib file }";
	cv::CommandLineParser cmd(argc, argv, keys);
	if (cmd.has("help") || !cmd.check())
	{
		cmd.printMessage();
		cmd.printErrors();
		return 0;
	}

	std::string calib_filename = cmd.get<std::string>("@calib");
	CameraParam camera_param;
	if (!boost::filesystem::exists(calib_filename))
		std::cout << "No calibrated file" << std::endl;
	else
		LoadCameraParam(calib_filename, camera_param);

	cv::Ptr<cv::FeatureDetector> featureDetector;
	if (!createDetectorDescriptorMatcher(kFeatureType, featureDetector))
		return -1;

	const char *img_path = "C:\\Users\\vponomarev\\Desktop\\01\\img_datasets\\e1i90v1a30_undistorted\\frame%06d.jpg";
	//const char *img_path = "C:\\Users\\vponomarev\\Desktop\\01\\img_datasets\\e1i90v1a30\\frame%06d.jpg";
	char filename[200];
	int  counter = 0;
	sprintf(filename, img_path, counter);

	cv::Mat img_prev = cv::imread(filename);
	if( img_prev.empty())
		return -1;
	
	cv::cvtColor(img_prev, img_prev, cv::COLOR_BGR2GRAY);
	cv::Mat img_prev_dst;
	cv::undistort(img_prev, img_prev_dst, camera_param.intrisic_mat, camera_param.dist_coeff);

	std::vector<cv::Point2f> points_prev;
	featureDetection(featureDetector, img_prev, points_prev, kFeatureType);//detect features in img1

	int keyFrame = 1;
	const double KeyFrThresh = 0.0;
	cv::Point2f point_keyfr(0, 0);

	cv::Mat traj = cv::Mat::zeros(600, 600, CV_8UC3);
	cv::Matx31d t_f(0, 0, 0);
	cv::Matx33d R_f(1, 0, 0, 0, 1, 0, 0, 0, 1);

	std::clock_t start;
	double duration;
	start = std::clock();
	// cycle through the directory
	for (counter = 1; counter < 384; ++counter) {
		sprintf(filename, img_path, counter);
		cv::Mat img_curr = cv::imread(filename);
		cv::cvtColor(img_curr, img_curr, cv::COLOR_BGR2GRAY);
		//cv::Mat img_curr_dst;
		//cv::undistort(img_curr, img_curr_dst, camera_param.intrisic_mat, camera_param.dist_coeff);

		// feature detection, tracking
		std::vector<cv::Point2f> points_curr;//vectors to store the coordinates of the feature points
		std::vector<uchar> status;
		featureTracking(img_prev, img_curr, points_prev, points_curr, status);

		cv::Point2f point_keyfr_curr = std::accumulate(points_curr.begin(), points_curr.end(),
			cv::Point2f(0, 0), std::plus<cv::Point2f>());
		point_keyfr_curr *= (1.0 / points_curr.size());
		double norm_movement = cv::norm(point_keyfr - point_keyfr_curr);

		cv::Matx31d t;
		cv::Matx33d R;
		cv::Mat E, mask;

		E = cv::findEssentialMat(points_curr, points_prev, camera_param.focal_length,
			camera_param.pp, cv::RANSAC, 0.999, 1.0, mask);

		// estimate R,T
		if (counter == 1) {
			cv::recoverPose(E, points_curr, points_prev, R, t, camera_param.focal_length,
				camera_param.pp, mask);
			R_f = R;
			t_f = t;
			img_prev = img_curr.clone();
			//img_prev_dst = img_curr_dst.clone();
			points_prev = points_curr;
			point_keyfr = point_keyfr_curr;
			continue;
		}
		else {
			if (norm_movement < KeyFrThresh) // if frame is not keyframe to skip
				continue;

			cv::Matx33d R1, R2;
			cv::Matx31d T;

			cv::decomposeEssentialMat(E, R1, R2, T);
			auto euler_angles1 = rotationMatrixToEulerAngles(R1);
			auto euler_angles2 = rotationMatrixToEulerAngles(R2);
			auto dist1 = abs(euler_angles1(0)) + abs(euler_angles1(1)) + abs(euler_angles1(2));
			auto dist2 = abs(euler_angles2(0)) + abs(euler_angles2(1)) + abs(euler_angles2(2));

			if (dist1 < dist2)
				cv::recoverPose(E, points_curr, points_prev, R, t, camera_param.focal_length,
					camera_param.pp, mask);
			else {
				R = R2;
				t = T;
			}
		}
		// a redetection is triggered in case the number of feautres being trakced go below a particular threshold
		if (points_prev.size() < MIN_NUM_FEAT) {
			featureDetection(featureDetector, img_prev, points_prev, FeatureType::GFTT);
			featureTracking(img_prev, img_curr, points_prev, points_curr, status);
		}

		double scale = 4.0;
		if ((scale > 0.1) && (t(2) > t(0)) && (t(2) > t(1))) {
			t_f = t_f + scale * (R_f*t);
			R_f = R * R_f;
		}

		img_prev = img_curr.clone();
		//img_prev_dst = img_curr_dst.clone();
		points_prev = points_curr;
		point_keyfr = point_keyfr_curr;

		DrawTrajectory(t_f, traj);

		if (cv::waitKey(1) == 27)
			break;
	}
	cv::destroyAllWindows();
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "printf: " << duration << '\n';
}
