#pragma once
#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace {
	char text[100];
	int fontFace = cv::FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int thickness = 1;
	cv::Point textOrg(10, 50);

	void DrawOpticalFlow(std::vector<cv::Point2f>& points_prev,
		std::vector<cv::Point2f>& points_curr, cv::Mat& img_curr) {
		cv::Mat img_curr_keypoints;
		std::vector<cv::KeyPoint> draw_points_curr;
		draw_points_curr.resize(points_curr.size());
		std::for_each(draw_points_curr.begin(), draw_points_curr.end(),
			[&points_curr, counter = 0](auto& it)  mutable
		{
			it.pt = points_curr[counter];
			counter++;
		});
		cv::drawKeypoints(img_curr, draw_points_curr, img_curr_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
		cv::imshow("Img_curr", img_curr_keypoints);

		/// DrawOpticalFlow
		auto draw_img = img_curr.clone();
		int counter = 0;
		for (const auto& pt2 : points_prev) {
			auto pt1 = points_curr[counter];
			cv::line(draw_img, pt1, pt2, cv::Scalar(0, 125, 0));
			cv::circle(draw_img, pt2, 2, cv::Scalar(255, 0, 0), -1);
			cv::circle(draw_img, pt1, 2, cv::Scalar(0, 0, 255), -1);
			cv::imshow("TrackDraw", draw_img);
			counter++;
		}
	}

	void DrawTrajectory(const cv::Matx31d& t_f, const cv::Mat traj) {
		int x = int(t_f(0));
		int y = int(t_f(2));
		circle(traj, cv::Point(600 - (x / 1 + 200), y / 1 + 300), 1, CV_RGB(255, 0, 0), 1);
		rectangle(traj, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);
		sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f(0), t_f(1), t_f(2));
		putText(traj, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
		cv::imshow("Trajectory", traj);
	}
}