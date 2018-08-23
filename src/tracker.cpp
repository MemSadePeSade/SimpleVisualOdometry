#include"tracker.h"

std::vector<cv::Point2f> tracker::CPUTracker::ToTrackandCorrectIndex(const cv::Mat& img) {
	calcOpticalFlowPyrLK(m_state.img, img, m_state.points, buffer_points, 
		                 buffer_status, buffer_err, winSize, 3, termcrit, 0, 0.001);
	int indexCorrection = 0;
	for (int i = 0; i < buffer_status.size(); i++) {
		cv::Point2d pt = buffer_points.at(i - indexCorrection);
		if ((buffer_status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)) {
			m_state.points.erase(m_state.points.begin() + (i - indexCorrection));
			buffer_points.erase(buffer_points.begin() + (i - indexCorrection));
			indexCorrection++;
		}
	}
	return buffer_points;
}

#ifdef WITH_CUDA
cv::cuda::GpuMat tracker::GPUTracker::ToTrackandCorrectIndex(const cv::cuda::GpuMat& img) {
	cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse =
			cv::cuda::SparsePyrLKOpticalFlow::create(winSize, maxLevel, iters);
	d_pyrLK_sparse->calc(m_state.img, img, m_state.points, buffer_points, buffer_status);
	
	int index = 0;
	for( int i = 0; i < buffer_status.cols;++i)
		if (buffer_status.at(i) == 1) {
			index = i;
			break;
		}
	if (index == 0)
		return buffer_points.reshape(0, 0);

	// Correct Index
	
	int indexCorrection = 0;
	cv::Point2d pt2 = buffer_points.at(index);
	cv::Point2d pt1 = m_state.points.at(index);
	for (int i = 0; i < buffer_status.size(); i++) {
		if ((buffer_status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)) {
		    buffer_points.at(i)  = pt2;
			m_state.points.at(i) = pt1;	
		}
	}
	return buffer_points;
}
#endif