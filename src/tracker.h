#pragma once
#include<vector>
#include"opencv2/opencv.hpp"
#ifdef WITH_CUDA
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

namespace {
	enum Processor { CPU, CUDA };
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
		if (!singular) {
			x = atan2(R(2, 1), R(2, 2));
			y = atan2(-R(2, 0), sy);
			z = atan2(R(1, 0), R(0, 0));
		}
		else {
			x = atan2(-R(1, 2), R(1, 1));
			y = atan2(-R(2, 0), sy);
			z = 0;
		}
		return cv::Vec3d(x, y, z);
	}
    
	void download(const cv::cuda::GpuMat& d_mat, std::vector<cv::Point2f>& vec){
		vec.resize(d_mat.cols);
		cv::Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
		d_mat.download(mat);
	}
    
	inline void Converter(const std::vector<cv::Point2f>& input, std::vector<cv::Point2f>& output){
		output = input;
	};
	inline void Converter(const cv::cuda::GpuMat& input, std::vector<cv::Point2f>& output){
		download(input, output);
	};

}

namespace preprocess {
	struct PreProcess {
		virtual cv::Mat operator() (const cv::Mat& src) const = 0;
        #ifdef WITH_CUDA
		virtual cv::cuda::GpuMat operator() (const cv::Mat& src) const = 0;
        #endif
	};
	struct PreProcessCPU : PreProcess{
		PreProcessCPU(const CameraParam& camera_param_init) {
			camera_param = camera_param_init;
		}
		virtual cv::Mat operator() (const cv::Mat& src) const override{
			cv::Mat src_gray;
			cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
			cv::Mat dst;
			cv::undistort(src_gray, dst, camera_param.intrisic_mat, camera_param.dist_coeff);
			return dst;
		}
		CameraParam camera_param;
	};
#ifdef WITH_CUDA
	struct PreProcessGPU : PreProcess{
		PreProcessGPU(const CameraParam& camera_param_init) {
			camera_param = camera_param_init;
		}
		virtual cv::cuda::GpuMat operator() (const cv::Mat& src) const override{
			cv::Mat map1, map2;
			cv::gpu::GpuMat gpu_map1, gpu_map2;
			cv::gpu::GpuMat dst;
			cv::initUndistortRectifyMap(camera_param.intrisic_mat, camera_param.dist_coeff,
				cv::Mat::eye(3, 3, CV_32FC1), cv::Mat(), src.size(), CV_32FC1, map1, map2);
			gpu_map1.upload(map1);
			gpu_map2.upload(map2);
			cv::gpu::cvtColor(src, src, cv::COLOR_BGR2GRAY);
			cv::gpu::remap(src, dst, gpu_map1, gpu_map2, cv::INTER_LINEAR);
			return dst;
		}
		CameraParam camera_param;
	};
#endif
	PreProcess* InitPreprocessor(Processor processor, const CameraParam& camera_param) {
		if (processor == CPU)
			return new PreProcessCPU(camera_param);
		else
        #ifdef WITH_CUDA
			return new PreProcessGPU(camera_param);
        #endif
		return nullptr;
	}
};

namespace detector {
	enum class FeatureType {
		BRISK, ORB, MSER, FAST,
		AGAST, GFTT, KAZE, SURF,
		SIFT
	};

	struct FeatureDetector {
		virtual	std::vector<cv::Point2f> operator()(const cv::Mat& img) = 0;
#ifdef WITH_CUDA
		virtual	cv::cuda::GpuMat operator()(const cv::cuda::GpuMat& img) = 0;
#endif 
	};

	struct GFTTDetector {
		const int maxCorners = 300;
		const double qualityLevel = 0.01;
		const double minDistance = 20.0;
		const int blockSize = 3;
		const bool useHarrisDetector = false;
		const double k = 0.04;
	};
	struct FASTDetector {
		const int  fast_threshold = 20;
		const bool nonmaxSuppression = true;
		const int  type = 2;
	};

	struct GFTTCPUDetector :FeatureDetector, GFTTDetector {
		virtual std::vector<cv::Point2f> operator()(const cv::Mat& img) override {
			buffer.clear();
			cv::Mat mask;
			cv::goodFeaturesToTrack(img, buffer, maxCorners, qualityLevel, minDistance,
				mask, blockSize, useHarrisDetector, k);
			return buffer;
		}
		std::vector<cv::Point2f> buffer;
	};

	struct FASTCPUDetector : FeatureDetector, FASTDetector {
		virtual std::vector<cv::Point2f> operator()(const cv::Mat& img) override {
			buffer.clear();
			std::vector<cv::KeyPoint> keypoints;
			FAST(img, keypoints, fast_threshold, nonmaxSuppression, type);
			cv::KeyPoint::convert(keypoints, buffer, std::vector<int>());
		}
		std::vector<cv::Point2f> buffer;
	};
#ifdef WITH_CUDA
	struct GFTTGPUDetector :FeatureDetector, GFTTDetector {
		cv::cuda::GpuMat operator()(const cv::cuda::GpuMat& img) override {
			cv::Ptr<cv::cuda::CornersDetector> detector = cv::cuda::createGoodFeaturesToTrackDetector(gpu_img.type(),
				maxCorners, qualityLevel, minDistance);
			detector->detect(gpu_img, gpu_points);
			return gpu_points;
		}
		cv::cuda::GpuMat gpu_points;
	};
	struct FASTGPUDetector : FeatureDetector, FASTDetector {
		cv::cuda::GpuMat operator()(const cv::cuda::GpuMat& img) override {
			cv::Ptr<cv::cuda::FastFeatureDetector> gpuFastDetector = cv::cuda::FastFeatureDetector::create(fast_threshold,
				nonmaxSuppression, type);
			gpuFastDetector->detect(gpu_img, gpu_points);
			return gpu_points;
		}
		cuda::GpuMat gpu_points;
}
#endif 
		FeatureDetector* InitFeatureDetector(Processor processor,FeatureType feature_type){
			if (processor == CPU) {
				if (feature_type == detector::FeatureType::GFTT)
					return new detector::GFTTCPUDetector();
				else if (feature_type == detector::FeatureType::FAST)
					return new detector::FASTCPUDetector();
			}
			else{
            #ifdef WITH_CUDA
				if (feature_type == detector::FeatureType::GFTT)
					return new detector::GFTTGPUDetector();
				else if (feature_type == detector::FeatureType::FAST)
					return new detector::FASTGPUDetector();
				preprocess::PreProcessGPU preprocess(camera_param);
            #endif
			}
			return nullptr;
		};
}

	namespace tracker {
		struct Tracker {
			virtual std::vector<cv::Point2f> ToTrackandCorrectIndex(const cv::Mat& img) = 0;
			virtual int  NumPoints() = 0;
			virtual const std::vector<cv::Point2f>& GetPoints() = 0;
			virtual void UpdateStateTrackerImage(const cv::Mat& img) = 0;
			virtual void UpdateStateTrackerPoints(const std::vector<cv::Point2f>& points) = 0;
			cv::Size winSize = cv::Size(21, 21);
			int iters = 30;
			int maxLevel = 3;
			cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
		};

		class CPUTracker : public Tracker {
		public:
			CPUTracker() {};
			std::vector<cv::Point2f> ToTrackandCorrectIndex(const cv::Mat& img);// Track points and  delete points
			int  NumPoints() { return m_state.points.size(); };
			const std::vector<cv::Point2f>& GetPoints() { return m_state.points; };
			void UpdateStateTrackerImage(const cv::Mat& img) { m_state.img = img.clone(); };
			void UpdateStateTrackerPoints(const std::vector<cv::Point2f>& points) { m_state.points = points; };
		private:
			struct StateCPUTracker {
				cv::Mat img;
				std::vector<cv::Point2f> points;
			};
			StateCPUTracker m_state;
			//// buffer for local variables
			std::vector<cv::Point2f> buffer_points;
			std::vector<uchar> buffer_status;
			std::vector<float> buffer_err;
		};
#ifdef WITH_CUDA
		class GPUTracker : public Tracker {
		public:
			GPUTracker() {};
			cv::cuda::GpuMat ToTrackandCorrectIndex(const cv::cuda::GpuMat& img);// Track points and  delete points
			void UpdateStateTrackerImage(const cv::cuda::GpuMat& img) { m_state.img = img; };
			void UpdateStateTrackerPoints(const cv::cuda::GpuMat& points) { m_state.points = points; };
		private:
			struct StateGPUTracker {
				cv::cuda::GpuMat img;
				cv::cuda::GpuMat points;
			};
			StateGPUTracker m_state;
			// buffer for local variables
			cv::cuda::GpuMat buffer_img;
			cv::cuda::GpuMat buffer_points;
			cv::cuda::GpuMat buffer_status;
		    // buffer for correct index
			cv::cuda::GpuMat buffer_points_corrects1;
			cv::cuda::GpuMat buffer_points_corrects2;
	};
#endif
		Tracker* InitTracker(Processor processor) {
			if (processor == CPU)
				return new CPUTracker();
			else
            #ifdef WITH_CUDA
				return new GPUTracker();
            #endif
			return nullptr;
		}
	}
