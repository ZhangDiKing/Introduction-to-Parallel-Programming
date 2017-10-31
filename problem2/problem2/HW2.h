#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
extern cv::Mat imageInputRGBA;
extern cv::Mat imageOutputRGBA;

extern uchar4 *d_inputImageRGBA__;
extern uchar4 *d_outputImageRGBA__;

extern float *h_filter__;