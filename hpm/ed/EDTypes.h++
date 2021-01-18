#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

using Segment = std::vector<cv::Point>;
using GradPix = short;
auto constexpr GRAD_PIX_CV_TYPE{CV_16SC1};

enum class EdgeDir { VERTICAL, HORIZONTAL, NONE };

int constexpr MAX_GRAD_VALUE{128 * 256};
double constexpr EPSILON{1.0};
size_t constexpr MIN_SEGMENT_LEN{10};
