#pragma once

#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#endif
#include <opencv2/opencv.hpp>
#pragma GCC diagnostic pop

using Segment = std::vector<cv::Point>;
using GradPix = short;
auto constexpr GRAD_PIX_CV_TYPE{CV_16SC1};

enum class EdgeDir { VERTICAL, HORIZONTAL, NONE };

int constexpr MAX_GRAD_VALUE{128 * 256};
double constexpr EPSILON{1.0};
size_t constexpr MIN_SEGMENT_LEN{10};
