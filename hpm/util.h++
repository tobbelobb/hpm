#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#endif
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#pragma GCC diagnostic pop

#include <hpm/types.h++>

void showImage(cv::InputArray image, std::string const &name);

void drawKeyPoints(cv::InputArray image,
                   std::vector<hpm::KeyPoint> const &keyPoints,
                   cv::InputOutputArray result);

cv::Mat imageWithKeyPoints(cv::InputArray image,
                           hpm::DetectionResult const &markers);

cv::Scalar ScalarBGR2HSV(cv::Scalar const &bgr);
