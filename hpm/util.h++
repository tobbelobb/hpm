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

#include <hpm/detection-result.h++>
#include <hpm/simple-types.h++>

void showImage(cv::InputArray image, std::string const &name);

void drawKeyPoints(cv::InputOutputArray image,
                   std::vector<hpm::KeyPoint> const &keyPoints,
                   cv::Scalar const &color);

void drawDetectionResult(cv::InputOutputArray image,
                         hpm::DetectionResult const &markers);

cv::Mat imageWithDetectionResult(cv::InputArray image,
                                 hpm::DetectionResult const &detectionResult);

cv::Scalar ScalarBGR2HSV(cv::Scalar const &bgr);

// Units of sphereCenter and sphereRadius must be the same.
// Returned values will have the same units as focalLength has.
std::pair<double, double>
sphereToEllipseWidthHeight(hpm::CameraFramedPosition const &sphereCenter,
                           double focalLength, double sphereRadius);

// An alternative, geometric derivation
std::pair<double, double>
sphereToEllipseWidthHeight2(hpm::CameraFramedPosition const &sphereCenter,
                            double focalLength, double sphereRadius);
