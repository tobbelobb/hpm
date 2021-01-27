#pragma once

#include <hpm/detection-result.h++>
#include <hpm/identified-hp-marks.h++>
#include <hpm/simple-types.h++>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
ENABLE_WARNINGS

void showImage(cv::InputArray image, std::string const &name);

void drawKeyPoints(cv::InputOutputArray image,
                   std::vector<hpm::KeyPoint> const &keyPoints,
                   cv::Scalar const &color);

void draw(cv::InputOutputArray image, hpm::DetectionResult const &markers);
void draw(cv::InputOutputArray image, hpm::IdentifiedHpMarks const &markers);

cv::Mat imageWith(cv::InputArray image,
                  hpm::IdentifiedHpMarks const &identifiedHpMarks);
cv::Mat imageWith(cv::InputArray image,
                  hpm::DetectionResult const &detectionResult);

cv::Scalar ScalarBGR2HSV(cv::Scalar const &bgr);

cv::Mat getSingleChannelCopy(cv::InputArray image, int channel);

cv::Mat getRedCopy(cv::InputArray image);

cv::Mat getGreenCopy(cv::InputArray image);

cv::Mat getBlueCopy(cv::InputArray image);

cv::Mat getValueChannelCopy(cv::InputArray image);

cv::Mat getSaturationChannelCopy(cv::InputArray image);

cv::Mat getHueChannelCopy(cv::InputArray image);

cv::Mat invertedCopy(cv::InputArray image);

struct EllipseProjection {
  double width;
  double height;
  double xt;
  double yt;
};

// Units of sphereCenter and sphereRadius must be the same.
// Returned values will have the same units as focalLength has.
EllipseProjection
sphereToEllipseWidthHeight(hpm::CameraFramedPosition const &sphereCenter,
                           double focalLength, double sphereRadius);

// An alternative, geometric derivation
std::pair<double, double>
sphereToEllipseWidthHeight2(hpm::CameraFramedPosition const &sphereCenter,
                            double focalLength, double sphereRadius);
