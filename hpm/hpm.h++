#pragma once

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>

namespace {
std::ostream &operator<<(std::ostream &os, cv::KeyPoint const &keyPoint) {
  return os << "Point: " << keyPoint.pt << "px Size: " << keyPoint.size
            << "px ";
}
} // namespace

using Position = cv::Point3d;

std::vector<cv::KeyPoint> detectMarkers(cv::InputArray const undistortedImage,
                                        bool showIntermediateImages);

void drawMarkers(cv::InputOutputArray image,
                 std::vector<cv::KeyPoint> const &markers);

Position toCameraPosition(cv::KeyPoint const &keyPoint, double focalLength,
                          cv::Point2f const &imageCenter,
                          cv::Size const &imageSize, double markerDiameter);
