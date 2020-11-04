#pragma once

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>

namespace {
std::ostream &operator<<(std::ostream &os, cv::KeyPoint const &keyPoint) {
  return os << "Point: " << keyPoint.pt << "px Diameter: " << keyPoint.size
            << "px ";
}
} // namespace

struct Position {
  double x{0};
  double y{0};
  double z{0};

  friend std::ostream &operator<<(std::ostream &os, Position const &position) {
    return os << "(" << position.x << ", " << position.y << ", " << position.z
              << ')';
  }
};

std::vector<cv::KeyPoint> detectMarkers(cv::InputArray const undistortedImage,
                                        bool showIntermediateImages);

void drawMarkers(cv::InputOutputArray image,
                 std::vector<cv::KeyPoint> const &markers);

Position toCameraPosition(cv::KeyPoint const &keyPoint, double focalLength,
                          double markerDiameter);
