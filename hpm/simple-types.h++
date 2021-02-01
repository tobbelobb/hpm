#pragma once

#include <hpm/ed/EDCircles.h++>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/core.hpp>
ENABLE_WARNINGS

#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace hpm {
using PixelPosition = cv::Point2d;

// using CameraFramedPosition = cv::Point3d;
class CameraFramedPosition : public cv::Point3d {
public:
  CameraFramedPosition(double x, double y, double z) : cv::Point3d(x, y, z){};
  CameraFramedPosition(cv::Point3d const &point) : cv::Point3d(point){};
};

class WorldPosition : public cv::Point3d {
public:
  WorldPosition(CameraFramedPosition const &cameraFramePosition,
                cv::Matx33d const &rotation, cv::Point3d const &translation)
      : cv::Point3d(rotation * cameraFramePosition + translation) {}
  WorldPosition(double x, double y, double z) : cv::Point3d(x, y, z){};
};

using Vector3d = cv::Vec3d;
size_t constexpr NUMBER_OF_MARKERS{6};
using ProvidedMarkerPositions = cv::Matx<double, NUMBER_OF_MARKERS, 3>;

} // namespace hpm
