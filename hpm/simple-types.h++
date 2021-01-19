#pragma once

#include <iostream>
#include <vector>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/core.hpp>
ENABLE_WARNINGS

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
using ProvidedMarkerPositions = cv::Matx<double, 6, 3>;

struct KeyPoint {
  PixelPosition center{0, 0};
  double size{0.0};

  explicit KeyPoint(cv::KeyPoint const &keyPointIn)
      : center(static_cast<PixelPosition>(keyPointIn.pt)),
        size(static_cast<double>(keyPointIn.size)) {}
  KeyPoint(PixelPosition const &center_, double size_)
      : center(center_), size(size_) {}

  cv::KeyPoint toCv() const {
    return {static_cast<cv::Point2f>(center), static_cast<float>(size)};
  }

  friend std::ostream &operator<<(std::ostream &out, KeyPoint const &keyPoint) {
    return out << keyPoint.center << ' ' << keyPoint.size;
  };

  bool operator==(KeyPoint const &) const = default;
};

} // namespace hpm
