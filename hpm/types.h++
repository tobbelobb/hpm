#pragma once

#include <iostream>

#include <opencv2/core.hpp>

using PixelPosition = cv::Point2f;

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

struct SixDof {
  cv::Mat rotation;
  cv::Mat translation;
  double reprojectionError;
  friend std::ostream &operator<<(std::ostream &out, SixDof const &sixDof) {
    return out << sixDof.rotation << '\n'
               << sixDof.translation << '\n'
               << sixDof.reprojectionError;
  };
};

