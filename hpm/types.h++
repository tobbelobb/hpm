#pragma once

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>

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
using InputMarkerPositions = cv::Matx<double, 6, 3>;

struct SixDof {
  Vector3d rotation{0, 0, 0};
  Vector3d translation{0, 0, 0};
  double reprojectionError{0};

  double x() const { return translation(0); }
  double y() const { return translation(1); }
  double z() const { return translation(2); }
  double rotX() const { return rotation(0); }
  double rotY() const { return rotation(1); }
  double rotZ() const { return rotation(2); }

  friend std::ostream &operator<<(std::ostream &out, SixDof const &sixDof) {
    return out << sixDof.rotation << '\n'
               << sixDof.translation << '\n'
               << sixDof.reprojectionError;
  };
};

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
};

struct DetectionResult {
  std::vector<hpm::KeyPoint> red;
  std::vector<hpm::KeyPoint> green;
  std::vector<hpm::KeyPoint> blue;

  size_t size() const { return red.size() + green.size() + blue.size(); }
};

static inline auto signed2DCross(PixelPosition const &v0,
                                 PixelPosition const &v1,
                                 PixelPosition const &v2) {
  return (v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y);
}

static inline auto isLeft(PixelPosition const &v0, PixelPosition const &v1,
                          PixelPosition const &v2) -> bool {
  return signed2DCross(v0, v1, v2) > 0.0;
}

static void fanSort(std::vector<hpm::KeyPoint> &fan) {
  // Establish a reference point
  const auto &pivot = fan[0];
  // Sort points in a ccw radially ordered "fan" with pivot in fan[0]
  std::sort(
      std::next(fan.begin()), fan.end(),
      [&pivot](hpm::KeyPoint const &lhs, hpm::KeyPoint const &rhs) -> bool {
        return isLeft(pivot.center, lhs.center, rhs.center);
      });
}

struct IdentifiedHpMarks {
  std::optional<PixelPosition> red0;
  std::optional<PixelPosition> red1;
  std::optional<PixelPosition> green0;
  std::optional<PixelPosition> green1;
  std::optional<PixelPosition> blue0;
  std::optional<PixelPosition> blue1;

  explicit IdentifiedHpMarks(PixelPosition const &red0_,
                             PixelPosition const &red1_,
                             PixelPosition const &green0_,
                             PixelPosition const &green1_,
                             PixelPosition const &blue0_,
                             PixelPosition const &blue1_)
      : red0(red0_), red1(red1_), green0(green0_), green1(green1_),
        blue0(blue0_), blue1(blue1_) {}

  explicit IdentifiedHpMarks(std::array<PixelPosition, 6> const marks)
      : red0(marks[0]), red1(marks[1]), green0(marks[2]), green1(marks[3]),
        blue0(marks[4]), blue1(marks[5]) {}

  explicit IdentifiedHpMarks(DetectionResult const &foundMarkers) {
    auto const reds = foundMarkers.red.size();
    auto const greens = foundMarkers.green.size();
    auto const blues = foundMarkers.blue.size();

    std::vector<hpm::KeyPoint> all{};
    all.reserve(reds + blues + greens);

    all.insert(all.end(), foundMarkers.red.begin(), foundMarkers.red.end());
    all.insert(all.end(), foundMarkers.green.begin(), foundMarkers.green.end());
    all.insert(all.end(), foundMarkers.blue.begin(), foundMarkers.blue.end());

    if (all.size() < 6) {
      return;
    }

    // Points come out left handed from the detector
    // We temporarily don't want that while we're sorting
    for (auto &keyPoint : all) {
      keyPoint.center.y = -keyPoint.center.y;
    }
    // First element will be used as pivot
    if (not(isLeft(all[0].center, all[1].center, all[2].center))) {
      std::swap(all[0], all[1]);
    }
    fanSort(all);
    for (auto &keyPoint : all) {
      keyPoint.center.y = -keyPoint.center.y;
    }

    if (reds == 2) {
      red0 = {all[0].center};
      red1 = {all[1].center};
    }
    if (greens == 2) {
      green0 = {all[reds].center};
      green1 = {all[reds + 1].center};
    }
    auto const nonBlues = reds + greens;
    if (blues == 2) {
      blue0 = {all[nonBlues].center};
      blue1 = {all[nonBlues + 1].center};
    }
  }

  bool allIdentified() const {
    return red0.has_value() and red1.has_value() and green0.has_value() and
           green1.has_value() and blue0.has_value() and blue1.has_value();
  }

  friend std::ostream &operator<<(std::ostream &out,
                                  IdentifiedHpMarks const &identifiedHpMarks) {
    if (identifiedHpMarks.red0.has_value()) {
      out << identifiedHpMarks.red0.value();
    } else {
      out << '?';
    }
    out << '\n';
    if (identifiedHpMarks.red1.has_value()) {
      out << identifiedHpMarks.red1.value();
    } else {
      out << '?';
    }
    out << '\n';
    if (identifiedHpMarks.green0.has_value()) {
      out << identifiedHpMarks.green0.value();
    } else {
      out << '?';
    }
    out << '\n';
    if (identifiedHpMarks.green1.has_value()) {
      out << identifiedHpMarks.green1.value();
    } else {
      out << '?';
    }
    out << '\n';
    if (identifiedHpMarks.blue0.has_value()) {
      out << identifiedHpMarks.blue0.value();
    } else {
      out << '?';
    }
    out << '\n';
    if (identifiedHpMarks.blue1.has_value()) {
      out << identifiedHpMarks.blue1.value();
    } else {
      out << '?';
    }
    out << '\n';
    return out;
  };
};
} // namespace hpm
