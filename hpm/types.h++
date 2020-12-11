#pragma once

#include <iostream>
#include <vector>

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

using Vector3d = cv::Matx31d;
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

struct DetectionResult {
  std::vector<cv::KeyPoint> red;
  std::vector<cv::KeyPoint> green;
  std::vector<cv::KeyPoint> blue;

  size_t size() const { return red.size() + green.size() + blue.size(); }
};

static inline auto signed2DCross(PixelPosition const &v0,
                                 PixelPosition const &v1,
                                 PixelPosition const &v2) {
  return (v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y);
}

static inline auto isLeft(PixelPosition const &v0, PixelPosition const &v1,
                          PixelPosition const &v2) -> bool {
  return signed2DCross(v0, v1, v2) > 0.0F;
}

static void fanSort(std::vector<cv::KeyPoint> &fan) {
  // Establish a reference point
  const auto &pivot = fan[0];
  // Sort points in a ccw radially ordered "fan" with pivot in fan[0]
  std::sort(std::next(fan.begin()), fan.end(),
            [&pivot](cv::KeyPoint const &lhs, cv::KeyPoint const &rhs) -> bool {
              return isLeft(pivot.pt, lhs.pt, rhs.pt);
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

    std::vector<cv::KeyPoint> all{};
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
      keyPoint.pt.y = -keyPoint.pt.y;
    }
    // First element will be used as pivot
    if (not(isLeft(all[0].pt, all[1].pt, all[2].pt))) {
      std::swap(all[0], all[1]);
    }
    fanSort(all);
    for (auto &keyPoint : all) {
      keyPoint.pt.y = -keyPoint.pt.y;
    }

    if (reds == 2) {
      red0 = {all[0].pt};
      red1 = {all[1].pt};
    }
    if (greens == 2) {
      green0 = {all[reds].pt};
      green1 = {all[reds + 1].pt};
    }
    auto const nonBlues = reds + greens;
    if (blues == 2) {
      blue0 = {all[nonBlues].pt};
      blue1 = {all[nonBlues + 1].pt};
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
