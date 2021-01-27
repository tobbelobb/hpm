#pragma once

#include <vector>

#include <pipes/pipes.hpp>

#include <hpm/simple-types.h++>

namespace hpm {

double zFromSemiMinor(double markerR, double f, double semiMinor);

double centerRayFromZ(double c, double markerR, double z);

struct KeyPoint {
  PixelPosition m_center{0, 0};
  double m_major{0.0};
  double m_minor{0.0};
  double m_rot{0.0};

  explicit KeyPoint(PixelPosition const center, double major, double minor,
                    double rot)
      : m_center(center), m_major(major), m_minor(minor), m_rot(rot) {}

  explicit KeyPoint(cv::KeyPoint const &keyPointIn)
      : m_center(static_cast<PixelPosition>(keyPointIn.pt)),
        m_major(static_cast<double>(keyPointIn.size)), m_minor(m_major),
        m_rot(0.0) {}

  KeyPoint(PixelPosition const &center_, double size_)
      : m_center(center_), m_major(size_), m_minor(size_), m_rot(0.0) {}

  KeyPoint(mCircle const &circle)
      : m_center(circle.center), m_major(2.0 * circle.r), m_minor(m_major),
        m_rot(0.0) {}

  KeyPoint(mEllipse const &ellipse);

  cv::KeyPoint toCv() const {
    return {static_cast<cv::Point2f>(m_center),
            static_cast<float>(std::midpoint(m_major, m_minor))};
  }

  friend std::ostream &operator<<(std::ostream &out, KeyPoint const &keyPoint) {
    return out << keyPoint.m_center << ' ' << keyPoint.m_major << ' '
               << keyPoint.m_minor;
  };

  bool operator==(KeyPoint const &) const = default;

  PixelPosition getCenterRay(double const markerR, double const f,
                             PixelPosition const &imageCenter) const;
};

struct DetectionResult {
  std::vector<hpm::KeyPoint> red;
  std::vector<hpm::KeyPoint> green;
  std::vector<hpm::KeyPoint> blue;

  size_t size() const { return red.size() + green.size() + blue.size(); }

  std::vector<hpm::KeyPoint> getFlatCopy() const {
    std::vector<hpm::KeyPoint> all{};
    all.reserve(size());
    all.insert(all.end(), red.begin(), red.end());
    all.insert(all.end(), green.begin(), green.end());
    all.insert(all.end(), blue.begin(), blue.end());
    return all;
  }
};

} // namespace hpm
