#pragma once

#include <hpm/simple-types.h++>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/core.hpp>
ENABLE_WARNINGS

double zFromSemiMinor(double markerR, double f, double semiMinor);

double centerRayFromZ(double c, double markerR, double z);

namespace hpm {
struct Ellipse {
  PixelPosition m_center{0, 0};
  double m_major{0.0};
  double m_minor{0.0};
  double m_rot{0.0};

  Ellipse(Ellipse const &ellipse) = default;

  Ellipse(PixelPosition const center, double major, double minor, double rot)
      : m_center(center), m_major(major), m_minor(minor), m_rot(rot) {}

  Ellipse(cv::KeyPoint const &keyPointIn)
      : m_center(static_cast<PixelPosition>(keyPointIn.pt)),
        m_major(static_cast<double>(keyPointIn.size)), m_minor(m_major),
        m_rot(0.0) {}

  Ellipse(PixelPosition const &center_, double size_)
      : m_center(center_), m_major(size_), m_minor(size_), m_rot(0.0) {}

  Ellipse(mCircle const &circle)
      : m_center(circle.center),
        m_major(2.0 * (circle.r + (sqrt(3) * circle.err) + 0.5)),
        m_minor(2.0 * (circle.r - (sqrt(3) * circle.err) + 0.5)), m_rot(0.0) {}

  Ellipse(mEllipse const &ellipse);

  cv::KeyPoint toCvKeyPoint() const {
    return {static_cast<cv::Point2f>(m_center),
            static_cast<float>(std::midpoint(m_major, m_minor))};
  }

  friend std::ostream &operator<<(std::ostream &out, Ellipse const &ellipse) {
    return out << ellipse.m_center << ' ' << ellipse.m_major << ' '
               << ellipse.m_minor;
  };

  bool operator==(Ellipse const &) const = default;

  PixelPosition getCenterRay(double const markerR, double const f,
                             PixelPosition const &imageCenter) const;
};

CameraFramedPosition toPosition(Ellipse const &ellipse, double focalLength,
                                PixelPosition const &imageCenter,
                                double markerDiameter, MarkerType markerType);

} // namespace hpm
