#pragma once

#include <hpm/ellipse.h++>
#include <hpm/simple-types.h++>

#include <vector>

namespace hpm {

struct Mark {
  enum class Type { SPHERE, DISK };

  Ellipse m_ellipse;
  Type m_type{Type::SPHERE};

  Mark(Ellipse const e) : m_ellipse(e) {}
  Mark(PixelPosition const center, double major, double minor, double rot)
      : m_ellipse(center, major, minor, rot) {}
  Mark(PixelPosition const &center, double size) : m_ellipse(center, size) {}

  explicit Mark(cv::KeyPoint const &keyPointIn) : m_ellipse(keyPointIn) {}

  bool operator==(Mark const &) const = default;

  hpm::CameraFramedPosition toPosition(double focalLength,
                                       hpm::PixelPosition const &imageCenter,
                                       double markerDiameter) const {
    return m_ellipse.toPosition(focalLength, imageCenter, markerDiameter);
  }

  PixelPosition getCenter() const { return m_ellipse.m_center; }
  PixelPosition getCenterRay(double const markerR, double const f,
                             PixelPosition const &imageCenter) const {
    return m_ellipse.getCenterRay(markerR, f, imageCenter);
  }
};

struct Marks {
  std::vector<Mark> m_marks;

  size_t size() const { return m_marks.size(); }
  std::vector<Mark> getCopy() const { return m_marks; }
  double identify(hpm::ProvidedMarkerPositions const &markPos,
                  double const focalLength,
                  hpm::PixelPosition const &imageCenter,
                  double const markerDiameter);
};
} // namespace hpm
