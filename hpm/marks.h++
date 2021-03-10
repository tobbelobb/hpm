#pragma once

#include <hpm/ellipse.h++>
#include <hpm/simple-types.h++>

#include <vector>

namespace hpm {

struct Mark {
  Ellipse m_ellipse;

  Mark(Ellipse const e) : m_ellipse(e) {}
  Mark(PixelPosition const center, double major, double minor, double rot)
      : m_ellipse(center, major, minor, rot) {}
  Mark(PixelPosition const &center, double size) : m_ellipse(center, size) {}

  explicit Mark(cv::KeyPoint const &keyPointIn) : m_ellipse(keyPointIn) {}

  bool operator==(Mark const &) const = default;

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
  double identify(ProvidedMarkerPositions const &markPos,
                  double const focalLength, PixelPosition const &imageCenter,
                  double const markerDiameter, MarkerType markerType);
};

CameraFramedPosition toPosition(Mark const &mark, double focalLength,
                                hpm::PixelPosition const &imageCenter,
                                double markerDiameter, MarkerType markerType);
} // namespace hpm
