#pragma once

#include <hpm/ellipse.h++>
#include <hpm/simple-types.h++>

#include <pipes/pipes.hpp>

#include <vector>

namespace hpm {

struct Mark {
  Ellipse m_ellipse;
  double m_hue = -1.0;

  Mark(Ellipse const e, double hue_in) : m_ellipse(e), m_hue(hue_in) {}
  Mark(Ellipse const e) : m_ellipse(e) {}
  Mark(PixelPosition const center, double major, double minor, double rot)
      : m_ellipse(center, major, minor, rot) {}
  Mark(PixelPosition const &center, double size) : m_ellipse(center, size) {}

  // Need move/copy constructible for swap.
  // TODO: Remove this when sortCcw is removed.
  Mark(Mark const &) = default;

  explicit Mark(cv::KeyPoint const &keyPointIn) : m_ellipse(keyPointIn) {}

  // Trying to std::swap() two Marks. Need move assignment operator or copy
  // assignment operator?
  // TODO: Remove when sortCcw is removed.
  // Mark &operator=(Mark const &) = default;
  Mark &operator=(Mark &&other) noexcept {
    if (this != &other) {
      m_ellipse = other.m_ellipse;
      m_hue = other.m_hue;
    }
    return *this;
  }
  Mark &operator=(const hpm::Mark &) = default;

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
  double getSize() const { // TODO: remove when blob-detector is removed
    return m_ellipse.m_minor;
  }
};
static_assert(std::is_move_constructible<Mark>::value);
static_assert(std::is_move_assignable<Mark>::value);
static_assert(std::is_swappable<Mark>::value);

struct Marks {
  std::vector<Mark> m_red;
  std::vector<Mark> m_green;
  std::vector<Mark> m_blue;

  size_t size() const { return m_red.size() + m_green.size() + m_blue.size(); }
  std::vector<Mark> getFlatCopy() const;
  double fit(hpm::ProvidedMarkerPositions const &markPos,
             double const focalLength, hpm::PixelPosition const &imageCenter,
             double const markerDiameter);
};
} // namespace hpm
