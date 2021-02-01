#pragma once

#include <hpm/ellipse.h++>
#include <hpm/simple-types.h++>

#include <pipes/pipes.hpp>

#include <vector>

namespace hpm {

using Mark = hpm::Ellipse;

struct Marks {
  std::vector<Mark> red;
  std::vector<Mark> green;
  std::vector<Mark> blue;

  size_t size() const { return red.size() + green.size() + blue.size(); }
  std::vector<Mark> getFlatCopy() const;
  void filterByDistance(hpm::ProvidedMarkerPositions const &markPos,
                        double const focalLength,
                        hpm::PixelPosition const &imageCenter,
                        double const markerDiameter);
  void filterAndSortByDistance(hpm::ProvidedMarkerPositions const &markPos,
                               double const focalLength,
                               hpm::PixelPosition const &imageCenter,
                               double const markerDiameter);
};
} // namespace hpm
