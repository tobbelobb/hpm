#pragma once

#include <hpm/marks.h++>
#include <hpm/simple-types.h++>
#include <hpm/six-dof.h++>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/core.hpp>
ENABLE_WARNINGS

#include <optional>

namespace hpm {
struct SolvePnpPoints {
  static size_t constexpr NUM_MARKERS{6};
  std::array<PixelPosition, NUM_MARKERS> m_pixelPositions{};
  std::array<bool, NUM_MARKERS> m_identified{false};

  explicit SolvePnpPoints(PixelPosition const &red0_,
                          PixelPosition const &red1_,
                          PixelPosition const &green0_,
                          PixelPosition const &green1_,
                          PixelPosition const &blue0_,
                          PixelPosition const &blue1_)
      : m_pixelPositions{red0_, red1_, green0_, green1_, blue0_, blue1_},
        m_identified{true, true, true, true, true, true} {}

  explicit SolvePnpPoints(
      std::array<PixelPosition, NUM_MARKERS> const positions_)
      : m_pixelPositions{positions_} {
    std::fill(m_identified.begin(), m_identified.end(), true);
  }

  explicit SolvePnpPoints(std::vector<hpm::Ellipse> const &marks,
                          double markerDiameter, double focalLength,
                          PixelPosition const &imageCenter,
                          MarkerType markerType);

  [[nodiscard]] bool isIdentified(size_t idx) const;
  [[nodiscard]] PixelPosition get(size_t idx) const;
  [[nodiscard]] bool allIdentified() const;

  friend std::ostream &operator<<(std::ostream &out,
                                  SolvePnpPoints const &points);
};
} // namespace hpm

std::optional<hpm::SixDof>
solvePnp(cv::InputArray cameraMatrix,
         cv::InputArray markerPositionsRelativeToNozzle,
         hpm::SolvePnpPoints const &points);
