#pragma once

#include <hpm/marks.h++>
#include <hpm/simple-types.h++>
#include <hpm/six-dof.h++>

#include <hpm/warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/core.hpp>
ENABLE_WARNINGS

#include <optional>

namespace hpm {
struct SolvePnpPoints {
  std::array<PixelPosition, NUMBER_OF_MARKERS> m_pixelPositions{};
  std::array<bool, NUMBER_OF_MARKERS> m_identified{false};

  explicit SolvePnpPoints(PixelPosition const &red0_,
                          PixelPosition const &red1_,
                          PixelPosition const &green0_,
                          PixelPosition const &green1_,
                          PixelPosition const &blue0_,
                          PixelPosition const &blue1_)
      : m_pixelPositions{red0_, red1_, green0_, green1_, blue0_, blue1_},
        m_identified{true, true, true, true, true, true} {}

  explicit SolvePnpPoints(
      std::array<PixelPosition, NUMBER_OF_MARKERS> const positions_)
      : m_pixelPositions{positions_} {
    std::fill(m_identified.begin(), m_identified.end(), true);
  }

  explicit SolvePnpPoints(
      std::vector<hpm::Ellipse> const &marks, double markerDiameter,
      double focalLength, PixelPosition const &imageCenter,
      MarkerType markerType,
      CameraFramedPosition const &expectedNormalDirection = {0.0, 0.0, 0.0});

  [[nodiscard]] bool isIdentified(size_t idx) const;
  [[nodiscard]] PixelPosition get(size_t idx) const;
  [[nodiscard]] bool allIdentified() const;

  friend std::ostream &operator<<(std::ostream &out,
                                  SolvePnpPoints const &points);
};
} // namespace hpm

std::optional<hpm::SixDof>
solvePnp(cv::InputArray cameraMatrix,
         cv::InputArray providedPositionsRelativeToNozzle,
         hpm::SolvePnpPoints const &points);
