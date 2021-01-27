#pragma once

#include <hpm/detection-result.h++>
#include <hpm/simple-types.h++>

#include <algorithm>
#include <array>
#include <functional>

namespace hpm {

struct IdentifiedHpMarks {
  static size_t constexpr NUM_MARKERS{6};
  std::array<PixelPosition, NUM_MARKERS> m_pixelPositions{};
  std::array<bool, NUM_MARKERS> m_identified{false};

  explicit IdentifiedHpMarks(PixelPosition const &red0_,
                             PixelPosition const &red1_,
                             PixelPosition const &green0_,
                             PixelPosition const &green1_,
                             PixelPosition const &blue0_,
                             PixelPosition const &blue1_)
      : m_pixelPositions{red0_, red1_, green0_, green1_, blue0_, blue1_},
        m_identified{true, true, true, true, true, true} {}

  explicit IdentifiedHpMarks(
      std::array<PixelPosition, NUM_MARKERS> const positions_)
      : m_pixelPositions{positions_} {
    std::fill(m_identified.begin(), m_identified.end(), true);
  }

  explicit IdentifiedHpMarks(DetectionResult const &foundMarkers,
                             double const markerR, double const f,
                             PixelPosition const &imageCenter);

  [[nodiscard]] bool isIdentified(size_t idx) const;
  [[nodiscard]] PixelPosition getPixelPosition(size_t idx) const;
  [[nodiscard]] bool allIdentified() const;

  friend std::ostream &operator<<(std::ostream &out,
                                  IdentifiedHpMarks const &identifiedHpMarks);
};
} // namespace hpm
