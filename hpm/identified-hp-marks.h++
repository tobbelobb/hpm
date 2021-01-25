#pragma once

#include <algorithm>
#include <array>
#include <functional>

#include <hpm/detection-result.h++>
#include <hpm/simple-types.h++>

namespace hpm {

static inline auto signed2DCross(PixelPosition const &v0,
                                 PixelPosition const &v1,
                                 PixelPosition const &v2) {
  return (v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y);
}

static inline auto isRight(PixelPosition const &v0, PixelPosition const &v1,
                           PixelPosition const &v2) -> bool {
  return signed2DCross(v0, v1, v2) <= 0.0;
}

static void fanSort(std::vector<hpm::KeyPoint> &fan) {
  const auto &pivot = fan[0];
  std::sort(
      std::next(fan.begin()), fan.end(),
      [&pivot](hpm::KeyPoint const &lhs, hpm::KeyPoint const &rhs) -> bool {
        return isRight(pivot.m_center, lhs.m_center, rhs.m_center);
      });
}

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
                             PixelPosition const &imageCenter) {
    if (foundMarkers.red.size() != 2 or foundMarkers.green.size() != 2 or
        foundMarkers.blue.size() != 2) {
      return;
    }

    std::vector<hpm::KeyPoint> all{foundMarkers.getFlatCopy()};
    if (not(isRight(all[0].m_center, all[1].m_center, all[2].m_center))) {
      std::swap(all[0], all[1]);
    }
    fanSort(all);

    for (size_t i{0}; i < m_pixelPositions.size() and i < all.size(); ++i) {
      m_pixelPositions[i] = all[i].getCenterRay(markerR, f, imageCenter);
      m_identified[i] = true;
    }
  }

  // clang-format off
  [[nodiscard]] bool isIdentified(size_t idx) const {
    return idx < m_identified.size() and m_identified[idx];
  }

  [[nodiscard]] PixelPosition getPixelPosition(size_t idx) const {
    return m_pixelPositions[idx];
  }
  // clang-format on

  [[nodiscard]] bool allIdentified() const {
    return std::all_of(m_identified.begin(), m_identified.end(),
                       std::identity());
  }

  friend std::ostream &operator<<(std::ostream &out,
                                  IdentifiedHpMarks const &identifiedHpMarks) {
    for (size_t i{0}; i < identifiedHpMarks.m_pixelPositions.size(); ++i) {
      if (identifiedHpMarks.isIdentified(i)) {
        out << identifiedHpMarks.m_pixelPositions[i];
      } else {
        out << '?';
      }
      out << '\n';
    }
    return out;
  };
};
} // namespace hpm
