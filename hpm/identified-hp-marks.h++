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
        return isRight(pivot.center, lhs.center, rhs.center);
      });
}

struct IdentifiedHpMarks {
  std::array<PixelPosition, 6> m_pixelPositions{};
  std::array<bool, 6> m_identified{false};

  explicit IdentifiedHpMarks(PixelPosition const &red0_,
                             PixelPosition const &red1_,
                             PixelPosition const &green0_,
                             PixelPosition const &green1_,
                             PixelPosition const &blue0_,
                             PixelPosition const &blue1_)
      : m_pixelPositions{red0_, red1_, green0_, green1_, blue0_, blue1_},
        m_identified{true, true, true, true, true, true} {}

  explicit IdentifiedHpMarks(std::array<PixelPosition, 6> const positions_)
      : m_pixelPositions{positions_}, m_identified{true, true, true,
                                                   true, true, true} {}

  explicit IdentifiedHpMarks(DetectionResult const &foundMarkers) {
    if (foundMarkers.red.size() != 2 or foundMarkers.green.size() != 2 or
        foundMarkers.blue.size() != 2) {
      return;
    }

    std::vector<hpm::KeyPoint> all{foundMarkers.getFlatCopy()};

    if (not(isRight(all[0].center, all[1].center, all[2].center))) {
      std::swap(all[0], all[1]);
    }
    fanSort(all);

    m_pixelPositions = {all[0].center, all[1].center, all[2].center,
                        all[3].center, all[4].center, all[5].center};
    m_identified = {true, true, true, true, true, true};
  }

  bool allIdentified() const {
    return std::all_of(m_identified.begin(), m_identified.end(),
                       std::identity());
  }

  friend std::ostream &operator<<(std::ostream &out,
                                  IdentifiedHpMarks const &identifiedHpMarks) {
    // Pipes mux?
    for (size_t i{0}; i < identifiedHpMarks.m_pixelPositions.size(); ++i) {
      if (identifiedHpMarks.m_identified[i]) {
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
